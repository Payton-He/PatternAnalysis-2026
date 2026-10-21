"""Evaluate a trained VAE and run inference on the test set.

    python predict.py --checkpoint models/vae_best.pt
    python predict.py --checkpoint models/vae_best.pt --reader_study 50

Writes figures and metrics.json to results/<checkpoint name>/. Run from the
project root so the relative data/ path used by dataset.py resolves.

Checks performed:
    0. train/validate/test split leakage (patient cases shared between splits)
    1. mode collapse   - diversity of generated vs real images, active latent dims
    2. plausibility    - outlier ranking of generated images against real data
    3. memorization    - nearest-neighbour distance to train vs test images
    4. inference       - reconstructions of unseen test images, train/test gap
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # duplicate libiomp5md.dll workaround (Windows)

import argparse
import json
import random
import re

import matplotlib
matplotlib.use("Agg")                      # write figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np                         # numpy is allowed here (not in modules.py)
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataset import get_dataloaders, get_file_list
from modules import VAE


# ------------------------------------------------------------- loading
def load_vae(path, device):
    """Rebuild a VAE from a checkpoint written by train.py and put it in eval mode.

    The checkpoint stores the constructor settings, so no command-line
    arguments are needed to recreate the same architecture.

    Args:
        path: path to vae_best.pt.
        device: torch.device to load onto.

    Returns:
        The VAE, in eval mode (BatchNorm uses its running statistics).
    """
    ck = torch.load(path, map_location=device)
    assert ck["model"] == "vae", f"{path} is a '{ck['model']}' checkpoint, not a VAE"
    model = VAE(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {path} (epoch {ck['epoch']}, val loss {ck['val_loss']:.2f})")
    return model.eval()


def collect(loader, max_n, seed=0):
    """Gather a loader's images into one (N, 1, H, W) CPU tensor, randomly capped at max_n.

    Args:
        loader: a DataLoader yielding (B, 1, H, W) batches.
        max_n: maximum number of images to keep.
        seed: seed for the random subset, so reruns use the same images.
    """
    x = torch.cat([b for b in loader])
    if len(x) > max_n:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(len(x), generator=g)[:max_n]]
    return x


@torch.no_grad()
def generate(model, n, bs=64):
    """Generate n images by decoding z ~ N(0, I). Returns a (n, 1, H, W) CPU tensor in [0, 1]."""
    return torch.cat([model.sample(min(bs, n - i)).cpu() for i in range(0, n, bs)])


# ------------------------------------------------------------- metrics
def ssim(x, y, win=7):
    """Structural similarity for each image pair in two (B, 1, H, W) batches in [0, 1].

    Uses a uniform window (a simple version of SSIM). 1 = identical, 0 = unrelated.

    Returns:
        (B,) tensor with one SSIM value per pair.
    """
    c1, c2, p = 0.01 ** 2, 0.03 ** 2, win // 2
    pool = lambda t: F.avg_pool2d(t, win, 1, p, count_include_pad=False)
    mx, my = pool(x), pool(y)
    sxx, syy, sxy = pool(x * x) - mx ** 2, pool(y * y) - my ** 2, pool(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def pairwise_ssim(x, n_pairs=2000, seed=0):
    """Mean SSIM between randomly chosen pairs of different images in x.

    A diversity measure: high = images look alike, low = images vary.
    """
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(x), (n_pairs,), generator=g)
    j = torch.randint(0, len(x), (n_pairs,), generator=g)
    keep = i != j
    return ssim(x[i[keep]], x[j[keep]]).mean().item()


def nn_distance(query, ref, bs=128):
    """Distance from each query image to its closest reference image (pixel space).

    Args:
        query: (N, 1, H, W) tensor.
        ref: (M, 1, H, W) tensor.

    Returns:
        dists: (N,) Euclidean distance to the nearest reference image.
        idxs: (N,) index into `ref` of that nearest image.
    """
    q, r = query.flatten(1), ref.flatten(1)
    dists, idxs = [], []
    for i in range(0, len(q), bs):
        m = torch.cdist(q[i:i + bs], r).min(1)
        dists.append(m.values); idxs.append(m.indices)
    return torch.cat(dists), torch.cat(idxs)


def pca_fit(x, k=50):
    """Fit PCA to real images. Returns (mean, top-k components, variance of each component)."""
    X = x.flatten(1).numpy().astype(np.float32)
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, len(S))
    return mean, Vt[:k], (S[:k] ** 2) / max(len(X) - 1, 1) + 1e-8


def pca_score(x, pca):
    """Mahalanobis distance of each image in PCA space (large = unusual compared with real data)."""
    mean, V, var = pca
    p = (x.flatten(1).numpy() - mean) @ V.T
    return ((p ** 2) / var).sum(1)


def summarize(a):
    """Mean, median and 95th percentile of an array, as plain floats for JSON."""
    a = np.asarray(a, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


# --------------------------------------------------------- VAE-specific
@torch.no_grad()
def reconstruct(model, x, device, bs=128):
    """Deterministic reconstruction: decode the encoder's mean (no random sampling)."""
    out = []
    for i in range(0, len(x), bs):
        mu, _ = model.encode(x[i:i + bs].to(device))
        out.append(model.decode(mu).cpu())
    return torch.cat(out)


def recon_stats(model, x, device):
    """Mean squared error per pixel and mean SSIM between images and their reconstructions."""
    r = reconstruct(model, x, device)
    return ((r - x) ** 2).mean().item(), ssim(r, x).mean().item()


@torch.no_grad()
def kl_per_dim(model, x, device, bs=128):
    """Average KL divergence of each latent dimension over a set of images. Returns (latent_dim,)."""
    mus, lvs = [], []
    for i in range(0, len(x), bs):
        mu, lv = model.encode(x[i:i + bs].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    mu, lv = torch.cat(mus), torch.cat(lvs)
    return (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean(0)


@torch.no_grad()
def interpolation_grid(model, device, rows=8, steps=8):
    """Decode straight-line paths between random latent points. Returns (rows*steps, 1, H, W)."""
    z0 = torch.randn(rows, model.latent_dim)
    z1 = torch.randn(rows, model.latent_dim)
    t = torch.linspace(0, 1, steps)
    z = z0[:, None] * (1 - t)[None, :, None] + z1[:, None] * t[None, :, None]   # (rows, steps, D)
    return model.decode(z.reshape(-1, model.latent_dim).to(device)).cpu()


def plot_kl_per_dim(kl, path):
    """Bar chart of per-dimension KL, sorted high to low. Bars near zero are unused ('collapsed') dimensions."""
    kl = np.sort(kl.numpy())[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(range(len(kl)), kl, color="#2a78d6", width=0.8)
    ax.axhline(0.01, color="#898781", lw=1, ls="--")          # threshold used for "active"
    ax.set_xlabel("latent dimension (sorted)", color="#52514e")
    ax.set_ylabel("mean KL (nats)", color="#52514e")
    ax.set_title("Information carried by each latent dimension", loc="left", color="#0b0b0b", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ------------------------------------------------------------- other checks
def check_split_leakage():
    """Return the case IDs that appear in more than one split (all three lists should be empty)."""
    def cases(split):
        return {m.group(1) for f in get_file_list(f"keras_slices_{split}")
                if (m := re.search(r"case_(\d+)", f))}
    tr, va, te = cases("train"), cases("validate"), cases("test")
    return {"train&validate": sorted(tr & va), "train&test": sorted(tr & te), "validate&test": sorted(va & te)}


def reader_study(real, fake, n, out_dir, seed=0):
    """Export n real and n generated slices as shuffled, anonymised PNGs plus an answer key (key.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    items = [("real", real[i]) for i in rng.sample(range(len(real)), min(n, len(real)))]
    items += [("fake", fake[i]) for i in rng.sample(range(len(fake)), min(n, len(fake)))]
    rng.shuffle(items)
    with open(os.path.join(out_dir, "key.csv"), "w") as f:
        f.write("file,label\n")
        for k, (label, img) in enumerate(items):
            name = f"{k:03d}.png"
            plt.imsave(os.path.join(out_dir, name), img[0].numpy(), cmap="gray", vmin=0, vmax=1)
            f.write(f"{name},{label}\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to vae_best.pt")
    ap.add_argument("--n_samples", type=int, default=2000, help="images to generate")
    ap.add_argument("--max_real", type=int, default=2000, help="cap on real train/test images used")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--reader_study", type=int, default=0, help="export N real + N fake blinded PNGs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(args.out, os.path.splitext(os.path.basename(args.checkpoint))[0])
    os.makedirs(out, exist_ok=True)

    model = load_vae(args.checkpoint, device)
    train_loader, _, test_loader = get_dataloaders(batch_size=args.batch_size)
    train_x = collect(train_loader, args.max_real, args.seed)
    test_x = collect(test_loader, args.max_real, args.seed)
    fake = generate(model, args.n_samples)
    print(f"train {tuple(train_x.shape)}, test {tuple(test_x.shape)}, generated {tuple(fake.shape)}")

    metrics = {"model": "vae", "checkpoint": args.checkpoint}
    save_image(fake[:64], os.path.join(out, "samples.png"), nrow=8)
    save_image(interpolation_grid(model, device), os.path.join(out, "interpolation.png"), nrow=8)

    # 0. split leakage
    metrics["split_overlap_by_case"] = check_split_leakage()

    # 1. mode collapse
    kl = kl_per_dim(model, test_x, device)
    plot_kl_per_dim(kl, os.path.join(out, "kl_per_dim.png"))
    metrics["mode_collapse"] = {
        "pairwise_ssim_real": pairwise_ssim(test_x, seed=args.seed),
        "pairwise_ssim_generated": pairwise_ssim(fake, seed=args.seed),
        "active_latent_dims": f"{int((kl > 0.01).sum())}/{kl.numel()}",
        "note": "generated SSIM far above real = low diversity; few active dims = posterior collapse",
    }
    # Reconstructions are as blurry as samples but as varied as real data, so this is the fair baseline
    recon_test = reconstruct(model, test_x, device)
    metrics["mode_collapse"]["pairwise_ssim_reconstructions"] = pairwise_ssim(recon_test, seed=args.seed)

    # 2. anatomical plausibility: outlier ranking against the real training distribution
    pca = pca_fit(train_x)
    s_test, s_fake = pca_score(test_x, pca), pca_score(fake, pca)
    metrics["plausibility"] = {
        "mahalanobis_real_test": summarize(s_test),
        "mahalanobis_generated": summarize(s_fake),
        "frac_generated_above_real_p99": float((s_fake > np.percentile(s_test, 99)).mean()),
        "note": "outlier ranking only; inspect worst_generated.png and use a reader study for real evidence",
    }
    worst = torch.from_numpy(np.argsort(-s_fake)[:16].copy())
    save_image(fake[worst], os.path.join(out, "worst_generated.png"), nrow=8)

    # 3. memorization: nearest neighbour in train vs test (equal-sized reference sets)
    n = min(len(train_x), len(test_x), len(fake))
    d_train, idx_train = nn_distance(fake[:n], train_x[:n])
    d_test, _ = nn_distance(fake[:n], test_x[:n])
    d_base, _ = nn_distance(test_x[:n], train_x[:n])      # how close real test images are to train
    metrics["memorization"] = {
        "nn_dist_generated_to_train": summarize(d_train.numpy()),
        "nn_dist_generated_to_test": summarize(d_test.numpy()),
        "nn_dist_test_to_train_baseline": summarize(d_base.numpy()),
        "ratio_gen_train_over_gen_test": float(d_train.mean() / d_test.mean()),
        "note": "ratio clearly below 1 suggests copying of training data",
    }
    closest = torch.argsort(d_train)[:16]
    save_image(torch.cat([fake[closest], train_x[:n][idx_train[closest]]], dim=3),   # fake | nearest train
               os.path.join(out, "closest_to_train.png"), nrow=4)

    # 4. inference on unseen test images + overfitting check
    tr_mse, tr_ssim = recon_stats(model, train_x, device)
    te_mse, te_ssim = recon_stats(model, test_x, device)
    metrics["inference"] = {
        "recon_mse_per_pixel_train": tr_mse, "recon_mse_per_pixel_test": te_mse,
        "recon_ssim_train": tr_ssim, "recon_ssim_test": te_ssim,
        "recon_psnr_test_db": float(10 * np.log10(1 / te_mse)),
        "note": "a large train/test gap indicates overfitting",
    }
    k = min(8, len(test_x))
    save_image(torch.cat([test_x[:k], reconstruct(model, test_x[:k], device)]),      # top: real, bottom: reconstruction
               os.path.join(out, "test_reconstructions.png"), nrow=k)

    if args.reader_study > 0:
        reader_study(test_x, fake, args.reader_study, os.path.join(out, "reader_study"), args.seed)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nFigures and metrics saved to {out}/")


if __name__ == "__main__":
    main()