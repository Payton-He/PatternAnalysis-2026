"""Evaluate a trained model on the test set: the baseline VAE, or the VQ-VAE + code prior.

    python predict.py vae   --checkpoint models/vae_best.pt
    python predict.py vae   --checkpoint models/vae_best.pt --reader_study 50
    python predict.py vqvae --vqvae models/vqvae_run1/vqvae_best.pt --prior models/prior_run1/prior_best.pt \
                            --vae models/vae_best.pt --settings 1.0:0,0.8:50

Writes figures and metrics.json to results/<name>/ (the checkpoint name for `vae`, the VQ-VAE run
folder name for `vqvae`). Run from the project root so the relative data/ path used by dataset.py
resolves. Helpers used by both modes (SSIM, PCA, split-leakage check, reader study) are defined once.

`vae` mode checks:
    0. split leakage   - patient cases shared between train/validate/test
    1. mode collapse   - diversity of generated vs real images, active latent dims
    2. plausibility    - outlier ranking of generated images against real data
    3. memorization    - nearest-neighbour distance to train vs test images
    4. inference       - reconstructions of unseen test images, train/test gap

`vqvae` mode checks:
    0. split leakage         - patient cases shared between train/validate/test
    1. reconstruction        - SSIM / PSNR of every test slice through encode -> quantize -> decode,
                               against the VAE baseline (this is the SSIM > 0.6 target)
    2. codebook health       - how many codes are used, and how evenly (the discrete mode-collapse check)
    3. diversity             - pairwise SSIM of generated images against real ones, per sampling setting
    4. distribution fidelity - FID-style Frechet distance, precision and coverage in a PCA feature space
    5. plausibility          - outlier ranking of generated images; the worst ones are the failure autopsy
    6. memorization          - nearest training image (whole training set) for every generated image
    7. interpolation         - blends between a generated image and its closest training scan
    8. profile               - parameters, training time, peak GPU memory, generation speed
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # duplicate libiomp5md.dll workaround (Windows)

import argparse
import json
import random
import re
import time

import matplotlib
matplotlib.use("Agg")                      # write figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np                         # numpy is allowed here (not in modules.py)
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataset import get_dataloaders, get_file_list
from modules import VAE, VQVAE, CodePrior


# ------------------------------------------------------------- progress
_T0 = time.perf_counter()


def log(msg):
    """Print a message with the seconds elapsed since the script started, so slow stages are visible."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def free_gpu():
    """Return cached but unused GPU memory to the driver, so later steps do not spill into slow shared memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



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


def load_vqvae(path, device):
    """Rebuild a VQ-VAE from a checkpoint written by train.py and put it in eval mode."""
    ck = torch.load(path, map_location=device)
    assert ck["model"] == "vqvae", f"{path} is a '{ck['model']}' checkpoint, not a VQ-VAE"
    model = VQVAE(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {path} (epoch {ck['epoch']}, val SSIM {ck['val_ssim']:.3f})")
    return model.eval()


def load_prior(path, device):
    """Rebuild the code prior from a checkpoint written by train.py and put it in eval mode."""
    ck = torch.load(path, map_location=device)
    assert ck["model"] == "prior", f"{path} is a '{ck['model']}' checkpoint, not a prior"
    model = CodePrior(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {path} (epoch {ck['epoch']}, val cross-entropy {ck['val_loss']:.3f})")
    return model.eval()


def n_params(model):
    """Number of parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def read_json(path):
    """Read a JSON file written by train.py, or return {} if it does not exist."""
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}



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


# ------------------------------------------------- metrics shared by both modes
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


# ------------------------------------------------------------ VAE helpers
@torch.no_grad()
def generate_vae(model, n, bs=64):
    """Generate n images by decoding z ~ N(0, I). Returns a (n, 1, H, W) CPU tensor in [0, 1]."""
    return torch.cat([model.sample(min(bs, n - i)).cpu() for i in range(0, n, bs)])


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


@torch.no_grad()
def reconstruct_vae(model, x, device, bs=128):
    """Deterministic reconstruction: decode the encoder's mean (no random sampling)."""
    out = []
    for i in range(0, len(x), bs):
        mu, _ = model.encode(x[i:i + bs].to(device))
        out.append(model.decode(mu).cpu())
    return torch.cat(out)


def recon_stats(model, x, device):
    """Mean squared error per pixel and mean SSIM between images and their reconstructions."""
    r = reconstruct_vae(model, x, device)
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


# ------------------------------------------------------- VQ-VAE helpers
@torch.no_grad()
def reconstruct_vq(model, x, device, bs=64):
    """Direct reconstruction (encode -> quantize -> decode) of a CPU tensor. Returns a CPU tensor."""
    return torch.cat([model(x[i:i + bs].to(device))[0].cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def encode_codes(model, x, device, bs=64):
    """Code grids (N, H/8, W/8) of a CPU tensor of images, on the CPU."""
    return torch.cat([model.encode_indices(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def generate_vq(vq, prior, n, temperature, top_k, device, bs=32, seed=0):
    """Sample n code grids from the prior and decode them. Returns (images, grids) as CPU tensors.

    A fixed seed makes the images identical on every run, so the sample grids are uncurated
    but reproducible.
    """
    torch.manual_seed(seed)
    imgs, grids = [], []
    for i in range(0, n, bs):
        g = prior.sample(min(bs, n - i), temperature=temperature, top_k=top_k or None)
        grids.append(g.cpu())
        imgs.append(vq.decode_indices(g).cpu())
    return torch.cat(imgs), torch.cat(grids)


@torch.no_grad()
def interpolate_codes(vq, idx_a, idx_b, steps=8):
    """Decode straight-line blends between the code embeddings of two images.

    The code grids are turned into their embedding vectors, mixed linearly, and decoded WITHOUT
    snapping back to the codebook, so the path between the two images is smooth.

    Args:
        idx_a, idx_b: (H, W) long code grids on the model's device.

    Returns:
        (steps, C, H, W) images; the first is image a, the last is image b.
    """
    za, zb = vq.vq.lookup(idx_a[None]), vq.vq.lookup(idx_b[None])        # (1, D, h, w)
    t = torch.linspace(0, 1, steps, device=za.device).view(-1, 1, 1, 1)
    return vq.decoder((1 - t) * za + t * zb)


# ------------------------------------------------------------- metrics
def nn_search(query, ref, device, chunk=2048):
    """Closest reference image to every query image, in pixel space, over the WHOLE reference set.

    The reference is moved to the GPU in chunks, so even the full 11 000-image training set fits.

    Args:
        query: (N, 1, H, W) CPU tensor.
        ref: (M, 1, H, W) CPU tensor.

    Returns:
        dists: (N,) Euclidean distance to the nearest reference image.
        idxs: (N,) index into `ref` of that image.
    """
    q = query.flatten(1).to(device)
    best_d = torch.full((len(q),), float("inf"), device=device)
    best_i = torch.zeros(len(q), dtype=torch.long, device=device)
    with torch.no_grad():
        for j in range(0, len(ref), chunk):
            d, i = torch.cdist(q, ref[j:j + chunk].flatten(1).to(device)).min(1)
            better = d < best_d
            best_d = torch.where(better, d, best_d)
            best_i = torch.where(better, i + j, best_i)
    result = best_d.cpu(), best_i.cpu()
    free_gpu()
    return result


def pca_project(x, pca):
    """Project images onto the PCA basis from pca_fit. Returns (N, k) float32 array."""
    mean, V, _ = pca
    return ((x.flatten(1).numpy() - mean) @ V.T).astype(np.float32)


def frechet_distance(a, b):
    """Frechet distance between Gaussians fitted to two feature sets (the formula behind FID).

    Here the features are PCA coefficients of the images, not Inception features, so values are
    NOT comparable with published FID numbers; compare only against the baselines in metrics.json.

    Args:
        a, b: (N, d) arrays.
    """
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvals(c1 @ c2).real, 0, None)).sum()   # trace of sqrt(c1 c2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr_sqrt)


def precision_coverage(real, fake, k=5):
    """k-nearest-neighbour precision and coverage of a generated feature set against a real one.

    Each real point gets a ball whose radius is the distance to its k-th nearest real neighbour.
        precision: fraction of generated points inside some real ball (are they realistic?)
        coverage:  fraction of real balls that contain at least one generated point (is the
                   real data covered? low coverage means missing modes)

    Args:
        real, fake: (N, d) float arrays.
    """
    r, f = torch.from_numpy(real), torch.from_numpy(fake)
    radii = torch.cdist(r, r).topk(k + 1, largest=False).values[:, -1]   # k-th neighbour (0th is the point itself)
    inside = torch.cdist(r, f) <= radii[:, None]                         # (N_real, N_fake)
    return inside.any(0).float().mean().item(), inside.any(1).float().mean().item()


def feature_metrics(real_f, fake_f):
    """Frechet distance, precision and coverage of a generated feature set against real features."""
    p, c = precision_coverage(real_f, fake_f)
    return {"frechet_pca": frechet_distance(real_f, fake_f), "precision": p, "coverage": c}


def psnr_db(a, b):
    """PSNR in dB between two image tensors in [0, 1] (computed from the mean squared error)."""
    return float(10 * np.log10(1.0 / ((a - b) ** 2).mean().item()))


# ------------------------------------------------------------- plotting
def _style(ax, title, xlabel, ylabel):
    """Shared chart look: left-aligned title, horizontal grid only, no top/right spines."""
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    ax.set_xlabel(xlabel, color="#52514e", fontsize=9)
    ax.set_ylabel(ylabel, color="#52514e", fontsize=9)
    ax.tick_params(colors="#898781", labelsize=9, length=0)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def plot_code_usage(counts, path):
    """Bar chart of how often each codebook entry is used by the training images, sorted high to low.

    A long flat tail near zero means many unused codes (codebook collapse); an even profile is healthy.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#fcfcfb")
    ax.bar(range(len(counts)), np.sort(counts)[::-1], color="#2a78d6", width=1.0)
    ax.set_yscale("log")
    _style(ax, "Codebook usage over the training images", "code (sorted by use)", "times used (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_ssim_hist(scores, path, target=0.6, baseline=None):
    """Histogram of per-image test reconstruction SSIM with the target (and VAE mean) marked."""
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#fcfcfb")
    ax.hist(scores, bins=40, color="#2a78d6")
    ax.axvline(target, color="#52514e", ls="--", lw=1.2, label=f"target {target}")
    if baseline is not None:
        ax.axvline(baseline, color="#eb6834", ls=":", lw=1.5, label="VAE mean")
    ax.legend(frameon=False, labelcolor="#52514e", fontsize=9)
    _style(ax, "Reconstruction SSIM of every test slice", "SSIM", "number of slices")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ----------------------------------------------------------- VAE evaluation
def evaluate_vae(args, device):
    """Evaluate the baseline VAE given by --checkpoint.

    Writes samples.png, interpolation.png, kl_per_dim.png, worst_generated.png, closest_to_train.png,
    test_reconstructions.png and metrics.json to results/<checkpoint name>/.
    """
    out = os.path.join(args.out, os.path.splitext(os.path.basename(args.checkpoint))[0])
    os.makedirs(out, exist_ok=True)

    model = load_vae(args.checkpoint, device)
    train_loader, _, test_loader = get_dataloaders(batch_size=args.batch_size)
    train_x = collect(train_loader, args.max_real, args.seed)
    test_x = collect(test_loader, args.max_real, args.seed)
    fake = generate_vae(model, args.n_samples)
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
    recon_test = reconstruct_vae(model, test_x, device)
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
    save_image(torch.cat([test_x[:k], reconstruct_vae(model, test_x[:k], device)]),      # top: real, bottom: reconstruction
               os.path.join(out, "test_reconstructions.png"), nrow=k)

    if args.reader_study > 0:
        reader_study(test_x, fake, args.reader_study, os.path.join(out, "reader_study"), args.seed)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nFigures and metrics saved to {out}/")


# -------------------------------------------------------- VQ-VAE evaluation
def evaluate_vqvae(args, device):
    """Evaluate the VQ-VAE + code prior given by --vqvae and --prior (checks 0-8 in the module docstring).

    Writes the figures and metrics.json to results/<VQ-VAE run folder name>/.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_name = os.path.basename(os.path.dirname(args.vqvae)) or "run"
    out = os.path.join(args.out, run_name)
    os.makedirs(out, exist_ok=True)
    settings = [(float(t), int(k)) for t, k in (s.split(":") for s in args.settings.split(","))]

    vq = load_vqvae(args.vqvae, device)
    prior = load_prior(args.prior, device)
    vae = load_vae(args.vae, device) if args.vae else None

    train_loader, _, test_loader = get_dataloaders(batch_size=args.batch_size)
    train_all = collect(train_loader, 10 ** 9, args.seed)                 # the WHOLE training set (memorization audit)
    test_x = collect(test_loader, 10 ** 9, args.seed)                     # the WHOLE test set (SSIM target)
    g = torch.Generator().manual_seed(args.seed)
    train_x = train_all[torch.randperm(len(train_all), generator=g)[:args.max_real]]   # random subset for PCA and diversity
    print(f"train {tuple(train_all.shape)} (subset {len(train_x)}), test {tuple(test_x.shape)}")
    log("data loaded")

    metrics = {"model": "vqvae+prior", "vqvae": args.vqvae, "prior": args.prior, "vae": args.vae}

    # 0. split leakage
    metrics["split_overlap_by_case"] = check_split_leakage()

    log("1. reconstruction")
    # 1. reconstruction: the SSIM > 0.6 target, on every unseen test slice
    recon = reconstruct_vq(vq, test_x, device)
    s_test = ssim(recon, test_x)
    recon_tr = reconstruct_vq(vq, train_x, device)
    metrics["reconstruction"] = {
        "ssim_test_mean": s_test.mean().item(),
        "ssim_test_summary": summarize(s_test.numpy()),
        "ssim_test_min": s_test.min().item(),
        "fraction_test_above_0.6": (s_test > 0.6).float().mean().item(),
        "ssim_train_mean": ssim(recon_tr, train_x).mean().item(),
        "psnr_test_db": psnr_db(recon, test_x),
        "note": "SSIM uses a uniform 7x7 window; each test slice is compared with its own reconstruction",
    }
    vae_ssim = None
    if vae is not None:
        vr = reconstruct_vae(vae, test_x, device)
        vae_ssim = ssim(vr, test_x).mean().item()
        metrics["reconstruction"]["vae_baseline_ssim_test_mean"] = vae_ssim
        metrics["reconstruction"]["vae_baseline_psnr_test_db"] = psnr_db(vr, test_x)
    plot_ssim_hist(s_test.numpy(), os.path.join(out, "test_ssim_hist.png"), baseline=vae_ssim)
    k = 8
    sel = torch.linspace(0, len(test_x) - 1, k).long()                    # spread over patients and slice levels
    save_image(torch.cat([test_x[sel], recon[sel]]), os.path.join(out, "test_reconstructions.png"), nrow=k)
    worst_r = torch.argsort(s_test)[:k]                                   # the k worst reconstructions
    save_image(torch.cat([test_x[worst_r], recon[worst_r]]), os.path.join(out, "worst_reconstructions.png"), nrow=k)

    log("2. codebook health")
    # 2. codebook health (the discrete form of the mode-collapse check)
    train_codes = encode_codes(vq, train_all, device)
    counts = torch.bincount(train_codes.flatten(), minlength=vq.num_codes).numpy().astype(np.float64)
    p = counts / counts.sum()
    plot_code_usage(counts, os.path.join(out, "code_usage.png"))
    metrics["codebook"] = {
        "codes_used_by_training_images": f"{int((counts > 0).sum())}/{vq.num_codes}",
        "perplexity_training_images": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
        "most_used_code_share": float(p.max()),
        "note": "perplexity near the codebook size = all codes used evenly; a low value signals codebook collapse",
    }

    log("fitting PCA (SVD of the real training subset; can take a minute)")
    # PCA feature space fitted on real training images, used by checks 4 and 5
    # The real reference is a random subset of TRAINING slices (many patients, and neighbouring slices of
    # one patient are rarely both picked). The test set cannot serve as the reference: it holds only a few
    # patients whose near-identical neighbouring slices make the k-NN balls tiny, so coverage reads 0 for everything.
    n_ref = min(500, len(train_x) // 4)
    n_pick = min(args.n_samples, len(train_x) // 4)
    pca = pca_fit(train_x[n_ref + n_pick:])                               # fitted on training images outside the reference
    real_f = pca_project(train_x[:n_ref], pca)
    rng = np.random.RandomState(args.seed)
    test_pick = torch.from_numpy(rng.permutation(len(test_x))[:args.n_samples])
    baseline = feature_metrics(real_f, pca_project(test_x[test_pick], pca))   # real UNSEEN images scored as if generated
    floor = feature_metrics(real_f, pca_project(train_x[n_ref:n_ref + n_pick], pca))   # other real TRAINING slices: the noise floor
    s_pca_test = pca_score(test_x, pca)
    worst_real = torch.from_numpy(np.argsort(-s_pca_test)[:16].copy())     # most outlying REAL test slices: the control
    save_image(test_x[worst_real], os.path.join(out, "worst_real_test.png"), nrow=8)

    metrics["settings"] = {}
    gen_images = {}
    closest_for_interp = None
    for si, (temperature, top_k) in enumerate(settings):
        name = f"T{temperature}_k{top_k}"
        free_gpu()
        log(f"sampling {args.n_samples} images at temperature {temperature}, top_k {top_k or 'off'}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fake, grids = generate_vq(vq, prior, args.n_samples, temperature, top_k, device, args.batch_size, args.seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
        log(f"    generated {len(fake)} images in {gen_time:.0f}s")
        gen_images[name] = fake
        save_image(fake[:64], os.path.join(out, f"samples_{name}.png"), nrow=8)     # uncurated: the first 64, fixed seed
        m = {"temperature": temperature, "top_k": top_k, "images_per_second": len(fake) / gen_time}

        # 3. diversity / mode collapse
        sample_counts = torch.bincount(grids.flatten(), minlength=vq.num_codes)
        m["diversity"] = {
            "pairwise_ssim_real_test": pairwise_ssim(test_x, seed=args.seed),
            "pairwise_ssim_generated": pairwise_ssim(fake, seed=args.seed),
            "pairwise_ssim_reconstructions": pairwise_ssim(recon, seed=args.seed),
            "distinct_codes_in_samples": f"{int((sample_counts > 0).sum())}/{vq.num_codes}",
            "note": "generated far above real = low diversity; temperature below 1 lowers diversity",
        }

        # 4. distribution fidelity in PCA feature space
        fake_f = pca_project(fake, pca)
        m["distribution"] = {**feature_metrics(real_f, fake_f), "floor_other_real_training_slices": floor, "baseline_real_test_as_generated": baseline,
                             "note": "PCA-50 features, not Inception; reference = random training slices. floor = other real training slices (best a generator can do); baseline = real unseen patients (differs from training by patient)"}

        # 5. anatomical plausibility: outlier ranking against the real training distribution
        s_fake = pca_score(fake, pca)
        m["plausibility"] = {
            "mahalanobis_real_test": summarize(s_pca_test),
            "mahalanobis_generated": summarize(s_fake),
            "frac_generated_above_real_p99": float((s_fake > np.percentile(s_pca_test, 99)).mean()),
            "note": "outlier ranking only; inspect worst_generated_*.png (the failure autopsy) by eye",
        }
        worst = torch.from_numpy(np.argsort(-s_fake)[:16].copy())
        save_image(fake[worst], os.path.join(out, f"worst_generated_{name}.png"), nrow=8)
        m["plausibility"]["worst_generated_indices"] = [int(i) for i in worst]

        log("    diversity, distribution and plausibility done; memorization audit (nearest training images)")
        # 6. memorization: nearest image in the WHOLE training set, then SSIM of that pair
        d_train, i_train = nn_search(fake, train_all, device)
        ssim_nn = ssim(fake, train_all[i_train])
        sub = torch.from_numpy(rng.permutation(len(train_all))[:len(test_x)])      # equal-sized train and test references
        d_gen_tr, _ = nn_search(fake, train_all[sub], device)
        d_gen_te, _ = nn_search(fake, test_x, device)
        m["memorization"] = {
            "nn_dist_generated_to_train": summarize(d_train.numpy()),
            "ratio_gen_train_over_gen_test_equal_sized_refs": float(d_gen_tr.mean() / d_gen_te.mean()),
            "nn_ssim_generated_to_train": summarize(ssim_nn.numpy()),
            "frac_generated_with_nn_ssim_above_0.6": (ssim_nn > 0.6).float().mean().item(),
            "note": "ratio clearly below 1, or many pairs above SSIM 0.6, suggests copying; compare with the test baseline",
        }
        closest = torch.argsort(d_train)[:16]
        save_image(torch.cat([fake[closest], train_all[i_train[closest]]], dim=3),   # generated | nearest training scan
                   os.path.join(out, f"closest_to_train_{name}.png"), nrow=4)
        if si == 0:
            closest_for_interp = (grids, i_train, torch.argsort(ssim_nn, descending=True), fake)
            metrics["_first_setting"] = name
        metrics["settings"][name] = m
        print(f"    {len(fake) / gen_time:.1f} images/s | diversity {m['diversity']['pairwise_ssim_generated']:.3f} "
              f"(real {m['diversity']['pairwise_ssim_real_test']:.3f}) | coverage {m['distribution']['coverage']:.2f} "
              f"| NN-SSIM>0.6: {m['memorization']['frac_generated_with_nn_ssim_above_0.6']:.3f}")

    # memorization baseline: how close are REAL unseen test images to the training set?
    d_t, i_t = nn_search(test_x, train_all, device)
    ssim_t = ssim(test_x, train_all[i_t])
    metrics["memorization_baseline_test_to_train"] = {
        "nn_dist": summarize(d_t.numpy()), "nn_ssim": summarize(ssim_t.numpy()),
        "frac_with_nn_ssim_above_0.6": (ssim_t > 0.6).float().mean().item(),
        "note": "real images from unseen patients; generated images should look no closer to training than this",
    }

    log("7. interpolation")
    # 7. interpolation between generated images and their closest training scans
    name0 = metrics.pop("_first_setting")
    grids0, i_train0, order, fake0 = closest_for_interp
    rows = []
    for j in order[:4].tolist():                                          # the 4 generated images most similar to training
        a = grids0[j].to(device)
        b = vq.encode_indices(train_all[i_train0[j]][None].to(device))[0]
        rows.append(interpolate_codes(vq, a, b).cpu())
    for j in range(4):                                                    # plus 4 uncurated generated -> training pairs
        a = grids0[j].to(device)
        b = vq.encode_indices(train_all[i_train0[j]][None].to(device))[0]
        rows.append(interpolate_codes(vq, a, b).cpu())
    save_image(torch.cat(rows), os.path.join(out, f"interpolation_{name0}.png"), nrow=8)   # row: generated -> nearest training scan

    log("VAE baseline samples")
    # baseline: VAE samples scored the same way
    if vae is not None:
        vfake = generate_vae(vae, args.n_samples)
        vf = pca_project(vfake, pca)
        save_image(vfake[:64], os.path.join(out, "vae_samples.png"), nrow=8)
        metrics["vae_baseline_samples"] = {
            "pairwise_ssim_generated": pairwise_ssim(vfake, seed=args.seed),
            "distribution": feature_metrics(real_f, vf),
            "frac_generated_with_nn_ssim_above_0.6":
                (ssim(vfake, train_all[nn_search(vfake, train_all, device)[1]]) > 0.6).float().mean().item(),
        }

    if args.reader_study > 0:
        reader_study(test_x, gen_images[f"T{settings[0][0]}_k{settings[0][1]}"], args.reader_study,
                     os.path.join(out, "reader_study"), args.seed)

    log("8. profile")
    # 8. profile
    vh, ph = read_json(os.path.join(os.path.dirname(args.vqvae), "vqvae_history.json")), \
        read_json(os.path.join(os.path.dirname(args.prior), "prior_history.json"))
    metrics["profile"] = {
        "vqvae_params_m": n_params(vq), "prior_params_m": n_params(prior),
        "vqvae_train_time_s": vh.get("profile", {}).get("train_time_s"),
        "prior_train_time_s": ph.get("profile", {}).get("train_time_s"),
        "vqvae_train_peak_gpu_mb": vh.get("profile", {}).get("peak_gpu_mb"),
        "prior_train_peak_gpu_mb": ph.get("profile", {}).get("peak_gpu_mb"),
        "predict_peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
        "generation_images_per_second": {n: v["images_per_second"] for n, v in metrics["settings"].items()},
    }

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n" + json.dumps(metrics, indent=2))
    print(f"\nFigures and metrics saved to {out}/")


# ---------------------------------------------------------------- main
def main():
    """Parse arguments, seed everything and run the evaluation for the chosen mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["vae", "vqvae"], help="which model to evaluate")
    ap.add_argument("--checkpoint", default=None, help="vae mode: path to vae_best.pt")
    ap.add_argument("--vqvae", default=None, help="vqvae mode: path to vqvae_best.pt")
    ap.add_argument("--prior", default=None, help="vqvae mode: path to prior_best.pt")
    ap.add_argument("--vae", default=None, help="vqvae mode: optional path to vae_best.pt, the baseline to compare against")
    ap.add_argument("--settings", default="1.0:0,0.8:50",
                    help="vqvae mode: comma-separated temperature:top_k sampling settings (top_k 0 = off)")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="images generated (default 2000 for vae; 256 per setting for vqvae, about 3 images/s)")
    ap.add_argument("--max_real", type=int, default=2000, help="cap on real images used for PCA and diversity")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--reader_study", type=int, default=0, help="export N real + N fake blinded PNGs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model == "vae":
        assert args.checkpoint, "vae mode needs --checkpoint path/to/vae_best.pt"
    else:
        assert args.vqvae and args.prior, "vqvae mode needs --vqvae and --prior"
    if args.n_samples is None:
        args.n_samples = 2000 if args.model == "vae" else 256

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "vae":
        evaluate_vae(args, device)
    else:
        evaluate_vqvae(args, device)


if __name__ == "__main__":
    main()
