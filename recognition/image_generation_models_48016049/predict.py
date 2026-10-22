"""Evaluate a trained VQ-VAE + prior and run inference on the test set.

    python predict.py --vqvae models/vqvae_run1/vqvae_best.pt --prior models/prior_run1/prior_best.pt \
                      --vae models/vae_best.pt --settings 1.0:0,0.8:50

Writes figures and metrics.json to results/<vqvae run name>/. Run from the project root so the
relative data/ path used by dataset.py resolves. The shared helpers (SSIM, nearest neighbours, PCA,
split-leakage check, reader study) are imported from predict_vae.py, the earlier VAE evaluation.

Checks performed:
    0. split leakage         - patient cases shared between train/validate/test
    1. reconstruction        - SSIM / PSNR of every test slice through encode -> quantize -> decode,
                               against the VAE baseline (this is the SSIM > 0.6 target)
    2. codebook health       - how many codes are used, and how evenly (the discrete mode-collapse check)
    3. diversity             - pairwise SSIM of generated images against real ones, for each sampling setting
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
import time

import matplotlib
matplotlib.use("Agg")                      # write figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np                         # numpy is allowed here (not in modules.py)
import torch
from torchvision.utils import save_image

from dataset import get_dataloaders
from modules import VQVAE, CodePrior
from predict_vae import (collect, ssim, pairwise_ssim, pca_fit, pca_score, summarize,
                         reader_study, check_split_leakage, load_vae, reconstruct as vae_reconstruct,
                         generate as vae_generate)


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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", required=True, help="path to vqvae_best.pt")
    ap.add_argument("--prior", required=True, help="path to prior_best.pt")
    ap.add_argument("--vae", default=None, help="optional path to vae_best.pt, the baseline to compare against")
    ap.add_argument("--settings", default="1.0:0,0.8:50",
                    help="comma-separated temperature:top_k sampling settings to evaluate (top_k 0 = off)")
    ap.add_argument("--n_samples", type=int, default=256, help="images generated per setting (about 3 images/s)")
    ap.add_argument("--max_real", type=int, default=2000, help="cap on real images used for PCA and diversity")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--reader_study", type=int, default=0, help="export N real + N fake blinded PNGs (first setting)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        vr = vae_reconstruct(vae, test_x, device)
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
        vfake = vae_generate(vae, args.n_samples)
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


if __name__ == "__main__":
    main()
