"""Two small checks for the report. Run from the project root.

    python verify_extras.py profile --vae models/vae_best.pt --vqvae models/vqvae_run1/vqvae_best.pt --prior models/prior_run1/prior_best.pt
    python verify_extras.py ssim    --vqvae models/vqvae_run1/vqvae_best.pt        # needs: pip install scikit-image

profile : parameter counts, VAE training-step peak GPU memory (batch 32), and inference latency of the
          VAE sampler and the VQ-VAE reconstruction path.
ssim    : compares this project's SSIM (uniform 7x7 window, border-padded) with scikit-image's on 100 test slices.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import time

import torch

from modules import VAE
from predict import load_vae, load_vqvae, load_prior, n_params, ssim


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeats=5):
    """Seconds per call, averaged over `repeats` calls after one warm-up call."""
    fn(); sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeats


def profile(args, device):
    vae = load_vae(args.vae, device)
    vq = load_vqvae(args.vqvae, device)
    prior = load_prior(args.prior, device)
    print(f"\nparameters (M): VAE {n_params(vae):.2f} | VQ-VAE {n_params(vq):.2f} | prior {n_params(prior):.2f}")

    # Peak GPU memory of VAE training steps on random data of the real shape (batch 32)
    cfg = torch.load(args.vae, map_location="cpu")["config"]
    model = VAE(**cfg).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(32, 1, 256, 128, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        recon, mu, lv = model(x)
        loss = ((recon - x) ** 2).sum() / 32 - 0.5 * (1 + lv - mu ** 2 - lv.exp()).sum() / 32
        opt.zero_grad(); loss.backward(); opt.step()
    sync(device)
    if device.type == "cuda":
        print(f"VAE training peak GPU memory (batch 32): {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

    # Inference latency
    s64 = timed(lambda: vae.sample(64), device)
    print(f"VAE sampling: {s64 * 1000:.1f} ms for 64 images ({64 / s64:.0f} images/s, {s64 / 64 * 1000:.2f} ms per image)")
    xb = torch.rand(32, 1, 256, 128, device=device)
    with torch.no_grad():
        r32 = timed(lambda: vq(xb), device)
    print(f"VQ-VAE reconstruction: {r32 * 1000:.1f} ms for 32 images ({r32 / 32 * 1000:.2f} ms per image)")
    print("(VQ-VAE + prior generation speed and training memory are already in metrics.json / the training histories.)")


def ssim_check(args, device):
    from skimage.metrics import structural_similarity
    from dataset import get_dataloaders
    vq = load_vqvae(args.vqvae, device)
    _, _, test_loader = get_dataloaders(batch_size=32)
    x = torch.cat(list(test_loader))
    x = x[torch.linspace(0, len(x) - 1, 100).long()]
    with torch.no_grad():
        r = torch.cat([vq(x[i:i + 50].to(device))[0].cpu() for i in range(0, len(x), 50)])
    mine = ssim(r, x)
    ref = torch.tensor([structural_similarity(a[0].numpy(), b[0].numpy(), data_range=1.0, win_size=7,
                                              use_sample_covariance=False, gaussian_weights=False)
                        for a, b in zip(r, x)])
    print(f"\nmean SSIM over 100 test slices: this project {mine.mean():.4f} | scikit-image (7x7 uniform) {ref.mean():.4f}")
    print(f"largest per-image difference: {(mine - ref).abs().max():.4f}")
    print("Small differences are expected: scikit-image averages only interior pixels, this project pads the border.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["profile", "ssim"])
    ap.add_argument("--vae"); ap.add_argument("--vqvae"); ap.add_argument("--prior")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mode == "profile":
        assert a.vae and a.vqvae and a.prior, "profile needs --vae --vqvae --prior"
        profile(a, dev)
    else:
        assert a.vqvae, "ssim needs --vqvae"
        ssim_check(a, dev)
