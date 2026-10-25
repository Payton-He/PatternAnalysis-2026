"""Measure image sharpness of real slices, reconstructions and generated samples. Run from the project root.

    python sharpness.py --vae models/final_vae/vae_best.pt --vqvae models/final_vqvae/vqvae_best.pt --prior models/final_prior/prior_best.pt

Two simple sharpness measures (higher = sharper), averaged over images:
    gradient  : mean absolute difference between neighbouring pixels (horizontal + vertical)
    laplacian : variance of the Laplacian (second-derivative) response
Each row is also given as a ratio to real test slices (1.00 = as sharp as real).
Comparing reconstructions with generated samples shows whether the blur comes from the decoder (it would appear in
reconstructions too) or from the prior.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse

import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from predict import (collect, load_vae, load_vqvae, load_prior, reconstruct_vq, generate_vq,
                     generate_vae, reconstruct_vae)

LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def sharpness(x):
    """Per-image (gradient, laplacian variance) for a (N, 1, H, W) CPU tensor in [0, 1]."""
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean((1, 2, 3))
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean((1, 2, 3))
    lap = F.conv2d(x, LAP).flatten(1).var(1)
    return dx + dy, lap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--vqvae", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--n", type=int, default=256, help="real images and reconstructions per row")
    ap.add_argument("--n_generated", type=int, default=128, help="prior samples (about 3 images/s)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, vq, prior = load_vae(a.vae, dev), load_vqvae(a.vqvae, dev), load_prior(a.prior, dev)
    train_loader, _, test_loader = get_dataloaders(batch_size=32)
    g = torch.Generator().manual_seed(a.seed)
    test_x = collect(test_loader, 10 ** 9, a.seed)
    test_x = test_x[torch.randperm(len(test_x), generator=g)[:a.n]]
    train_x = collect(train_loader, a.n, a.seed)

    rows = {
        "real test slices": test_x,
        "real training slices": train_x,
        "VQ-VAE reconstructions (of the real test slices)": reconstruct_vq(vq, test_x, dev),
        "VAE reconstructions (of the real test slices)": reconstruct_vae(vae, test_x, dev),
        "VAE samples": generate_vae(vae, a.n),
        "VQ-VAE + prior samples (T=1.0)": generate_vq(vq, prior, a.n_generated, 1.0, 0, dev, seed=a.seed)[0],
    }
    ref_g, ref_l = [v.mean().item() for v in sharpness(rows["real test slices"])]
    print(f"\n{'':52s}{'gradient':>10s}{'(ratio)':>9s}{'laplacian':>12s}{'(ratio)':>9s}")
    for name, x in rows.items():
        gr, lp = (v.mean().item() for v in sharpness(x))
        print(f"{name:52s}{gr:10.4f}{gr / ref_g:9.2f}{lp:12.5f}{lp / ref_l:9.2f}")
