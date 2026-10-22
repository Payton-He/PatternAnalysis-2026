"""Train, validate, test and save the baseline VAE, the VQ-VAE, or the VQ-VAE's code prior.

    python train.py vae   --epochs 50
    python train.py vae   --debug --epochs 1      # quick smoke test (data mostly zeros)
    python train.py vqvae --epochs 40 --vae_ref 0.49 --out models/vqvae_run1
    python train.py vqvae --debug --epochs 2      # quick smoke test
    python train.py prior --vqvae models/vqvae_run1/vqvae_best.pt --epochs 30 --out models/prior_run1

Checkpoints are written to --out (default models/). Run from the project root so the
relative data/ path used by dataset.py resolves.
"""
import time
import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # work around duplicate libiomp5md.dll on Windows

import matplotlib
matplotlib.use("Agg")                      # render to files; no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataset import get_dataloaders
from modules import VAE, VQVAE, CodePrior


def parse_args():
    """Define and parse the command-line arguments.

    The first positional argument selects what to train ("vae", "vqvae" or "prior");
    the rest are optional hyperparameters. Arguments labelled VAE, VQ-VAE or prior
    only affect that model.
    """
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["vae", "vqvae", "prior"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 (vae), 2e-4 (vqvae), 3e-4 (prior)")
    p.add_argument("--latent_dim", type=int, default=128, help="VAE: size of the latent vector")
    p.add_argument("--base", type=int, default=32, help="VAE: channel width of the first conv layer")
    p.add_argument("--beta", type=float, default=1.0, help="VAE: weight on the KL term")
    p.add_argument("--kl_warmup", type=int, default=10, help="VAE: epochs to ramp the KL weight from 0 to beta")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", default="models", help="directory for checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="load only ~20 files per split")
    p.add_argument("--num_codes", type=int, default=512, help="VQ-VAE: codebook size K")
    p.add_argument("--code_dim", type=int, default=64, help="VQ-VAE: size of each code vector")
    p.add_argument("--vq_hidden", type=int, default=128, help="VQ-VAE: channel width of the encoder and decoder")
    p.add_argument("--commit", type=float, default=0.25, help="VQ-VAE: commitment-loss weight")
    p.add_argument("--ssim_weight", type=float, default=1.0, help="VQ-VAE: weight on the (1 - SSIM) term of the reconstruction loss")
    p.add_argument("--vae_ref", type=float, default=None, help="VQ-VAE: VAE baseline reconstruction SSIM, drawn on the plot")
    p.add_argument("--vqvae", default=None, help="prior: path to a trained vqvae_best.pt")
    p.add_argument("--prior_dim", type=int, default=256, help="prior: transformer width")
    p.add_argument("--prior_layers", type=int, default=6, help="prior: transformer layers")
    p.add_argument("--prior_heads", type=int, default=8, help="prior: attention heads")
    p.add_argument("--prior_dropout", type=float, default=0.1, help="prior: dropout (main defence against memorisation)")
    p.add_argument("--temperature", type=float, default=1.0, help="prior: sampling temperature for the final sample grid")
    p.add_argument("--top_k", type=int, default=0, help="prior: sample only from the k likeliest codes (0 = off)")
    return p.parse_args()


# ------------------------------------------------------------- shared metrics
def ssim(x, y, win=7):
    """Structural similarity for each image pair in two (B, 1, H, W) batches in [0, 1].

    Uses a uniform 7x7 window. This is not the Gaussian 11x11 window of the reference
    SSIM implementations, so values differ slightly from library results. Quote the
    definition in your report and use the same function everywhere.

    Returns:
        (B,) tensor, one SSIM per pair (1 = identical).
    """
    c1, c2, p = 0.01 ** 2, 0.03 ** 2, win // 2
    pool = lambda t: F.avg_pool2d(t, win, 1, p, count_include_pad=False)
    mx, my = pool(x), pool(y)
    sxx, syy, sxy = pool(x * x) - mx ** 2, pool(y * y) - my ** 2, pool(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)

# ------------------------------------------------------------------ VAE
def vae_loss(model, x, beta):
    """Compute the VAE loss for one batch (the negative ELBO, per image).

        loss = reconstruction error + beta * KL divergence

    - Reconstruction: squared error between the input and the reconstruction,
      summed over all pixels. This pushes the decoder to reproduce the input.
    - KL divergence: how far the encoder's distribution N(mu, sigma^2) is from
      the standard normal prior N(0, I), summed over latent dimensions. This
      keeps the latent space smooth so that sampling z ~ N(0, I) gives valid
      images. For two Gaussians it has this closed form:
          KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Both terms are summed within each image and averaged over the batch, so the
    loss value does not depend on the batch size.

    Args:
        model: a VAE instance.
        x: batch of images, shape (B, C, H, W), values in [0, 1].
        beta: weight on the KL term (0 = plain autoencoder, 1 = standard VAE).

    Returns:
        total: scalar tensor, rec + beta * kld (the value to backpropagate).
        rec: scalar tensor, reconstruction term (for logging).
        kld: scalar tensor, unweighted KL term (for logging).
    """
    recon, mu, logvar = model(x)
    b = x.size(0)
    rec = F.mse_loss(recon, x, reduction="sum") / b   # sum over pixels, mean over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return rec + beta * kld, rec, kld

# ------------------------------------------------------------ plotting
# Colour follows the entity, never the panel: train is always blue and
# validation is always orange, in every panel and every run. These two hues
# pass colour-blind and contrast checks against the light chart surface.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
SURFACE = "#fcfcfb"   # figure and axes background
INK = "#0b0b0b"       # titles
INK_2 = "#52514e"     # axis labels, legend text
MUTED = "#898781"     # tick labels
GRID = "#e1e0d9"      # horizontal gridlines
AXIS = "#c3c2b7"      # axis lines and the warm-up marker


def psnr(sse_per_image, n_pixels):
    """Peak signal-to-noise ratio in dB for images scaled to [0, 1].

    Args:
        sse_per_image: reconstruction error summed over all pixels of one image
            (the `rec` term from vae_loss, averaged over images).
        n_pixels: number of pixels per image (C * H * W).

    Returns:
        10 * log10(1 / MSE). Higher is better; unlike the summed error it does
        not depend on image size, so it is easier to compare between runs.
    """
    mse = max(sse_per_image / n_pixels, 1e-12)   # guard against log(0)
    return 10.0 * math.log10(1.0 / mse)

def fmt_time(seconds):
    """Format a duration in seconds as H:MM:SS (e.g. 3725 -> '1:02:05')."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def plot_vae_history(history, path, warmup=0):
    """Draw the training curves as a 2x2 grid of line charts and save a PNG.

    Panels: total loss, reconstruction error, KL divergence and PSNR, each with
    a train line (blue) and a validation line (orange). The orange dot marks the
    best validation epoch (the saved checkpoint). The dashed vertical line marks
    the end of KL warm-up. Called every epoch, so you can open the file while
    training is running.

    Args:
        history: dict of equal-length lists. Needs "epoch" plus
            "train_<k>" and "val_<k>" for k in total, rec, kl, psnr.
        path: where to write the PNG.
        warmup: epoch at which KL warm-up ends (0 = no warm-up marker).
    """
    epochs = history["epoch"]
    panels = [
        ("Total loss (rec + \u03b2\u00b7KL, full \u03b2)", "total", "loss per image"),
        ("Reconstruction error", "rec", "squared error per image"),
        ("KL divergence", "kl", "KL per image"),
        ("PSNR (higher is better)", "psnr", "dB"),
    ]
    best_i = min(range(len(epochs)), key=lambda i: history["val_total"][i])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for k, (ax, (title, key, ylabel)) in enumerate(zip(axes.flat, panels)):
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, history[f"train_{key}"], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(epochs, history[f"val_{key}"], color=VAL_COLOR, lw=2, label="validation")
        # Best epoch: filled dot with a surface-coloured ring so it stays legible on the line
        ax.plot(epochs[best_i], history[f"val_{key}"][best_i], "o", ms=8, mfc=VAL_COLOR,
                mec=SURFACE, mew=2, label="best epoch (saved)", zorder=3)
        if 0 < warmup <= epochs[-1]:
            ax.axvline(warmup, color=AXIS, lw=1, ls="--", label="KL warm-up ends")

        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)       # horizontal hairlines only
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

    # One shared legend (every panel shows the same entities)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               labelcolor=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)                                   # free memory; this runs every epoch

def run_vae(args, loaders, shape, device):
    """Train, validate, test and save a VAE, recording and plotting metrics.

    Each epoch trains on the training set, scores the validation set, appends
    the numbers to a history, and rewrites the plot and a JSON copy of the
    history. The checkpoint with the lowest validation loss is saved, and at the
    end that checkpoint is reloaded and scored once on the test set.

    Args:
        args: parsed command-line arguments (see parse_args).
        loaders: (train_loader, val_loader, test_loader) from get_dataloaders.
        shape: (C, H, W) of one image, read from a real batch in main().
        device: torch.device to train on.

    Saves (all in args.out):
        vae_best.pt            best checkpoint (weights + constructor config)
        vae_history.json       per-epoch metrics, for your own plots or report
        vae_training_curves.png the plot, refreshed after every epoch
    """
    train_loader, val_loader, test_loader = loaders
    C, H, W = shape
    n_pixels = C * H * W

    cfg = dict(in_channels=C, img_size=(H, W), latent_dim=args.latent_dim, base=args.base)
    model = VAE(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or 1e-3)

    @torch.no_grad()
    def evaluate(loader):
        """Average loss, reconstruction term and KL term per image over a loader.

        Uses eval mode and no gradients, weights each batch by its size, and
        always uses the full beta so scores are comparable between epochs.
        """
        model.eval()
        tot = rec = kl = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            l, r, k = vae_loss(model, x, args.beta)
            b = x.size(0)
            tot += l.item() * b; rec += r.item() * b; kl += k.item() * b; n += b
        return tot / n, rec / n, kl / n

    path = os.path.join(args.out, "vae_best.pt")
    hist_path = os.path.join(args.out, "vae_history.json")
    plot_path = os.path.join(args.out, "vae_training_curves.png")

    # One list per metric; index i holds epoch i+1.
    history = {k: [] for k in (
        "epoch", "beta",
        "train_total", "val_total",
        "train_rec", "val_rec",
        "train_kl", "val_kl",
        "train_psnr", "val_psnr",
        "epoch_time_s", "elapsed_s",      # seconds for this epoch, and cumulative seconds
    )}
    best = float("inf")
    train_start = time.perf_counter()      # start the clock just before the loop

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()   # evaluate() leaves the model in eval mode
        beta = args.beta * min(1.0, epoch / args.kl_warmup) if args.kl_warmup > 0 else args.beta

        run_rec = run_kl = 0.0
        n_batches = 0
        for x in train_loader:
            x = x.to(device)
            loss, rec, kld = vae_loss(model, x, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_rec += rec.item(); run_kl += kld.item(); n_batches += 1

        train_rec, train_kl = run_rec / n_batches, run_kl / n_batches
        # Recompute the train total at the FULL beta so it can be compared with
        # validation. (The loss actually optimised uses the ramped beta.)
        train_total = train_rec + args.beta * train_kl
        val, vrec, vkl = evaluate(val_loader)
        now = time.perf_counter()
        epoch_time, elapsed = now - epoch_start, now - train_start
        eta = elapsed / epoch * (args.epochs - epoch)   # remaining epochs at the average pace so far

        for key, value in (
            ("epoch", epoch), ("beta", beta),
            ("train_total", train_total), ("val_total", val),
            ("train_rec", train_rec), ("val_rec", vrec),
            ("train_kl", train_kl), ("val_kl", vkl),
            ("train_psnr", psnr(train_rec, n_pixels)), ("val_psnr", psnr(vrec, n_pixels)),
            ("epoch_time_s", epoch_time), ("elapsed_s", elapsed),
        ):
            history[key].append(value)

        print(f"epoch {epoch:3d} | train {train_total:9.2f} | val {val:9.2f} "
              f"(rec {vrec:.2f}, kl {vkl:.2f}, psnr {history['val_psnr'][-1]:.2f} dB) | beta {beta:.2f} "
              f"| {epoch_time:5.1f}s (elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)})")

        # Refresh the plot and history every epoch so a crash or Ctrl-C loses nothing
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        plot_vae_history(history, plot_path, warmup=args.kl_warmup)

        if val < best:
            best = val
            torch.save({"model": "vae", "config": cfg, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val}, path)

    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    test, trec, tkl = evaluate(test_loader)
    print(f"TEST | loss {test:.2f} (rec {trec:.2f}, kl {tkl:.2f}, psnr {psnr(trec, n_pixels):.2f} dB)"
          f"  -> saved {path}")
    print(f"Training curves: {plot_path}")
    print(f"Total training time: {fmt_time(time.perf_counter() - train_start)} for {args.epochs} epochs")


# --------------------------------------------------------------- VQ-VAE
def vq_loss(model, x, ssim_weight):
    """Compute the VQ-VAE loss for one batch.

        loss = L1 + ssim_weight * (1 - SSIM) + commitment

    - L1 (mean absolute pixel error) keeps intensities right; it is less blurry than squared error.
    - (1 - SSIM) optimises the structural similarity metric the project is judged on.
    - The commitment term comes from the quantizer and keeps encoder outputs close to their codes.
      (The codebook itself is updated by moving averages, so it needs no loss term.)

    Args:
        model: a VQVAE.
        x: (B, C, H, W) images in [0, 1].
        ssim_weight: weight on the (1 - SSIM) term.

    Returns:
        total: scalar tensor to backpropagate.
        stats: dict of detached scalar tensors for logging: loss, ssim, mse, commit, perp.
    """
    recon, commit, _, perp = model(x)
    s = ssim(recon, x).mean()
    total = F.l1_loss(recon, x) + ssim_weight * (1 - s) + commit
    stats = {"loss": total.detach(), "ssim": s.detach(), "mse": F.mse_loss(recon, x).detach(),
             "commit": commit.detach(), "perp": perp.detach()}
    return total, stats


def plot_vqvae_history(history, path, target=0.6, ref=None, num_codes=512):
    """Draw the VQ-VAE training curves as a 2x2 grid and save a PNG (rewritten every epoch).

    Panels: total loss, reconstruction SSIM (with the 0.6 target and the VAE baseline), PSNR, and
    codebook perplexity (effective number of codes in use, against the codebook size).
    Train is blue and validation is orange; the orange dot marks the best epoch (the saved model).

    Args:
        history: dict of equal-length lists (see run_vqvae).
        path: output PNG path.
        target: SSIM target drawn as a dashed line.
        ref: VAE baseline SSIM to draw as a dotted line, or None.
        num_codes: codebook size, drawn as the ceiling of the perplexity panel.
    """
    ep = history["epoch"]
    ssim_lines = [(target, f"target {target}", "--")] + ([(ref, "VAE baseline", ":")] if ref is not None else [])
    panels = [
        ("Total loss", "L1 + (1 \u2212 SSIM) + commitment", "loss", []),
        ("Reconstruction SSIM", "SSIM", "ssim", ssim_lines),
        ("PSNR (higher is better)", "dB", "psnr", []),
        ("Codebook perplexity", "effective codes in use", "perp", [(num_codes, f"codebook size {num_codes}", ":")]),
    ]
    best_i = max(range(len(ep)), key=lambda i: history["val_ssim"][i])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for ax, (title, ylabel, key, hlines) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(ep, history[f"train_{key}"], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(ep, history[f"val_{key}"], color=VAL_COLOR, lw=2, label="validation")
        ax.plot(ep[best_i], history[f"val_{key}"][best_i], "o", ms=8, mfc=VAL_COLOR, mec=SURFACE, mew=2,
                zorder=3, label="best epoch (saved)")
        for value, label, style in hlines:
            ax.axhline(value, color=INK_2, lw=1.2, ls=style, label=label)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        if hlines:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               labelcolor=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_vqvae(args, loaders, shape, device):
    """Train, validate, test and save a VQ-VAE, recording and plotting metrics.

    Each epoch trains on the training set and scores the validation set with direct
    reconstruction (encode -> quantize -> decode), which is exactly what the SSIM > 0.6 target
    measures. The checkpoint with the highest validation SSIM is kept. At the end it is reloaded
    and scored once on the test set, and codebook usage over the training set is counted.

    Args:
        args: parsed command-line arguments (see parse_args).
        loaders: (train_loader, val_loader, test_loader) from get_dataloaders.
        shape: (C, H, W) of one image.
        device: torch.device to train on.

    Saves (all in args.out):
        vqvae_best.pt                     best checkpoint (weights, codebook and constructor config)
        vqvae_history.json                per-epoch metrics, GPU memory and timing
        vqvae_training_curves.png         the plot, refreshed after every epoch
        vqvae_best_test_reconstructions.png  top: real test images, bottom: reconstructions
    """
    train_loader, val_loader, test_loader = loaders
    C, H, W = shape
    n_pixels = C * H * W
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cfg = dict(in_channels=C, img_size=(H, W), hidden=args.vq_hidden, num_codes=args.num_codes,
               code_dim=args.code_dim, beta=args.commit)
    model = VQVAE(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or 2e-4)
    print(f"VQ-VAE {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"code grid {model.grid[0]}x{model.grid[1]}, {args.num_codes} codes")

    keys = ("loss", "ssim", "mse", "commit", "perp")

    @torch.no_grad()
    def evaluate(loader):
        """Average of each metric over a loader (eval mode, so the codebook is not updated)."""
        model.eval()
        tot, n = dict.fromkeys(keys, 0.0), 0
        for x in loader:
            x = x.to(device)
            _, st = vq_loss(model, x, args.ssim_weight)
            for k in keys:
                tot[k] += st[k].item() * x.size(0)
            n += x.size(0)
        return {k: v / n for k, v in tot.items()}

    path = os.path.join(args.out, "vqvae_best.pt")
    hist_path = os.path.join(args.out, "vqvae_history.json")
    plot_path = os.path.join(args.out, "vqvae_training_curves.png")
    history = {k: [] for k in ["epoch", "epoch_time_s", "elapsed_s"] +
               [f"{sp}_{m}" for sp in ("train", "val") for m in ("loss", "ssim", "psnr", "perp", "commit")]}
    best = -1.0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()                                         # evaluate() leaves the model in eval mode
        tot, n_batches = dict.fromkeys(keys, 0.0), 0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            loss, st = vq_loss(model, x, args.ssim_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in keys:
                tot[k] += st[k].item()
            n_batches += 1
        tr = {k: v / n_batches for k, v in tot.items()}       # train metrics: running means over the epoch
        va = evaluate(val_loader)
        now = time.perf_counter()
        epoch_time, elapsed = now - epoch_start, now - start
        eta = elapsed / epoch * (args.epochs - epoch)

        values = {"epoch": epoch, "epoch_time_s": epoch_time, "elapsed_s": elapsed}
        for sp, d in (("train", tr), ("val", va)):
            values.update({f"{sp}_loss": d["loss"], f"{sp}_ssim": d["ssim"], f"{sp}_perp": d["perp"],
                           f"{sp}_commit": d["commit"], f"{sp}_psnr": psnr(d["mse"] * n_pixels, n_pixels)})
        for k, v in values.items():
            history[k].append(v)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} SSIM {tr['ssim']:.3f} | val loss {va['loss']:.4f} "
              f"SSIM {va['ssim']:.3f} (target 0.6) PSNR {history['val_psnr'][-1]:.2f} dB | codes in use "
              f"{va['perp']:.0f}/{args.num_codes} | {epoch_time:5.1f}s (elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)})")

        if va["ssim"] > best:                                 # checkpoint first, so a plotting problem never costs the model
            best = va["ssim"]
            torch.save({"model": "vqvae", "config": cfg, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_ssim": va["ssim"]}, path)
        try:
            with open(hist_path, "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)
            plot_vqvae_history(history, plot_path, ref=args.vae_ref, num_codes=args.num_codes)
        except OSError as e:
            print(f"  (could not write history/plot this epoch: {e}; will retry next time)")

    # ---- final test with the best checkpoint
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    te = evaluate(test_loader)
    model.eval()
    used = torch.zeros(args.num_codes, dtype=torch.bool, device=device)   # codes the training images actually use
    with torch.no_grad():
        for x in train_loader:
            used[model.encode_indices(x.to(device)).unique()] = True
    test_x = torch.cat(list(test_loader))
    sel = torch.linspace(0, len(test_x) - 1, 8).long()                    # spread over patients and slice levels
    with torch.no_grad():
        recon = model(test_x[sel].to(device))[0].cpu()
    save_image(torch.cat([test_x[sel], recon]), os.path.join(args.out, "vqvae_best_test_reconstructions.png"), nrow=8)
    total_time = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    print(f"TEST | best epoch {ck['epoch']} | SSIM {te['ssim']:.3f} (target 0.6) | PSNR "
          f"{psnr(te['mse'] * n_pixels, n_pixels):.2f} dB | codes used by training images "
          f"{int(used.sum())}/{args.num_codes} -> saved {path}")
    print(f"Total training time: {fmt_time(total_time)} for {args.epochs} epochs | peak GPU memory {peak_mb:.0f} MB")
    print(f"Training curves: {plot_path}")
    try:                                                                  # keep the final numbers with the history
        with open(hist_path, "w") as f:
            json.dump({"args": vars(args), "history": history,
                       "test": {**te, "psnr": psnr(te["mse"] * n_pixels, n_pixels), "codes_used": int(used.sum()),
                                "best_epoch": ck["epoch"]},
                       "profile": {"train_time_s": total_time, "peak_gpu_mb": peak_mb}}, f, indent=2)
    except OSError as e:
        print(f"  (could not write the final history: {e})")


# ------------------------------------------------------------- code prior
@torch.no_grad()
def encode_all(vq, loader, device):
    """Encode every image in a loader to its integer code grid. Returns (N, H/8, W/8) int16 on `device`."""
    vq.eval()
    return torch.cat([vq.encode_indices(x.to(device)).to(torch.int16).cpu() for x in loader]).to(device)


def plot_prior_history(history, path, num_codes=512):
    """Draw the prior's training curves (two panels) and save a PNG, rewritten every epoch.

    Left: cross-entropy in nats per code (train blue, validation orange; the orange dot is the
    best epoch, which is the saved model). Right: the same as perplexity, exp(cross-entropy), the
    effective number of codes the model is choosing between at each position; the dotted line is
    the codebook size, which a model that had learned nothing would score. A validation curve that
    turns upward while training keeps falling means the prior has started memorising.
    """
    ep = history["epoch"]
    best_i = min(range(len(ep)), key=lambda i: history["val_loss"][i])
    panels = [("Cross-entropy", "nats per code", "loss", []),
              ("Perplexity (effective choices per code)", "exp(cross-entropy)", "ppl", [(num_codes, f"codebook size {num_codes}")])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, (title, ylabel, key, hlines) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(ep, history[f"train_{key}"], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(ep, history[f"val_{key}"], color=VAL_COLOR, lw=2, label="validation")
        ax.plot(ep[best_i], history[f"val_{key}"][best_i], "o", ms=8, mfc=VAL_COLOR, mec=SURFACE, mew=2,
                zorder=3, label="best epoch (saved)")
        for value, label in hlines:
            ax.axhline(value, color=INK_2, lw=1.2, ls=":", label=label)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    axes[1].set_yscale("log")                                 # perplexity spans orders of magnitude
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               labelcolor=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_prior(args, loaders, shape, device):
    """Train the autoregressive prior over a trained VQ-VAE's code grids, then draw samples.

    The VQ-VAE (loaded from --vqvae and frozen) first turns every train, validation and test
    image into a code grid. The prior learns to predict each code from the ones before it. The
    checkpoint with the lowest validation cross-entropy is kept (validation images come from
    different patients, so this is the guard against memorising the training grids). At the end it
    is scored on the test set and a fixed, uncurated grid of 64 generated images is written.

    Saves (all in args.out):
        prior_best.pt              best checkpoint (weights, config and the VQ-VAE path it belongs to)
        prior_history.json         per-epoch metrics, test score and generation speed
        prior_training_curves.png  the plot, refreshed after every epoch
        prior_samples.png          64 generated images (uncurated)
    """
    assert args.vqvae, "the prior needs a trained VQ-VAE: pass --vqvae path/to/vqvae_best.pt"
    train_loader, val_loader, test_loader = loaders
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ck = torch.load(args.vqvae, map_location=device)
    assert ck["model"] == "vqvae", f"{args.vqvae} is a '{ck['model']}' checkpoint, not a VQ-VAE"
    vq = VQVAE(**ck["config"]).to(device)
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    codes = {name: encode_all(vq, ld, device)
             for name, ld in (("train", train_loader), ("val", val_loader), ("test", test_loader))}
    print(f"encoded code grids: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in codes.items()))

    cfg = dict(num_codes=vq.num_codes, grid=vq.grid, dim=args.prior_dim, layers=args.prior_layers,
               heads=args.prior_heads, dropout=args.prior_dropout)
    prior = CodePrior(**cfg).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr or 3e-4, weight_decay=0.01)
    print(f"prior {sum(p.numel() for p in prior.parameters()) / 1e6:.2f}M params, sequence length {prior.seq_len}")

    @torch.no_grad()
    def evaluate(name):
        """Mean cross-entropy (nats per code) over one split, with dropout off."""
        prior.eval()
        grids, tot, n = codes[name], 0.0, 0
        for i in range(0, len(grids), 64):
            batch = grids[i:i + 64].long()
            tot += prior.loss(batch).item() * len(batch)
            n += len(batch)
        return tot / n

    path = os.path.join(args.out, "prior_best.pt")
    hist_path = os.path.join(args.out, "prior_history.json")
    plot_path = os.path.join(args.out, "prior_training_curves.png")
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_ppl", "val_ppl", "epoch_time_s", "elapsed_s")}
    best = float("inf")
    start = time.perf_counter()
    n_train = len(codes["train"])

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        prior.train()
        perm = torch.randperm(n_train, device=device)
        run, n_batches = 0.0, 0
        for i in range(0, n_train - args.batch_size + 1, args.batch_size):
            batch = codes["train"][perm[i:i + args.batch_size]].long()
            loss = prior.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)       # keeps rare large gradients from derailing training
            opt.step()
            run += loss.item(); n_batches += 1
        tr, va = run / n_batches, evaluate("val")
        now = time.perf_counter()
        epoch_time, elapsed = now - epoch_start, now - start
        eta = elapsed / epoch * (args.epochs - epoch)
        for k, v in (("epoch", epoch), ("train_loss", tr), ("val_loss", va), ("train_ppl", math.exp(tr)),
                     ("val_ppl", math.exp(va)), ("epoch_time_s", epoch_time), ("elapsed_s", elapsed)):
            history[k].append(v)
        print(f"epoch {epoch:3d} | train CE {tr:.3f} | val CE {va:.3f} (perplexity {math.exp(va):.1f} of "
              f"{vq.num_codes}) | {epoch_time:5.1f}s (elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)})")

        if va < best:                                          # checkpoint first, so a plotting problem never costs the model
            best = va
            torch.save({"model": "prior", "config": cfg, "state_dict": prior.state_dict(), "epoch": epoch,
                        "val_loss": va, "vqvae": args.vqvae}, path)
        try:
            with open(hist_path, "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)
            plot_prior_history(history, plot_path, vq.num_codes)
        except OSError as e:
            print(f"  (could not write history/plot this epoch: {e}; will retry next time)")

    # ---- final test, and an uncurated sample grid from the best checkpoint
    best_ck = torch.load(path, map_location=device)
    prior.load_state_dict(best_ck["state_dict"])
    test_ce = evaluate("test")
    prior.eval()
    torch.manual_seed(args.seed)                               # fixed seed: the same 64 images every time, no cherry-picking
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    grids = prior.sample(64, temperature=args.temperature, top_k=args.top_k or None)
    images = vq.decode_indices(grids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0
    save_image(images.cpu(), os.path.join(args.out, "prior_samples.png"), nrow=8)
    total_time = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    print(f"TEST | best epoch {best_ck['epoch']} | CE {test_ce:.3f} (perplexity {math.exp(test_ce):.1f}) "
          f"-> saved {path}")
    print(f"Generated 64 images in {gen_time:.1f}s ({64 / gen_time:.1f} images/s); samples: "
          f"{os.path.join(args.out, 'prior_samples.png')}")
    print(f"Total training time: {fmt_time(total_time)} for {args.epochs} epochs | peak GPU memory {peak_mb:.0f} MB")
    try:
        with open(hist_path, "w") as f:
            json.dump({"args": vars(args), "history": history,
                       "test": {"loss": test_ce, "ppl": math.exp(test_ce), "best_epoch": best_ck["epoch"]},
                       "profile": {"train_time_s": total_time, "peak_gpu_mb": peak_mb,
                                   "generate_64_s": gen_time}}, f, indent=2)
    except OSError as e:
        print(f"  (could not write the final history: {e})")



def main():
    """Parse arguments, build the data loaders and run the selected model."""
    args = parse_args()
    torch.manual_seed(args.seed)               # reproducible initialisation and noise
    os.makedirs(args.out, exist_ok=True)       # create the output directory if it does not exist
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers,
                              early_stop=args.debug)

    # Read the image shape from a real batch instead of hardcoding 256x128
    _, C, H, W = next(iter(loaders[0])).shape
    print(f"device {device} | image shape ({C}, {H}, {W})")

    if args.model == "vae":
        run_vae(args, loaders, (C, H, W), device)
    elif args.model == "vqvae":
        run_vqvae(args, loaders, (C, H, W), device)
    else:
        run_prior(args, loaders, (C, H, W), device)


if __name__ == "__main__":
    main()
