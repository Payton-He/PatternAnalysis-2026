"""Train, validate, test and save the VAE or the WGAN-GP.

    python train.py vae  --epochs 50
    python train.py vae  --debug --epochs 1     # quick smoke test (data mostly zeros)

Checkpoints are written to models/. Run from the project root so the relative
data/ path used by dataset.py resolves.
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
from modules import VAE#, Generator, Critic


def parse_args():
    """Define and parse the command-line arguments.

    The first positional argument selects the model ("vae" or "wgan"); the
    rest are optional hyperparameters. Arguments prefixed "VAE" or "WGAN" only
    affect that model.
    """
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["vae", "wgan"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 (vae), 1e-4 (wgan)")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--base", type=int, default=32, help="channel width of the first conv layer")
    p.add_argument("--beta", type=float, default=1.0, help="VAE: weight on the KL term")
    p.add_argument("--kl_warmup", type=int, default=10, help="VAE: epochs to ramp the KL weight from 0 to beta")
    p.add_argument("--n_critic", type=int, default=5, help="WGAN: critic updates per generator update")
    p.add_argument("--lam", type=float, default=10.0, help="WGAN: gradient penalty weight")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", default="models", help="directory for checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="load only ~20 files per split")
    return p.parse_args()


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


def main():
    """Parse arguments, build the data loaders and run the selected model."""
    args = parse_args()
    torch.manual_seed(args.seed)               # reproducible initialisation and noise
    os.makedirs(args.out, exist_ok=True)       # create models/ if it does not exist
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers,
                              early_stop=args.debug)

    # Read the image shape from a real batch instead of hardcoding 256x128
    _, C, H, W = next(iter(loaders[0])).shape
    print(f"device {device} | image shape ({C}, {H}, {W})")

    if args.model == "vae":
        run_vae(args, loaders, (C, H, W), device)
    else:
        run_wgan(args, loaders, (C, H, W), device)  # defined in the WGAN section


if __name__ == "__main__":
    main()