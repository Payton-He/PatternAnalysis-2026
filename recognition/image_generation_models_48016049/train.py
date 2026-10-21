"""Train, validate, test and save the VAE or the WGAN-GP.

    python train.py vae  --epochs 50
    python train.py vae  --debug --epochs 1     # quick smoke test (data mostly zeros)
    python train.py wgan --iters 10000 --eval_every 250 --vae_ref 0.49 --out models/wgan_run1
    python train.py wgan --debug --iters 20 --eval_every 10 --n_eval 16 --n_recon 8 --recon_steps 20

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
from modules import VAE, Generator, Critic


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
    p.add_argument("--iters", type=int, default=10000, help="WGAN: number of generator updates")
    p.add_argument("--wgan_base", type=int, default=64, help="WGAN: channel width multiplier (generator and critic)")
    p.add_argument("--eval_every", type=int, default=500, help="WGAN: iterations between evaluations, plots and checkpoints")
    p.add_argument("--n_eval", type=int, default=128, help="WGAN: generated images scored by the SSIM metrics")
    p.add_argument("--snapshot_every", type=int, default=2000, help="WGAN: iterations between permanent checkpoints")
    p.add_argument("--augment", type=float, default=0.0, help="WGAN: max random shift of critic inputs as a fraction of image size (0 = off, 0.1 suggested)")
    p.add_argument("--n_recon", type=int, default=32, help="WGAN: real validation images reconstructed at each evaluation (test uses 4x)")
    p.add_argument("--recon_steps", type=int, default=150, help="WGAN: Adam steps of the latent search per image (test uses 2x)")
    p.add_argument("--vae_ref", type=float, default=None, help="WGAN: VAE baseline reconstruction SSIM, drawn on the plot")
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

# ----------------------------------------------------------- WGAN-GP helpers
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


def spread(x, n):
    """Pick n evenly spaced images from x (all of x if it has fewer than n).

    Used to choose a fixed, representative set of validation/test targets: the files are
    sorted by case and slice, so even spacing covers many patients and slice positions.
    """
    idx = torch.linspace(0, len(x) - 1, min(n, len(x))).long()
    return x[idx]


@torch.enable_grad()
def invert_images(G, target, steps=150, lr=0.05, z_weight=0.01, restarts=1, bs=16):
    """Find, for each real image, the latent z whose generated image looks most like it (GAN inversion).

    A GAN has no encoder, so "reconstruct this image" means searching for z. Adam minimises
        (1 - SSIM(G(z), x)) + z_weight * mean(z^2)
    over z, with the generator frozen and in eval mode. The small penalty on z keeps the
    solution near the region the generator was trained on (z ~ N(0, I)), so the score cannot
    be inflated by exotic z values. The SSIM of the final G(z) is the reconstruction SSIM:
    how well the generator can reproduce an image it was never given directly. It is high
    only if the generator can express real anatomy, so it measures coverage and quality.

    Runs even when called inside torch.no_grad() (it re-enables gradients itself).

    Args:
        G: trained Generator (its mode and requires_grad flags are restored afterwards).
        target: (N, 1, H, W) real images in [0, 1], on the same device as G.
        steps: Adam steps per restart. More steps give a higher (more honest) score.
        lr: Adam learning rate for z.
        z_weight: weight of the penalty on the size of z.
        restarts: independent random starting points; the best result per image is kept.
        bs: images optimised together (bounds GPU memory).

    Returns:
        recon: (N, 1, H, W) best reconstructions in [0, 1].
        score: (N,) SSIM of each reconstruction against its target.
        z_rms: (N,) root-mean-square of each final z. About 1 is typical for N(0, I); much
            larger means the search left the region the generator was trained on.
    """
    was_training = G.training
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    recon, score, z_rms = [], [], []
    for i in range(0, len(target), bs):
        x = target[i:i + bs]
        best_s = torch.full((len(x),), -2.0, device=x.device)
        best_r = torch.zeros_like(x)
        best_z = torch.zeros(len(x), device=x.device)
        for _ in range(restarts):
            z = torch.randn(len(x), G.z_dim, device=x.device, requires_grad=True)
            opt = torch.optim.Adam([z], lr=lr)
            for _ in range(steps):
                r = (G(z) + 1) / 2
                loss = ((1 - ssim(r, x)) + z_weight * z.pow(2).mean(1)).sum()   # sum: images are independent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                r = ((G(z) + 1) / 2).clamp(0, 1)
                s = ssim(r, x)
                better = s > best_s                                # keep the best restart per image
                best_s = torch.where(better, s, best_s)
                best_r[better] = r[better]
                best_z[better] = z.pow(2).mean(1).sqrt()[better]
        recon.append(best_r); score.append(best_s); z_rms.append(best_z)
    for p in G.parameters():
        p.requires_grad_(True)
    G.train(was_training)
    return torch.cat(recon), torch.cat(score), torch.cat(z_rms)



def translate(x, frac):
    """Randomly shift each image by up to +-frac of the image size (differentiable augmentation).

    Applied to everything the critic sees (real, generated and interpolated images), so the
    critic cannot memorise exact training images but cannot tell real from fake by the shift
    either. Edges are filled by repeating the border pixels (the background is dark anyway).
    Only translation is used: flips would swap left and right anatomy.

    Args:
        x: (B, C, H, W) images.
        frac: maximum shift as a fraction of the image size; 0 returns x unchanged.
    """
    if frac <= 0:
        return x
    theta = torch.zeros(x.size(0), 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, :, 2] = (torch.rand(x.size(0), 2, device=x.device) * 2 - 1) * frac * 2   # coordinates span [-1, 1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


@torch.no_grad()
def pairwise_ssim(x, n_pairs=500, chunk=125, seed=0):
    """Mean SSIM between random pairs of different images in x: a diversity measure.

    High values mean the images look alike. Compare generated images with the same number
    for real images; generated values far above the real one suggest mode collapse.
    """
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(x), (n_pairs,), generator=g)
    j = torch.randint(0, len(x), (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep].to(x.device), j[keep].to(x.device)
    vals = [ssim(x[i[k:k + chunk]], x[j[k:k + chunk]]) for k in range(0, len(i), chunk)]
    return torch.cat(vals).mean().item()


def infinite(loader):
    """Yield batches from a DataLoader forever, reshuffling each time it is exhausted."""
    while True:
        for batch in loader:
            yield batch


def gradient_penalty(D, real, fake):
    """WGAN-GP gradient penalty: E[(||grad D(x_hat)||_2 - 1)^2] at random interpolates x_hat.

    Pushing the critic's gradient norm towards 1 along lines between real and generated
    images is what enforces the (approximate) 1-Lipschitz constraint on the critic.

    Args:
        D: the critic.
        real, fake: (B, C, H, W) batches in [-1, 1]; `fake` must not require gradients.

    Returns:
        Scalar tensor (differentiable, so it can be added to the critic loss).
    """
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)       # one mixing weight per image
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grads = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


# ------------------------------------------------------------ WGAN plotting
def _style_axis(ax, title, ylabel):
    """Apply the shared look: left-aligned title, recessive horizontal grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("generator iteration", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def plot_wgan_history(history, ref, path, target=0.6):
    """Draw the WGAN-GP training curves as a 2x3 grid of line charts and save a PNG.

    Top row: critic loss, generator loss, Wasserstein estimate (train and validation).
    Bottom row: gradient penalty, validation reconstruction SSIM against the target (and the
    VAE baseline, if given), and sample diversity (pairwise SSIM) against real images.
    The orange dot on the SSIM panel marks the best checkpoint. Rewritten at every
    evaluation, so it can be opened while training runs.

    Args:
        history: dict of equal-length lists, one entry per evaluation (see run_wgan).
        ref: {"pair_ssim": ..., "vae_ssim": ... or None}, reference values to draw as dotted lines.
        path: output PNG path.
        target: SSIM target drawn as a dashed line.
    """
    its = history["iter"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
    ssim_lines = [(target, f"target {target}", "--")]
    if ref.get("vae_ssim") is not None:
        ssim_lines.append((ref["vae_ssim"], "VAE baseline", ":"))
    # (title, y label, [(history key, legend label, colour)], [(y value, legend label, linestyle)])
    panels = [
        ("Critic loss", "\u2212W + \u03bb\u00b7GP", [("train_d_loss", "train", TRAIN_COLOR)], []),
        ("Generator loss", "\u2212E[D(G(z))]", [("train_g_loss", "train", TRAIN_COLOR)], []),
        ("Wasserstein estimate", "E[D(real)] \u2212 E[D(fake)]",
         [("train_w", "train", TRAIN_COLOR), ("val_w", "validation", VAL_COLOR)], []),
        ("Gradient penalty", "mean (\u2016\u2207D\u2016 \u2212 1)\u00b2", [("train_gp", "train", TRAIN_COLOR)], []),
        ("Validation SSIM (reconstruction)", "SSIM", [("val_ssim", "generator", VAL_COLOR)], ssim_lines),
        ("Diversity: pairwise SSIM", "SSIM between random pairs", [("div_ssim", "generated", VAL_COLOR)],
         [(ref["pair_ssim"], "real images", ":")]),
    ]
    for ax, (title, ylabel, series, hlines) in zip(axes.flat, panels):
        _style_axis(ax, title, ylabel)
        for key, label, colour in series:
            ax.plot(its, history[key], color=colour, lw=2, label=label)
        for value, label, style in hlines:
            ax.axhline(value, color=INK_2, lw=1.2, ls=style, label=label)
        if len(series) + len(hlines) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    best_i = max(range(len(its)), key=lambda i: history["val_ssim"][i])
    axes[1, 1].plot(its[best_i], history["val_ssim"][best_i], "o", ms=8, mfc=VAL_COLOR,
                    mec=SURFACE, mew=2, zorder=3)                      # best checkpoint
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)



# ----------------------------------------------------------------- WGAN-GP
def run_wgan(args, loaders, shape, device):
    """Train, validate, test and save a WGAN-GP.

    One iteration = `n_critic` critic updates (each on a fresh batch of real images) followed
    by one generator update. Every `eval_every` iterations the generator is scored on the
    validation set, a sample grid and the plot are written, and checkpoints are saved. The
    checkpoint with the highest validation reconstruction SSIM is kept as wgan_best.pt. At the
    end, that checkpoint is scored once on the test set.

    The SSIM metric is reconstruction SSIM: each real image is reproduced by searching for the
    latent z whose generated image matches it best (see invert_images), then compared with SSIM.
    Metrics use the generator in eval mode (BatchNorm running statistics), which is how
    predict.py will use it, so what you select is what you deploy.

    Args:
        args: parsed command-line arguments (see parse_args).
        loaders: (train_loader, val_loader, test_loader) from get_dataloaders.
        shape: (C, H, W) of one image.
        device: torch.device to train on.

    Saves (all in args.out):
        wgan_best.pt, wgan_last.pt    checkpoints (G and D weights plus constructor configs)
        wgan_iter_XXXXXX.pt           permanent snapshots every --snapshot_every iterations
        wgan_history.json             arguments, reference values and per-evaluation metrics
        wgan_training_curves.png      the plot, refreshed at every evaluation
        wgan_samples/iter_XXXXXX.png  fixed-noise sample grid at every evaluation
        wgan_best_samples.png         sample grid from the best checkpoint
        wgan_best_test_reconstructions.png  top: real test images, bottom: their reconstructions
    """
    train_loader, val_loader, test_loader = loaders
    C, H, W = shape
    torch.backends.cudnn.benchmark = True      # fixed input sizes, so cuDNN can pick the fastest kernels

    g_cfg = dict(z_dim=args.latent_dim, out_channels=C, img_size=(H, W), base=args.wgan_base)
    d_cfg = dict(in_channels=C, img_size=(H, W), base=args.wgan_base)
    G, D = Generator(**g_cfg).to(device), Critic(**d_cfg).to(device)
    print(f"generator {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params, "
          f"critic {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M params")

    lr = args.lr or 1e-4
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))   # betas from the WGAN-GP paper
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    Da = lambda x: D(translate(x, args.augment))      # the critic as it is trained and scored (augmented inputs)

    sample_dir = os.path.join(args.out, "wgan_samples")
    os.makedirs(sample_dir, exist_ok=True)
    fixed_z = torch.randn(64, args.latent_dim, device=device)          # same noise at every evaluation

    # Real images kept on the GPU (about 90 MB each). The same fixed, evenly spaced targets are
    # reconstructed at every evaluation, so the SSIM curve changes only because the model does.
    val_x = torch.cat(list(val_loader)).to(device)
    test_x = torch.cat(list(test_loader)).to(device)
    val_targets = spread(val_x, args.n_recon)
    test_targets = spread(test_x, 4 * args.n_recon)
    data = infinite(train_loader)

    @torch.no_grad()
    def evaluate(loader, targets, steps, restarts=1):
        """Score the current generator (eval mode) against a real split.

        Args:
            loader: the split's DataLoader (for the Wasserstein estimate).
            targets: real images to reconstruct (for the reconstruction SSIM).
            steps, restarts: effort of the latent search (see invert_images).

        Returns:
            (Wasserstein estimate, mean reconstruction SSIM, pairwise SSIM of generated images,
             mean z-rms of the found latents, reconstructions).
        """
        G.eval()
        tot, n = 0.0, 0
        for x in loader:
            real = x.to(device) * 2 - 1
            z = torch.randn(real.size(0), args.latent_dim, device=device)
            tot += (Da(real).mean() - Da(G(z)).mean()).item() * real.size(0)
            n += real.size(0)
        fake = torch.cat([G.generate(min(32, args.n_eval - i)) for i in range(0, args.n_eval, 32)])
        recon, score, z_rms = invert_images(G, targets, steps=steps, restarts=restarts)
        out = (tot / n, score.mean().item(), pairwise_ssim(fake), z_rms.mean().item(), recon)
        G.train()
        return out

    def save_ckpt(path, it, val_ssim):
        torch.save({"model": "wgan", "config": {"G": g_cfg, "D": d_cfg},
                    "G_state": G.state_dict(), "D_state": D.state_dict(),
                    "iter": it, "val_ssim": val_ssim}, path)

    # Reference from REAL images: how alike two different real slices are (the diversity target).
    ref = {"pair_ssim": pairwise_ssim(val_x[:args.n_eval]), "vae_ssim": args.vae_ref}
    print(f"real-image reference: pairwise SSIM {ref['pair_ssim']:.3f}"
          + (f" | VAE baseline reconstruction SSIM {args.vae_ref:.3f}" if args.vae_ref is not None else ""))

    hist_path = os.path.join(args.out, "wgan_history.json")
    plot_path = os.path.join(args.out, "wgan_training_curves.png")
    history = {k: [] for k in ("iter", "elapsed_s", "train_d_loss", "train_g_loss", "train_w", "train_gp",
                               "val_w", "val_ssim", "div_ssim", "val_zrms")}
    acc = {k: [] for k in ("d", "g", "w", "gp")}      # per-step values, averaged at each evaluation
    best_ssim = -1.0
    G.train(); D.train()
    start = time.perf_counter()

    for it in range(1, args.iters + 1):
        # ---- critic: n_critic updates, each on a fresh batch of real images
        for _ in range(args.n_critic):
            real = next(data).to(device, non_blocking=True) * 2 - 1        # [0, 1] -> [-1, 1] for tanh
            z = torch.randn(real.size(0), args.latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            w_est = Da(real).mean() - Da(fake).mean()                        # Wasserstein estimate
            gp = gradient_penalty(Da, real, fake)
            d_loss = -w_est + args.lam * gp
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
            acc["d"].append(d_loss.detach()); acc["w"].append(w_est.detach()); acc["gp"].append(gp.detach())

        # ---- generator: one update; the critic's weights are frozen so no wasted gradients
        D.requires_grad_(False)
        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        g_loss = -Da(G(z)).mean()
        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()
        D.requires_grad_(True)
        acc["g"].append(g_loss.detach())

        if it % args.eval_every == 0 or it == args.iters:
            m = {k: torch.stack(v).mean().item() for k, v in acc.items()}   # means over this interval
            for v in acc.values():
                v.clear()
            if not all(math.isfinite(x) for x in m.values()):
                raise RuntimeError(f"loss became NaN/inf at iteration {it}; try a lower --lr")

            val_w, val_ssim, div, zrms, _ = evaluate(val_loader, val_targets, args.recon_steps)
            elapsed = time.perf_counter() - start
            eta = elapsed / it * (args.iters - it)
            for key, value in (("iter", it), ("elapsed_s", elapsed), ("train_d_loss", m["d"]),
                               ("train_g_loss", m["g"]), ("train_w", m["w"]), ("train_gp", m["gp"]),
                               ("val_w", val_w), ("val_ssim", val_ssim), ("div_ssim", div), ("val_zrms", zrms)):
                history[key].append(value)
            print(f"iter {it:6d}/{args.iters} | D {m['d']:7.3f} G {m['g']:7.3f} | W train {m['w']:6.3f} "
                  f"val {val_w:6.3f} | GP {m['gp']:.3f} | val recon SSIM {val_ssim:.3f} (target 0.6, z-rms {zrms:.2f}) "
                  f"| div {div:.3f} (real {ref['pair_ssim']:.3f}) "
                  f"| {fmt_time(elapsed)} ETA {fmt_time(eta)}")

            # Checkpoints first, so a plotting problem can never cost you a model
            save_ckpt(os.path.join(args.out, "wgan_last.pt"), it, val_ssim)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_ckpt(os.path.join(args.out, "wgan_best.pt"), it, val_ssim)

            if it % args.snapshot_every == 0:
                save_ckpt(os.path.join(args.out, f"wgan_iter_{it:06d}.pt"), it, val_ssim)   # permanent snapshot (about 85 MB)

            G.eval()
            with torch.no_grad():
                grid = torch.cat([G(fixed_z[k:k + 32]) for k in range(0, 64, 32)])
            G.train()
            save_image((grid + 1) / 2, os.path.join(sample_dir, f"iter_{it:06d}.png"), nrow=8)

            try:
                with open(hist_path, "w") as f:
                    json.dump({"args": vars(args), "reference": ref, "history": history}, f, indent=2)
                plot_wgan_history(history, ref, plot_path)
            except OSError as e:
                print(f"  (could not write history/plot this evaluation: {e}; will retry next time)")

    # ---- final test with the best checkpoint (a stronger latent search than during training)
    ck = torch.load(os.path.join(args.out, "wgan_best.pt"), map_location=device)
    G.load_state_dict(ck["G_state"]); D.load_state_dict(ck["D_state"])
    t_w, t_ssim, t_div, t_zrms, t_recon = evaluate(test_loader, test_targets, 2 * args.recon_steps, restarts=2)
    G.eval()
    with torch.no_grad():
        grid = torch.cat([G(fixed_z[k:k + 32]) for k in range(0, 64, 32)])
    save_image((grid + 1) / 2, os.path.join(args.out, "wgan_best_samples.png"), nrow=8)
    k = min(8, len(test_targets))
    save_image(torch.cat([test_targets[:k], t_recon[:k]]),                 # top: real, bottom: reconstruction
               os.path.join(args.out, "wgan_best_test_reconstructions.png"), nrow=k)
    print(f"TEST | best checkpoint: iteration {ck['iter']} | W {t_w:.3f} | reconstruction SSIM {t_ssim:.3f} "
          f"(target 0.6, z-rms {t_zrms:.2f}) | diversity {t_div:.3f}")
    print(f"Total training time: {fmt_time(time.perf_counter() - start)} for {args.iters} iterations")
    print(f"Training curves: {plot_path}")


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