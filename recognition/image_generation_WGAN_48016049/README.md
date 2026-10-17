Artificial Intelligence Usage Disclosure

Claude AI
- 

ChatGPT
- Assistance in generating the feasibility review, with manual editing after generation.



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Feasibility Review

Feasibility Review: WGAN for Synthetic HipMRI Generation

Project objective
Develop a Wasserstein Generative Adversarial Network (WGAN) to generate realistic, diverse, and anatomically coherent 2D hip MRI slices. The project investigates whether synthetic images can achieve reasonable structural fidelity without mode collapse, anatomical hallucinations, or memorization of training scans.

Intended users
Medical imaging researchers and machine-learning engineers seeking to augment limited MRI datasets while reducing reliance on sharing original patient scans.

Prototype scope
Train a 2D WGAN with a Lipschitz-constrained critic using the HipMRI dataset, consisting of .nii.gz grayscale images with dimensions 256 x 128. Use a basic convolutional autoencoder as the reconstruction benchmark.

Acceptance criteria
1.Structural fidelity: Achieve mean SSIM > 0.60 under a documented evaluation protocol, reporting sample variation and reference-selection methodology.
2.Image diversity: Demonstrate that generated images cover a meaningful range of real-image features, using feature-space distances and coverage metrics to detect mode collapse.
3.Memorization audit: Compare synthetic images against training and held-out MRI slices using pixel-space and perceptual nearest-neighbour distances; investigate suspiciously close matches.
4.Anatomical plausibility: Evaluate uncurated sample grids, latent-space interpolation, and 3 - 5 flawed generations for distorted tissue boundaries, artifacts, and implausible structures.
5.Computational feasibility: Able to train on the Rangpur cluster in a reasonable amount of time

Dataset Configuration
Load the Nifti files using NiBabel. Verify image orientation, intensity ranges, dimensions, and patient identifiers. Preserve the 256 x 128 spatial resolution initially, remove the singleton channel dimension where appropriate, and normalize intensities consistently.

Chosen Model/Difficulty
WGAN/WGAN-GP (Normal Difficulty) . Potentially upgrading to VQ-VAE (Hard Difficulty) if time remains after completing WGAN.

Estimated Training Time
A rough estimate based on the size of the training data and the Rangpur cluster's capabilities
Training configuration      |    Estimated time per epoch    |          100 epochs
Basic autoencoder           |          15 - 60 seconds       |  25 min - 1.7 hours
WGAN-GP, 3 critic updates   |          1 - 2.5 minutes       |     1.7 - 4.2 hours
WGAN-GP, 5 critic updates   |          1.5 - 4 minutes       |     2.5 - 6.7 hours

Computing budget
UQ Rangpur cluster.

Next experiment
Run the HipMRI data audit, establish autoencoder reconstruction results, then train a small WGAN-GP pilot. Compare image quality, diversity, nearest-neighbour distances, training stability, and GPU usage.

Fallback
If WGAN training is unstable, reduce model capacity and adjust the gradient penalty and critic-update ratio. If the model remains unreliable, retain the autoencoder as a reconstruction benchmark and report the limitations rather than claiming successful unconditional synthesis.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------