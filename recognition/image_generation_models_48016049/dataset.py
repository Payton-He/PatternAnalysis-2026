"""Data loading for the hip MRI slice dataset.

Expected layout, relative to the project root:

    data/
        keras_slices_train/      *.nii.gz   (2D slices, e.g. case_004_week_0_slice_0.nii.gz)
        keras_slices_validate/   *.nii.gz
        keras_slices_test/       *.nii.gz

Typical use (from train.py or predict.py):

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

Each batch is a float32 tensor of shape (B, 1, H, W) with values in [0, 1].
"""
import numpy as np
import nibabel as nib
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob

# Directory that contains the three split folders. Relative, so scripts must be
# run from the project root (or pass an absolute path as `root`).
DATA_ROOT = "data"

### Converting medical images to numpy

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """One-hot encode an integer label map into channels.

    Intended for segmentation masks only (not needed for the VAE/VQ-VAE, which
    are trained on the images themselves).

    Args:
        arr: label map of shape (H, W), where each pixel holds a class label.
        dtype: dtype of the returned array.

    Returns:
        Array of shape (H, W, K) where K = number of unique labels found in
        `arr`, and channel c is 1 where the pixel has label c.

    Caveats:
        - K is computed per image, so an image missing a label produces fewer
          channels than another one, which breaks the pre-allocated array in
          load_data_2D.
        - The label value is used directly as the channel index, so labels
          must be contiguous integers starting at 0 (e.g. {0, 1, 2}, not
          {0, 1, 2, 4}).
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """Load a list of 2D NIfTI (.nii.gz) files into one pre-allocated numpy array.

    The array is pre-allocated (instead of appending to a list and stacking) to
    avoid holding two copies of the data in memory. Its size is taken from the
    FIRST file, so every file must have the same 2D shape or loading will fail.

    Args:
        imageNames: list of file paths to load (see get_file_list).
        normImage: if True, z-score normalise each image to mean 0 and standard
            deviation 1. Use for MRI images only, never for masks. Note that
            this does NOT scale to [0, 1].
        categorical: if True, one-hot encode each image with to_channels. Use
            for segmentation masks only.
        dtype: numpy dtype of the returned array.
        getAffines: if True, also return the list of NIfTI affine matrices
            (voxel-to-world transforms), one per file.
        early_stop: if True, stop loading after about 20 files. The rest of the
            array stays all zeros, so this is for quick debugging only and the
            result must not be used for real training.

    Returns:
        images: array of shape (N, rows, cols), or (N, rows, cols, K) when
            categorical=True.
        affines: only returned when getAffines=True; list of N (4, 4) arrays.
    """
    affines = []

    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    # tqdm shows a progress bar while the files are read from disk
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # z-score; the 1e-8 avoids division by zero for constant images
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage
        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

### Loading medical images into training, validation, and testing sets

def get_file_list(split_dir: str, root: str = DATA_ROOT) -> list:
    """Sorted list of .nii.gz paths in data/<split_dir>/ (e.g. 'keras_slices_train').

    Sorting makes the order deterministic, which keeps images and masks aligned
    if both are loaded from parallel directories.

    Args:
        split_dir: name of the split folder inside `root`.
        root: directory containing the split folders.

    Returns:
        List of file paths.

    Raises:
        FileNotFoundError: if the folder has no .nii.gz files. This usually
            means the script was run from the wrong directory or the folder
            name is misspelled.
    """
    files = sorted(glob(os.path.join(root, split_dir, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {os.path.join(root, split_dir)}")
    return files

class SliceDataset(Dataset):
    """PyTorch Dataset over an in-memory array of 2D slices.

    Each image is min-max scaled to [0, 1] independently when the dataset is
    created, then stored as a float32 tensor. Scaling per image (not over the
    whole dataset) removes brightness differences between scans, and [0, 1]
    matches the Sigmoid output of the VAE and VQ-VAE decoders.

    Indexing returns a tensor of shape (1, H, W), where the leading 1 is the
    channel axis. The DataLoader stacks these into (B, 1, H, W).
    """
    def __init__(self, images: np.ndarray):
        """
        Args:
            images: float array of shape (N, H, W), as returned by
                load_data_2D with categorical=False.
        """
        # Per-image min-max scaling to [0, 1]
        mins = images.min(axis=(1, 2), keepdims=True)
        maxs = images.max(axis=(1, 2), keepdims=True)
        # 1e-8 avoids division by zero for blank (constant) slices, which map to 0
        images = (images - mins) / (maxs - mins + 1e-8)
        # from_numpy shares memory with the array; unsqueeze(1) adds the channel axis
        self.images = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        """Number of slices in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return slice `idx` as a (1, H, W) float32 tensor in [0, 1]."""
        return self.images[idx]


def get_dataloaders(root=DATA_ROOT, batch_size=32, num_workers=0, early_stop=False, img_size=(256, 128)):
    """Build DataLoaders for the train, validation and test splits.

    All files in each split are loaded into memory up front (see load_data_2D),
    so memory use scales with dataset size: roughly 128 KB per 256x128 float32
    slice.

    Args:
        root: directory containing keras_slices_train/, keras_slices_validate/
            and keras_slices_test/.
        batch_size: images per batch.
        num_workers: DataLoader worker processes.
        early_stop: debugging only; load about 20 files per split and leave the
            rest of the array as zeros (see load_data_2D).
        img_size: (rows, cols) every image must have. Files with a different
            size are skipped (see below).

    Returns:
        (train_loader, val_loader, test_loader). Each batch is a float32 tensor
        of shape (B, 1, H, W) with values in [0, 1].
    """
    loaders = {}
    for split in ("train", "validate", "test"):
        names = get_file_list(f"keras_slices_{split}", root)

        # Keep only slices already at img_size. A few scans (e.g. case_019_week_1) have a
        # different matrix size (256x144); cropping or resizing them would cut or distort anatomy.
        kept = [f for f in names if nib.load(f).shape[:2] == tuple(img_size)]
        if len(kept) < len(names):
            print(f"{split}: skipped {len(names) - len(kept)} of {len(names)} files not matching {tuple(img_size)}")

        images = load_data_2D(kept, normImage=False, early_stop=early_stop)
        loaders[split] = DataLoader(
            SliceDataset(images),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available(),
        )
    return loaders["train"], loaders["validate"], loaders["test"]