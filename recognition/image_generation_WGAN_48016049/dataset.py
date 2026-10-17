import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from glob import glob

DATA_ROOT = "data"

### Converting medical images to numpy

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2D to avoid excessive memory usage.

    NormImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely. Leaves arrays mostly empty, for quick loading and testing scripts.
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

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
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
    """Sorted list of .nii.gz paths in data/<split_dir>/ (e.g. 'keras_slices_train')."""
    files = sorted(glob(os.path.join(root, split_dir, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {os.path.join(root, split_dir)}")
    return files

def load_split(split: str, root: str = DATA_ROOT, normImage=False, early_stop=False):
    """split is 'train', 'validate' or 'test'. Returns array of shape (N, H, W, 1)."""
    names = get_file_list(f"keras_slices_{split}", root)
    images = load_data_2D(names, normImage=normImage, early_stop=early_stop)
    return images[..., np.newaxis]          # add channel axis for conv layers

def load_all(root: str = DATA_ROOT, **kwargs):
    train = load_split("train", root, **kwargs)
    val   = load_split("validate", root, **kwargs)
    test  = load_split("test", root, **kwargs)
    return train, val, test