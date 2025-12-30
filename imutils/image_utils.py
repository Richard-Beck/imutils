# image_utils.py

import os
import numpy as np
from tifffile import imread

def build_raw_group_map(raw_dir):
    """
    Scans a directory for .tif files and groups them by a unique key
    derived from their filenames.
    """
    print("Building image group map from raw files...")
    tiff_paths = [os.path.join(raw_dir, f)
                  for f in os.listdir(raw_dir)
                  if f.lower().endswith('.tif')]
    raw_groups = {}
    for path in tiff_paths:
        parts = os.path.basename(path).split('_')
        # Construct key from parts, excluding the Z-stack index
        key = '_'.join(parts[:len(parts)-4] + parts[len(parts)-3:])
        root, _ = os.path.splitext(key)
        raw_groups.setdefault(root, []).append(path)
    print(f"Found {len(raw_groups)} unique image groups.")
    return raw_groups

def get_raw_group_from_key(key: str, raw_groups: dict) -> list:
    """
    Retrieves the list of file paths for a given unique key from the raw_groups map.

    Args:
        key (str): The unique identifier for an image group.
                   e.g., "MCF10A_A00-IncucyteRawDataLiveDead-varyGlucose-241015_2N-Ctrl_B2_4_00d00h00m"
        raw_groups (dict): The dictionary created by build_raw_group_map.

    Returns:
        list: A list of file paths corresponding to the key. Returns an empty list if the key is not found.
    """
    return raw_groups.get(key, [])

def robust_normalize(arr):
    """Normalizes an array using 1st and 99th percentiles to resist outliers."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    clipped = np.clip(arr, p1, p99)
    if p99 > p1:
        normalized = (clipped - p1) / (p99 - p1) * 255.0
    else:
        normalized = np.zeros_like(arr)
    return normalized.astype(np.uint8)


def robust_normalize_float(arr):
    """
    Normalizes array to 0.0-1.0 float32 using 1st/99th percentiles.
    Safe for 32-bit inputs and prevents integer wrapping artifacts.
    """
    arr = arr.astype(np.float32)
    # Avoid crashing on empty arrays
    if arr.size == 0: return arr
    
    p1, p99 = np.percentile(arr, [1, 99])
    
    # Clip outliers to the calculated range
    clipped = np.clip(arr, p1, p99)
    
    # Scale to 0-1
    if p99 > p1:
        normalized = (clipped - p1) / (p99 - p1)
    else:
        normalized = np.zeros_like(clipped)
        
    return normalized # Returns float32 in [0, 1]
def make_composite(root, raw_groups):
    """
    Creates a Clean 3-Channel RGB Stack for Curation/Classification.
    
    Structure (H, W, 3):
    - Ch 0 (Red):   Dead Signal (Sum of all 'dead' channels)
    - Ch 1 (Green): Alive Signal (Sum of all 'alive' channels)
    - Ch 2 (Blue):  Phase Contrast
    
    Returns: Float32 image (0.0 - 1.0)
    """
    raw_imgs = {'phase': [], 'alive': [], 'dead': []}
    
    for p in raw_groups.get(root, []):
        # FIX: Split filename to isolate the channel name.
        # Your file format: "..._dead_c2_3_..." -> parts[-4] is "dead"
        parts = os.path.basename(p).split('_')
        if len(parts) >= 4:
            name_token = parts[-4].lower()
        else:
            # Fallback if filename structure is unexpected
            name_token = os.path.basename(p).lower()

        img = imread(p).astype(np.float32) 
        
        # Strict matching on the token, not the full string
        if 'phase' in name_token:
            raw_imgs['phase'].append(img)
        elif 'dead' in name_token:
            raw_imgs['dead'].append(img)
        elif 'alive' in name_token:
            raw_imgs['alive'].append(img)

    # 2. Establish shape from phase
    if not raw_imgs['phase']:
        raise ValueError(f"No phase image found for {root}")
    base_shape = raw_imgs['phase'][0].shape

    # 3. Aggregation 
    phase_agg = raw_imgs['phase'][0]
    
    # Sum fluorescence (Safe in float32)
    if raw_imgs['alive']:
        alive_agg = np.sum(raw_imgs['alive'], axis=0)
    else:
        alive_agg = np.zeros(base_shape, dtype=np.float32)

    if raw_imgs['dead']:
        dead_agg = np.sum(raw_imgs['dead'], axis=0)
    else:
        dead_agg = np.zeros(base_shape, dtype=np.float32)

    # 4. Normalize channels INDEPENDENTLY (0.0 - 1.0)
    phase_norm = robust_normalize_float(phase_agg)
    alive_norm = robust_normalize_float(alive_agg)
    dead_norm = robust_normalize_float(dead_agg)

    # 5. Stack into RGB (R=Dead, G=Alive, B=Phase)
    composite = np.stack([dead_norm, alive_norm, phase_norm], axis=-1)
    
    return composite

def make_cpose_input(root, raw_groups):
    """
    Creates a 2-channel Float32 image for Cellpose.
    - Channel 1 (Cytoplasm): Phase contrast
    - Channel 2 (Nuclei): Sum of all fluorescence images
    """
    phase_imgs = []
    other_imgs = [] 

    for p in raw_groups.get(root, []):
        # FIX: Same strict token matching here
        parts = os.path.basename(p).split('_')
        if len(parts) >= 4:
            name_token = parts[-4].lower()
        else:
            name_token = os.path.basename(p).lower()

        img = imread(p).astype(np.float32)
        
        if 'phase' in name_token:
            phase_imgs.append(img)
        else:
            # All non-phase images (alive/dead) go into the "nuclei" channel
            other_imgs.append(img)

    if not phase_imgs: 
        raise ValueError(f"No phase image found for {root}")
    if not other_imgs: 
        raise ValueError(f"No fluorescence images found for {root}")

    phase_agg = phase_imgs[0]
    nuclei_agg = np.sum(other_imgs, axis=0)

    cytoplasm_norm = robust_normalize_float(phase_agg)
    nuclei_norm = robust_normalize_float(nuclei_agg)

    return np.stack([cytoplasm_norm, nuclei_norm], axis=0)
