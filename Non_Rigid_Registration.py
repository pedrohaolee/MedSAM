import cv2
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import disk, closing
from skimage.filters import rank
import scipy.interpolate as si
import matplotlib.pyplot as plt

##############################################################################
# MORPHOLOGICAL BOTTOM-HAT
##############################################################################
def morphological_bottom_hat(img, selem_radius=15):
    """
    Perform morphological bottom-hat:
      bottom_hat(I) = closing(I) - I
    for grayscale images, removing dark objects smaller than the structuring element.
    The "enhanced" image is then I - bottom_hat(I) = 2I - closing(I).
    """
    if len(img.shape) != 2:
        raise ValueError("Please provide a single-channel (grayscale) image.")

    # Convert to float32 for safe arithmetic
    fimg = img.astype(np.float32)

    # Create a disk structuring element
    selem = disk(selem_radius)

    # 'closing' with rank filters: rank.minimum then rank.maximum
    # NOTE: We replaced 'selem=' with 'footprint=' for new scikit-image
    closed_min = rank.minimum(img, footprint=selem)
    closed = rank.maximum(closed_min, footprint=selem).astype(np.float32)

    # bottom-hat = closed - original
    bh = closed - fimg

    # Enhanced = original - bottom_hat
    # equivalently: 2I - closing(I)
    enhanced = fimg - bh

    # Clip to valid grayscale
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

##############################################################################
# HYBRID FEATURE-BASED CONTROL POINT SELECTION (live image)
##############################################################################
def select_control_points_live(live_img, max_points=200, edge_thresh=0.3, min_dist=25):
    """
    1. Compute a simple Sobel gradient magnitude + threshold for edges.
    2. Compute a Harris corner map.
    3. Combine them (R = alpha*edges + beta*corners).
    4. Pick local maxima with min_dist spacing, up to max_points.
    """
    # Ensure grayscale float32
    f = live_img.astype(np.float32)

    #--- (A) Edge map via Sobel magnitude ---
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag_norm = mag / (mag.max() + 1e-9)
    # Threshold
    edge_map = (mag_norm > edge_thresh).astype(np.float32)

    #--- (B) Harris corners with skimage ---
    # corner_harris expects [0,1], so we normalize
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-9)
    harris_resp = corner_harris(f_norm, method='k', k=0.12, sigma=1.0)
    coords_corners = corner_peaks(harris_resp, min_distance=5, threshold_rel=0.01)

    # Build a corner-mask
    corner_mask = np.zeros_like(f, dtype=np.float32)
    for (rr, cc) in coords_corners:
        corner_mask[rr, cc] = 1.0

    # Combine
    alpha, beta = 0.3, 0.7
    R = alpha * edge_map + beta * corner_mask

    # We'll pick local maxima in R above 0.1, with min_dist
    yx = np.argwhere(R > 0.1)
    # Sort them by descending R
    vals = [R[r, c] for (r, c) in yx]
    idx_sort = np.argsort(vals)[::-1]
    yx_sorted = yx[idx_sort]

    selected = []
    used = np.zeros_like(R, dtype=bool)

    for (r, c) in yx_sorted:
        if not used[r, c]:
            selected.append((r, c))
            rmin = max(0, r - min_dist)
            rmax = min(R.shape[0], r + min_dist + 1)
            cmin = max(0, c - min_dist)
            cmax = min(R.shape[1], c + min_dist + 1)
            used[rmin:rmax, cmin:cmax] = True

    if len(selected) > max_points:
        selected = selected[:max_points]

    # Convert to (x,y)
    pts = np.array([[s[1], s[0]] for s in selected], dtype=np.float32)
    return pts

##############################################################################
# ENTROPY-OF-DIFFERENCES SIMILARITY
##############################################################################
def histogram_entropy(img1_patch, img2_patch):
    """
    Compute the negative of the entropy of (img1_patch - img2_patch),
    so that a better match => higher return value.
    """
    diff = img1_patch.astype(np.int16) - img2_patch.astype(np.int16)
    hist, _ = np.histogram(diff, bins=511, range=(-255,256), density=True)
    hist = hist + 1e-12
    ent = -np.sum(hist * np.log2(hist))
    # Inverse sign => maximize measure
    return -ent

##############################################################################
# LOCAL DISPLACEMENT ESTIMATION (TEMPLATE MATCHING + MULTI-INIT HILL-CLIMB)
##############################################################################
def local_displacement(mask, live, pt, window=15, disp_limit=20,
                       init_offsets=[(5,5), (5,-5), (-5,5), (-5,-5)]):
    """
    For the point 'pt' in LIVE, find (dx, dy) in MASK that maximizes the ENT measure.
    We'll do a small 'hill-climbing' from multiple init offsets.
    """
    h, w = live.shape
    x0, y0 = int(round(pt[0])), int(round(pt[1]))

    # Extract the patch from live
    xA = max(0, x0 - window)
    xB = min(w - 1, x0 + window)
    yA = max(0, y0 - window)
    yB = min(h - 1, y0 + window)

    if (xB - xA) < 2 or (yB - yA) < 2:
        return (0,0)

    patch_live = live[yA:yB+1, xA:xB+1]

    def climb(dx_init, dy_init):
        step = 2
        best_dx, best_dy = dx_init, dy_init
        best_val = None
        stall_count = 0
        while step >= 1:
            improved = False
            for ddx in [0, -step, step]:
                for ddy in [0, -step, step]:
                    trial_dx = best_dx + ddx
                    trial_dy = best_dy + ddy
                    # Check displacement limit
                    if abs(trial_dx) > disp_limit or abs(trial_dy) > disp_limit:
                        continue

                    # Build patch coords for mask
                    cxA = max(0, x0 + trial_dx - window)
                    cxB = min(w-1, x0 + trial_dx + window)
                    cyA = max(0, y0 + trial_dy - window)
                    cyB = min(h-1, y0 + trial_dy + window)

                    # Must match shape
                    if (cxB - cxA) != (xB - xA) or (cyB - cyA) != (yB - yA):
                        # skip shape mismatch
                        continue
                    patch_mask = mask[cyA:cyB+1, cxA:cxB+1]
                    if patch_live.shape != patch_mask.shape:
                        continue

                    val = histogram_entropy(patch_live, patch_mask)
                    if best_val is None or val > best_val:
                        best_val = val
                        best_dx = trial_dx
                        best_dy = trial_dy
                        improved = True

            if not improved:
                stall_count += 1
                step = step // 2
            else:
                stall_count = 0
            if stall_count >= 2:
                break

        return (best_dx, best_dy, best_val if best_val is not None else -999999)

    best_global_val = None
    best_global_pair = (0,0)

    for (ox, oy) in init_offsets:
        bdx, bdy, bval = climb(ox, oy)
        if best_global_val is None or bval > best_global_val:
            best_global_val = bval
            best_global_pair = (bdx, bdy)

    return best_global_pair

##############################################################################
# MULTILEVEL B-SPLINE INTERPOLATION (SIMPLIFIED)
##############################################################################
def build_displacement_function(pts_src, disp_vals):
    """
    In the original paper, a true multilevel B-spline method is described.
    Here, for simplicity, we use a Radial Basis Function (RBF) or 
    "thin_plate" approach from scipy.interpolate to approximate scattered data.
    If you want the genuine multi-level B-spline approach, you have to code
    a special iterative approach from Lee et al. 1997.
    """
    from scipy.interpolate import Rbf
    xcoords = pts_src[:,0]
    ycoords = pts_src[:,1]
    # We'll do a small "smooth" factor to avoid overfitting
    rbf = Rbf(xcoords, ycoords, disp_vals, function='thin_plate', smooth=0.1)

    def disp_func(xx, yy):
        return rbf(xx, yy)

    return disp_func

def warp_mask_image(mask_img, pts, disps, shape):
    """
    Warp mask_img to align with 'live' given scattered displacement points.
    We'll:
      1) build RBF for dx and dy
      2) create a map grid
      3) remap
    """
    disp_x = disps[:,0]
    disp_y = disps[:,1]

    f_x = build_displacement_function(pts, disp_x)
    f_y = build_displacement_function(pts, disp_y)

    H, W = shape
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    # Evaluate displacements
    # We pass the flattened coords to Rbf
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()

    out_dx = f_x(flat_x, flat_y)
    out_dy = f_y(flat_x, flat_y)

    map_x = flat_x + out_dx
    map_y = flat_y + out_dy

    map_x_2d = map_x.reshape(H, W).astype(np.float32)
    map_y_2d = map_y.reshape(H, W).astype(np.float32)

    warped = cv2.remap(
        mask_img,
        map_x_2d,
        map_y_2d,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

##############################################################################
# MAIN DEMO
##############################################################################
def demo_nejati_registration(mask_path="mask.jpg", live_path="live.jpg"):
    """
    Demonstration of nonrigid registration using:
     - Morphological bottom-hat
     - Hybrid feature selection (edge + corner)
     - Local displacement (template matching w/ ENT)
     - Approximated B-spline (via RBF) warping
     - Final DSA comparison
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    live_img = cv2.imread(live_path, cv2.IMREAD_GRAYSCALE)

    if mask_img is None or live_img is None:
        raise IOError("Failed to load mask or live image. Check file paths.")

    # 1) Morphological bottom-hat => reduce gray distortions
    mask_enh = morphological_bottom_hat(mask_img, selem_radius=15)
    live_enh = morphological_bottom_hat(live_img, selem_radius=15)

    # 2) Select control points from live
    pts_live = select_control_points_live(live_enh, max_points=200, edge_thresh=0.3, min_dist=25)
    print(f"Selected {len(pts_live)} control points.")

    # 3) For each control point, find displacement
    disps = []
    for i, pt in enumerate(pts_live):
        dx, dy = local_displacement(mask_enh, live_enh, pt, window=15, disp_limit=20)
        disps.append([dx, dy])
    disps = np.array(disps, dtype=np.float32)

    # 4) Warp the enhanced mask
    H, W = live_enh.shape
    warped_mask = warp_mask_image(mask_enh, pts_live, disps, (H, W))

    # 5) Display results
    dsa_no_reg = cv2.absdiff(live_enh, mask_enh)
    dsa_reg    = cv2.absdiff(live_enh, warped_mask)

    fig, axs = plt.subplots(1, 4, figsize=(16,4))
    axs[0].imshow(mask_img, cmap='gray'); axs[0].set_title("Original Mask")
    axs[1].imshow(live_img, cmap='gray'); axs[1].set_title("Original Live")
    axs[2].imshow(dsa_no_reg, cmap='gray'); axs[2].set_title("No Registration Subtraction")
    axs[3].imshow(dsa_reg, cmap='gray'); axs[3].set_title("With Registration Subtraction")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# If you want to run from command-line, do something like:
if __name__ == "__main__":
    # Example usage (update paths as needed):
    demo_nejati_registration("33.jpg", "34.jpg")
