# =============================================================================
# Image Registration Pipeline
# 
# This script implements a deformable image registration system using:
# - Control point detection via edge/corner features
# - Local displacement estimation through template matching
# - Triangular mesh-based warping using affine transforms
#
# Main Functions:
# 1. find_control_points() : Detect feature points in mask image
# 2. compute_displacements(): Estimate local motion vectors
# 3. warp_image_triangular(): Deform image using Delaunay triangulation
# 4. demo_registration()   : Full pipeline demonstration
# =============================================================================

# -----------------------------------------------------------------------------
# Library Imports
# -----------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# =============================================================================
# Control Point Detection
# =============================================================================
def find_control_points(mask, num_points=200, edge_thresh=50):
    """
    1. Detect strong edges in the mask image
    2. Select 'num_points' strongest edge/corner locations as control points
    ------------------------------------------------------------------------
    mask        : Grayscale mask image (e.g. 0.jpg)
    num_points  : Maximum number of control points to pick
    edge_thresh : Threshold for Canny or gradient-based detection
    returns: (points Nx2) in (x, y) float coordinates
    """
    # --- Detect edges (e.g. using Canny for simplicity here) ---
    edges = cv2.Canny(mask, edge_thresh, 2 * edge_thresh)

    # --- (Optional) We can also use corners or local maxima of gradient:
    # corners = cv2.goodFeaturesToTrack(mask, maxCorners=num_points, ...)
    # but for now, just sample points from edges:

    # Grab all edge coordinates
    y_coords, x_coords = np.where(edges > 0)
    all_points = np.column_stack((x_coords, y_coords))

    if len(all_points) == 0:
        # fallback: just pick a few random points
        h, w = mask.shape
        all_points = np.column_stack((
            np.random.randint(0, w, size=(num_points,)),
            np.random.randint(0, h, size=(num_points,))
        ))

    # If too many edge pixels, randomly downsample
    if len(all_points) > num_points:
        idx = np.random.choice(len(all_points), num_points, replace=False)
        selected = all_points[idx]
    else:
        selected = all_points

    # Convert to float32 for later usage
    selected = np.array(selected, dtype=np.float32)
    return selected

# =============================================================================
# Displacement Estimation
# =============================================================================
def compute_displacements(mask, target, pts, window=15):
    """
    2. Estimate the local displacement for each control point by
       template matching in a local neighborhood of the target image.
    ----------------------------------------------------------------
    mask   : Grayscale mask image
    target : Grayscale image we want to align to the mask
    pts    : Nx2 control points [x, y] in the mask
    window : Half-size of the search patch. For example 15 => 31x31
    returns: displacements, Nx2 array (dx, dy) for each control point
    """
    displacements = []
    h, w = mask.shape

    # For each point, we will:
    #  - Extract a patch around the point in mask
    #  - Search in a slightly larger region in the target
    #  - Compute the best matching location via some similarity measure
    #    (e.g., Sum of Squared Differences, cross-correlation, etc.)
    
    # Patch half-size (radius)
    r = window

    # Precompute something for boundary checks
    for (x, y) in pts:
        x, y = int(round(x)), int(round(y))
        # Define patch in mask
        x0a = max(0, x - r)
        x0b = min(w, x + r + 1)
        y0a = max(0, y - r)
        y0b = min(h, y + r + 1)

        patch_mask = mask[y0a:y0b, x0a:x0b]

        best_score = 1e15
        best_dx, best_dy = 0, 0

        # We'll restrict our search range to ±r in x and y for simplicity
        # so we search a (2r+1)x(2r+1) region in the target
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                xx0a = x0a + dx
                xx0b = x0b + dx
                yy0a = y0a + dy
                yy0b = y0b + dy

                # Check boundaries
                if xx0a < 0 or yy0a < 0 or xx0b > w or yy0b > h:
                    continue

                patch_target = target[yy0a:yy0b, xx0a:xx0b]

                # If patch sizes do not match, skip
                if patch_target.shape != patch_mask.shape:
                    continue

                # Compute a difference measure (SSD)
                diff = patch_mask.astype(np.float32) - patch_target.astype(np.float32)
                ssd = np.sum(diff * diff)

                if ssd < best_score:
                    best_score = ssd
                    best_dx = dx
                    best_dy = dy

        displacements.append([best_dx, best_dy])

    return np.array(displacements, dtype=np.float32)

# =============================================================================
# Image Warping
# =============================================================================
def warp_image_triangular(image, pts_src, pts_dst, output_shape=None):
    """
    3. Given an image and two corresponding sets of points (same triangulation),
       warp the 'image' so that pts_src align with pts_dst. We build a Delaunay
       triangulation on pts_src, then do piecewise affine transforms to map
       each triangle into position specified by pts_dst.
    ----------------------------------------------------------------------------
    image        : Grayscale or BGR input image
    pts_src      : Nx2 array of source points (original)
    pts_dst      : Nx2 array of destination points (where they should map)
    output_shape : (H, W) for the result image; if None, use original size
    returns: warped image (same type as input)
    """
    if output_shape is None:
        output_shape = image.shape[:2]  # H, W
    Hout, Wout = output_shape

    # Delaunay on source points
    tri = Delaunay(pts_src)

    # Prepare output
    if len(image.shape) == 2:
        warped = np.zeros((Hout, Wout), dtype=image.dtype)
    else:
        warped = np.zeros((Hout, Wout, image.shape[2]), dtype=image.dtype)

    for simplex in tri.simplices:
        # Indices of the vertices of the triangle
        pts_src_tri = pts_src[simplex, :]
        pts_dst_tri = pts_dst[simplex, :]

        # Convert to float32
        src_tri_32 = np.array(pts_src_tri, dtype=np.float32)
        dst_tri_32 = np.array(pts_dst_tri, dtype=np.float32)

        # Compute bounding rectangle in destination
        r_dst = cv2.boundingRect(dst_tri_32)
        x, y, w, h = r_dst

        # Crop the triangular region from the dst "bounding box"
        dst_rect_coords = np.array([
            [dst_tri_32[0,0]-x, dst_tri_32[0,1]-y],
            [dst_tri_32[1,0]-x, dst_tri_32[1,1]-y],
            [dst_tri_32[2,0]-x, dst_tri_32[2,1]-y]
        ], dtype=np.float32)

        # Same for source
        r_src = cv2.boundingRect(src_tri_32)
        xS, yS, wS, hS = r_src
        src_rect_coords = np.array([
            [src_tri_32[0,0]-xS, src_tri_32[0,1]-yS],
            [src_tri_32[1,0]-xS, src_tri_32[1,1]-yS],
            [src_tri_32[2,0]-xS, src_tri_32[2,1]-yS]
        ], dtype=np.float32)

        # Extract the small patch from source
        patch_src = image[yS:yS+hS, xS:xS+wS]

        # Compute affine transform
        M = cv2.getAffineTransform(src_rect_coords, dst_rect_coords)

        # Warp the triangular patch
        patch_warped = cv2.warpAffine(
            patch_src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Create a mask for this triangle
        mask_tri = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_tri, np.int32(dst_rect_coords), 255)

        # Place it into the warped image
        if len(warped.shape) == 2:
            # grayscale
            patch_roi = warped[y:y+h, x:x+w]
            patch_roi[mask_tri==255] = patch_warped[mask_tri==255]
        else:
            # color
            patch_roi = warped[y:y+h, x:x+w, :]
            for c in range(patch_warped.shape[2]):
                patch_roi[mask_tri==255, c] = patch_warped[mask_tri==255, c]

    return warped

# =============================================================================
# Demonstration and Visualization
# =============================================================================
def demo_registration(mask_path="0.jpg", target_path="10.jpg"):
    """
    Demo function to:
      1) Load mask and target
      2) Find control points on the mask
      3) Estimate displacements
      4) Warp target
      5) Show subtractions with and without registration
      6) Show a simple displacement "vector map"
    """
    # --- 1. Load images ---
    mask_bgr = cv2.imread(mask_path)
    target_bgr = cv2.imread(target_path)
    if mask_bgr is None or target_bgr is None:
        raise IOError("Could not load images. Check paths.")

    # Convert to grayscale
    mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    # Optionally resize for speed (comment out if not needed)
    # mask_gray = cv2.resize(mask_gray, (0,0), fx=0.5, fy=0.5)
    # target_gray = cv2.resize(target_gray, (0,0), fx=0.5, fy=0.5)

    # --- 2. Find control points in the mask (edge-based) ---
    control_points = find_control_points(mask_gray, num_points=200, edge_thresh=50)

    # --- 3. Estimate local displacements for each control point ---
    disp = compute_displacements(mask_gray, target_gray, control_points, window=15)

    # Compute the "destination" points = control_points + disp
    dst_points = control_points + disp

    # --- 4. Warp target to match the mask, using triangulation ---
    warped_target_gray = warp_image_triangular(
        image=target_gray,
        pts_src=control_points,
        pts_dst=dst_points,
        output_shape=mask_gray.shape
    )

    # --- 5. Show subtractions with and without registration ---
    # Without registration
    diff_no_reg = cv2.absdiff(mask_gray, target_gray)
    # With registration
    diff_reg = cv2.absdiff(mask_gray, warped_target_gray)

    # --- 6. Display a simple displacement vector map ---
    #    We'll just plot quivers for the control_points
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].set_title("Mask (Grayscale)")
    axs[0].imshow(mask_gray, cmap="gray")
    axs[0].axis("off")

    axs[1].set_title("Target (Grayscale)")
    axs[1].imshow(target_gray, cmap="gray")
    axs[1].axis("off")

    axs[2].set_title("Subtraction: No Registration")
    axs[2].imshow(diff_no_reg, cmap="gray")
    axs[2].axis("off")

    axs[3].set_title("Subtraction: With Registration")
    axs[3].imshow(diff_reg, cmap="gray")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    # Now show the displacement field (quiver plot)
    # We'll draw it over the mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_gray, cmap='gray')
    plt.title("Displacement Field (Control Points)")

    # Transpose control_points so we get x,y arrays
    x_cp = control_points[:, 0]
    y_cp = control_points[:, 1]
    dx = disp[:, 0]
    dy = disp[:, 1]

    # Quiver wants y first, then x for display coords
    plt.quiver(x_cp, y_cp, dx, dy, color='red', angles='xy', scale_units='xy', scale=1)
    plt.axis("off")
    plt.show()

# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    # Run the demo with default image names:
    demo_registration("33.jpg", "34.jpg")
