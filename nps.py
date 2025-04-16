# =============================================================================
# Noise Power Spectrum (NPS) Analysis Tool
#
# This script implements standardized methods for computing and comparing
# Noise Power Spectra in medical/sonar imaging systems, following IEC 62220-1
# and related standards for noise characterization in digital imaging.
# =============================================================================

# -----------------------------------------------------------------------------
# Core Dependencies
# -----------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d

# =============================================================================
# Noise Power Spectrum Computation
# =============================================================================

def compute_nps(image, sigma=5, target_size=(256, 256)):
    """
    Compute the Noise Power Spectrum (NPS) of a grayscale image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    sigma : float
        Standard deviation for the Gaussian filter (high-pass filtering).
    target_size : tuple of int
        Target size (height, width) for resizing the image.
    
    Returns
    -------
    np.ndarray
        1D array representing the radially averaged Noise Power Spectrum.
    """
    # Resize image for consistent NPS comparison
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # High-pass filter to extract noise
    blurred = gaussian_filter(image_resized, sigma=sigma)
    noise = image_resized - blurred

    # 2D Fourier Transform
    noise_fft = fft2(noise)
    noise_fft_shifted = fftshift(noise_fft)

    # Compute Power Spectral Density (PSD)
    psd = np.abs(noise_fft_shifted) ** 2

    # Radially average the PSD
    height, width = psd.shape
    y_indices, x_indices = np.indices((height, width))
    center = (height // 2, width // 2)
    radii = np.sqrt((x_indices - center[1]) ** 2 +
                    (y_indices - center[0]) ** 2).astype(np.int32)

    radial_sum = np.bincount(radii.ravel(), psd.ravel())
    radial_count = np.bincount(radii.ravel())
    radial_profile = radial_sum / (radial_count + 1e-10)  # avoid div-by-zero

    return radial_profile


# =============================================================================
# NPS Comparison Metrics
# =============================================================================
def compare_nps(nps_one, nps_two):
    """
    Compare two Noise Power Spectra using similarity metrics.
    
    Parameters
    ----------
    nps_one : np.ndarray
        1D array representing the first NPS.
    nps_two : np.ndarray
        1D array representing the second NPS.
    
    Returns
    -------
    dict
        Dictionary of similarity measures:
        - cosine_similarity
        - normalized_l2_norm_difference
    """
    # Interpolate both NPS arrays to the same length
    min_length = min(len(nps_one), len(nps_two))
    x_one = np.linspace(0, 1, len(nps_one))
    x_two = np.linspace(0, 1, len(nps_two))

    f_one = interp1d(x_one, nps_one, kind="linear")
    f_two = interp1d(x_two, nps_two, kind="linear")

    nps_one_resampled = f_one(np.linspace(0, 1, min_length))
    nps_two_resampled = f_two(np.linspace(0, 1, min_length))

    cosine_similarity = 1 - cosine(nps_one_resampled, nps_two_resampled)
    l2_diff = np.linalg.norm(nps_one_resampled - nps_two_resampled)
    l2_sum = np.linalg.norm(nps_one_resampled + nps_two_resampled)
    normalized_l2_norm_difference = l2_diff / l2_sum if l2_sum != 0 else 0

    return {
        "cosine_similarity": cosine_similarity,
        "normalized_l2_norm_difference": normalized_l2_norm_difference
    }

# =============================================================================
# Demonstration and Validation
# =============================================================================
if __name__ == "__main__":

    # Load grayscale images
    image_one = cv2.imread("A2.jpg", cv2.IMREAD_GRAYSCALE)
    image_two = cv2.imread("48.jpg", cv2.IMREAD_GRAYSCALE)

    # Compute NPS for both images
    nps_one = compute_nps(image_one)
    nps_two = compute_nps(image_two)

    # Compare the NPS results
    similarity_scores = compare_nps(nps_one, nps_two)
    print("NPS similarity scores:")
    for metric, score in similarity_scores.items():
        print(f"{metric}: {score:.4f}")

    # Plot each NPS on a log scale
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(nps_one)
    plt.yscale("log")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Spectral Density (log scale)")
    plt.title("Image One NPS")

    plt.subplot(1, 2, 2)
    plt.plot(nps_two)
    plt.yscale("log")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Spectral Density (log scale)")
    plt.title("Image Two NPS")

    plt.tight_layout()
    plt.show()
