import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

def compute_nps(image, sigma=5):
    """
    Compute the Noise Power Spectrum (NPS) of a grayscale image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    sigma : float
        Standard deviation for the Gaussian filter (used for noise extraction).
    
    Returns
    -------
    np.ndarray
        1D array representing the radially averaged NPS.
    """
    blurred = gaussian_filter(image, sigma=sigma)
    noise = image - blurred

    noise_fft = fft2(noise)
    noise_fft_shifted = fftshift(noise_fft)
    psd = np.abs(noise_fft_shifted) ** 2

    height, width = psd.shape
    y_indices, x_indices = np.indices((height, width))
    center = (height // 2, width // 2)
    radii = np.sqrt((x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2)
    radii = radii.astype(np.int32)

    radial_sum = np.bincount(radii.ravel(), psd.ravel())
    radial_count = np.bincount(radii.ravel())
    radial_profile = radial_sum / (radial_count + 1e-10)  # avoid division by zero
    
    return radial_profile

def compute_speckle_contrast(image):
    """
    Compute the speckle contrast (std / mean) of an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    
    Returns
    -------
    float
        The computed speckle contrast.
    """
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    speckle_contrast = std_intensity / (mean_intensity + 1e-10)
    return speckle_contrast

def compute_entropy(image):
    """
    Compute the entropy of a grayscale image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    
    Returns
    -------
    float
        The Shannon entropy of the image in bits.
    """
    hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # remove zero entries
    return -np.sum(hist * np.log2(hist))

def plot_histogram(image, title="Histogram of Image"):
    """
    Plot a histogram of pixel intensities with optional KDE for enhanced visualization.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    title : str
        Title for the histogram plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(image.ravel(), bins=256, kde=True)
    plt.axvline(np.mean(image), linestyle='--', linewidth=1,
                label=f'Mean: {np.mean(image):.2f}')
    plt.axvline(np.median(image), linestyle='--', linewidth=1,
                label=f'Median: {np.median(image):.2f}')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load a grayscale image
    image_one = cv2.imread("A1.png", cv2.IMREAD_GRAYSCALE)

    # Compute NPS
    nps_result = compute_nps(image_one)

    # Compute speckle noise metrics
    speckle_contrast_val = compute_speckle_contrast(image_one)
    entropy_val = compute_entropy(image_one)

    print(f"Speckle Contrast: {speckle_contrast_val:.4f}")
    print(f"Image Entropy: {entropy_val:.4f}")

    # Example: Uncomment to plot NPS or the histogram
    # plt.figure(figsize=(8, 5))
    # plt.plot(nps_result, label="Noise Power Spectrum")
    # plt.yscale("log")
    # plt.xlabel("Spatial Frequency")
    # plt.ylabel("Power Spectral Density (Log Scale)")
    # plt.title("NPS of Grayscale Image")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plot_histogram(image_one, title="Histogram of A1 Image")
