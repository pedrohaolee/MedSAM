import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def compute_histogram(image, bins=256):
    """
    Compute the normalized histogram of a grayscale image.
    
    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.
    bins : int
        Number of bins for the histogram.

    Returns
    -------
    hist : np.ndarray
        Normalized histogram (sum of histogram values equals 1).
    """
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = hist / hist.sum()  # Normalize histogram
    return hist


def compute_histogram_similarity(hist1, hist2):
    """
    Compute various histogram similarity metrics between two histograms.
    
    Parameters
    ----------
    hist1 : np.ndarray
        First normalized histogram.
    hist2 : np.ndarray
        Second normalized histogram.

    Returns
    -------
    similarity_scores : dict
        Dictionary of similarity metrics, including:
        - Correlation
        - Chi-Square
        - Intersection
        - Bhattacharyya Distance
    """
    similarity_scores = {
        "correlation": cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL),
        "chi_square": cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR),
        "intersection": cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT),
        "bhattacharyya_distance": cv2.compareHist(hist1, hist2,
                                                  cv2.HISTCMP_BHATTACHARYYA),
    }
    return similarity_scores


def crop_center(image):
    """
    Crop the center region of an image to one-third of its half-size.

    Parameters
    ----------
    image : np.ndarray
        2D array representing a grayscale image.

    Returns
    -------
    cropped_image : np.ndarray
        Center-cropped portion of the input image.
    """
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    crop_size = min(height, width) // 2  # half-size crop
    return image[center_y - crop_size // 3 : center_y + crop_size // 3,
                 center_x - crop_size // 3 : center_x + crop_size // 3]


if __name__ == "__main__":

    # Load images in grayscale
    image_one = cv2.imread("48.jpg", cv2.IMREAD_GRAYSCALE)
    image_two = cv2.imread("217.jpg", cv2.IMREAD_GRAYSCALE)

    # Crop center regions
    image_one_cropped = crop_center(image_one)
    image_two_cropped = crop_center(image_two)

    # Display the cropped images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_one_cropped, cmap="gray")
    axes[0].set_title("Cropped Center of Image 1")
    axes[0].axis("off")

    axes[1].imshow(image_two_cropped, cmap="gray")
    axes[1].set_title("Cropped Center of Image 2")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Compute histograms
    hist_one = compute_histogram(image_one_cropped)
    hist_two = compute_histogram(image_two_cropped)

    # Compute similarity metrics
    similarity_scores = compute_histogram_similarity(hist_one, hist_two)

    # Display similarity in a heatmap
    similarity_df = pd.DataFrame.from_dict(similarity_scores, orient="index",
                                           columns=["similarity_score"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(similarity_df, annot=True, fmt=".4f", linewidths=0.5)
    plt.title("Histogram Similarity Metrics")
    plt.show()

    # Plot histograms for comparison
    plt.figure(figsize=(8, 5))
    # Note: removed explicit color specification to keep styling minimal
    sns.lineplot(x=np.arange(len(hist_one)), y=hist_one.ravel(),
                 label="Image 1 Cropped Histogram")
    sns.lineplot(x=np.arange(len(hist_two)), y=hist_two.ravel(),
                 label="Image 2 Cropped Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title("Histogram Comparison of Cropped Images")
    plt.legend()
    plt.grid()
    plt.show()
