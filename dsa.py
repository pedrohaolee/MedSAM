# =============================================================================
# Digital Subtraction Angiography (DSA) Visualization
# 
# Demonstrates basic image subtraction for motion detection between two frames
# 
# Input Requirements:
# - Two input images (0.jpg and 10.jpg) in working directory
# - OpenCV and Matplotlib installed
# =============================================================================

# -----------------------------------------------------------------------------
# Core Dependencies
# -----------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt

# =============================================================================
# Image Subtraction and Visualization
# =============================================================================

def display_image_subtraction():
    """
    Load two images (0.jpg and 10.jpg), compute their absolute difference,
    and display the results side by side.
    """
    # Read images from the current folder
    image_zero = cv2.imread("0.jpg")
    image_ten = cv2.imread("10.jpg")

    # Check that both images loaded properly
    if image_zero is None:
        print("Could not load 0.jpg")
        return
    if image_ten is None:
        print("Could not load 10.jpg")
        return

    # Compute the absolute difference of the two images
    abs_diff = cv2.absdiff(image_zero, image_ten)

    # Display the images side by side
    fig = plt.figure(figsize=(10, 4))

    # Original image: 0.jpg
    axis1 = fig.add_subplot(1, 3, 1)
    axis1.imshow(cv2.cvtColor(image_zero, cv2.COLOR_BGR2RGB))
    axis1.set_title("Image 0")
    axis1.axis("off")

    # Original image: 10.jpg
    axis2 = fig.add_subplot(1, 3, 2)
    axis2.imshow(cv2.cvtColor(image_ten, cv2.COLOR_BGR2RGB))
    axis2.set_title("Image 10")
    axis2.axis("off")

    # Absolute difference
    axis3 = fig.add_subplot(1, 3, 3)
    axis3.imshow(cv2.cvtColor(abs_diff, cv2.COLOR_BGR2RGB))
    axis3.set_title("Subtraction")
    axis3.axis("off")

    plt.tight_layout()
    plt.show()

# =============================================================================
# Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    display_image_subtraction()
