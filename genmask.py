import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_grayscale_mask_with_context(image, hsv_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Loop through each class and apply the corresponding HSV range
    for label, (lower, upper) in hsv_ranges.items():
        class_mask = cv2.inRange(hsv, lower, upper)

        # Use a Gaussian Blur to consider the surrounding context of each pixel
        blurred_mask = cv2.GaussianBlur(class_mask, (5, 5), 0)
        
        # Threshold the blurred mask to consider more surrounding areas
        _, binary_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        # Assign the label value to pixels that are part of the current class
        mask[binary_mask > 0] = label  # Assign the label value (0-6)

    return mask

# Load the image
image_path = '5169.png'
image = cv2.imread(image_path)
if image is None:
    print("Error loading image")
    exit()

# If the image has 4 channels (RGBA), remove the alpha channel
if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Define HSV ranges for each class (background, building, road, water, vegetation, farmland, bare land)
hsv_ranges = {
    1: (np.array([0, 0, 120]),   np.array([180, 50, 255])),   # Buildings: light grays/whites
    2: (np.array([0, 0, 60]),    np.array([180, 50, 119])),   # Roads: darker grays
    3: (np.array([90, 50, 50]),  np.array([130, 255, 255])),  # Water: typical blue range
    4: (np.array([35, 40, 40]),  np.array([85, 255, 255])),   # Vegetation: green
    5: (np.array([20, 50, 50]),  np.array([35, 255, 255])),   # Farmland: yellow-green
    6: (np.array([10, 50, 50]),  np.array([20, 255, 255])),   # Bare land: orange/tan
}

# Generate grayscale mask with restricted pixel values (0-6)
grayscale_mask = create_grayscale_mask_with_context(image, hsv_ranges)

# Save the grayscale mask with restricted pixel values
cv2.imwrite("loveda_grayscale_mask_contextual.png", grayscale_mask)

# Display the original image and the grayscale mask
plt.figure(figsize=(12, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Show grayscale mask
plt.subplot(1, 2, 2)
plt.imshow(grayscale_mask, cmap='gray', vmin=0, vmax=6)
plt.title("Grayscale Mask with Context")
plt.axis('off')

plt.show()
