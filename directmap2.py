"""
===============================================================================================================================
DirecMap
===============================================================================================================================
The following code produces DirectMaps for an input image. It is intended to be used for Offline Handwritten Chinese Character
Recognition.

Method to generate directMaps:
1. Preprocess input image by reversing gray levels and setting the background to 0 and the foreground to [1,255].
2. Apply shape normalization to the image so that the characters are normalized.
3. Use the Sobel Operator to calculate the gradient direction and magnitude for the image.
4. Process direction decomposition from the gradients and map them into 8 equidistant directions.

INPUT:
Path to image must be added to the main function

OUTPUT:
The visualise function is used to plot the original, preprocessed, normalized object, gradient magnitude directional maps and 
its average.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, foreground_range=(1, 255), normalization_range=(0, 255)):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reverse gray levels
    reversed_image = 255 - gray_image

    # Set background to 0 and foreground to [1, 255]
    foreground_mask = (reversed_image > 50).astype(np.uint8) * foreground_range[1]

    preprocessed_image = foreground_mask + (1 - (reversed_image > 0).astype(np.uint8)) * foreground_range[0]

    # Non-linear normalization of foreground gray levels to the specified range
    preprocessed_image = np.clip(preprocessed_image, normalization_range[0], normalization_range[1])

    plt.show()

    return preprocessed_image

def shape_normalization(image):
    # Threshold the image to get a binary mask of the object
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the object
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty canvas to draw the normalized image
    normalized_image = np.zeros_like(image)

    # Iterate through all contours and draw them on the canvas
    for contour in contours:
        # Get the bounding box of the object
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the object from the original image
        cropped_object = image[y:y+h, x:x+w]

        # Resize the object to match the bounding box size
        normalized_object = cv2.resize(cropped_object, (w, h))

        # Draw the normalized object on the canvas
        normalized_image[y:y+h, x:x+w] = normalized_object

    return normalized_image


def sobel_operator(image):
    # Apply the Sobel operator separately in the horizontal and vertical directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    # Compute the gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * (360 / np.pi)  # Convert radians to degrees

    return gradient_magnitude, gradient_direction

def decompose_direction(gradient_magnitude, gradient_direction, image):
    # Define the standard chaincode directions
    num_directions = 8
    bins = np.linspace(0, 360, num_directions, endpoint=False)

    # Initialize arrays for directional maps
    directional_maps = []

    # Iterate over bins and compute directional maps
    for i in range(num_directions):
        lower_bound = bins[i]
        upper_bound = bins[i + 1] if i < num_directions - 1 else 360

        # Create a binary mask for the current direction based on both magnitude and direction
        direction_mask = np.logical_and.reduce([
            gradient_direction >= lower_bound,
            gradient_direction < upper_bound,
            gradient_magnitude > 50  # Ensure magnitude is greater than 0
        ])

        # Dilate the binary mask to create smoother and more connected strokes
        kernel_size = 1  # Adjust the kernel size as needed
        direction_mask = cv2.dilate(direction_mask.astype(np.uint8), np.ones((kernel_size, kernel_size)))

        # Apply the parallelogram rule to decompose the direction into two adjacent chaincode directions
        parallelogram_mask = np.logical_or(
            np.roll(direction_mask, shift=1, axis=0),  # Shift vertically
            np.roll(direction_mask, shift=1, axis=1)   # Shift horizontally
        )

        # Apply the mask to the gradient direction
        direction_map = gradient_direction * parallelogram_mask

        directional_maps.append(direction_map)

    return directional_maps

def visualise(original_image, preprocessed_image, normalized_object, gradient_magnitude, directional_maps, average_directional_map):
    # Display the original, preprocessed, normalized object, gradient magnitude directional maps and its average
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 13, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 13, 2)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')

    plt.subplot(1, 13, 3)
    plt.imshow(normalized_object, cmap='gray')
    plt.title('Normalized')
    plt.axis('off')

    plt.subplot(1, 13, 4)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Mag.')
    plt.axis('off')

    for i, direction_map in enumerate(directional_maps):
        plt.subplot(1, 13, i + 5)
        plt.imshow(direction_map, cmap='gray', vmin=0, vmax=360)
        plt.title(f'Map {i + 1}')
        plt.axis('off')

    plt.subplot(1, 13, 13)
    plt.imshow(average_directional_map, cmap='gray', vmin=0, vmax=360)
    plt.title('Average')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Read an image
    image_path = 'INSERT PATH TO IMAGE HERE'
    original_image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(original_image)

    # Shape normalization
    normalized_object = shape_normalization(preprocessed_image)

    if normalized_object is not None:
        # Apply the Sobel operator
        gradient_magnitude, gradient_direction = sobel_operator(normalized_object)

        # Decompose the direction into 8 equal-sized directions around 360 degrees
        directional_maps = decompose_direction(gradient_magnitude, gradient_direction, normalized_object)

        # Calculate the average directional map
        average_directional_map = np.sum(directional_maps, axis=0)

        # Visualise
        visualise(original_image, preprocessed_image, normalized_object, gradient_magnitude, directional_maps, average_directional_map)
    else:
        print("Shape normalization did not find any contours.")