import struct
import cv2
import numpy as np
import os


def preprocess_image(image, foreground_range=(1, 255), normalization_range=(0, 255)):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reverse gray levels
    reversed_image = 255 - gray_image

    # Set background to 0 and foreground to [1, 255]
    foreground_mask = (reversed_image > 0).astype(np.uint8) * foreground_range[1]
    preprocessed_image = foreground_mask + (1 - (reversed_image > 0).astype(np.uint8)) * foreground_range[0]

    # Non-linear normalization of foreground gray levels to the specified range
    preprocessed_image = np.clip(preprocessed_image, normalization_range[0], normalization_range[1])

    return preprocessed_image


def shape_normalization(image):
    # Threshold the image to get a binary mask of the object
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the object
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Assume the largest contour corresponds to the object
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the object
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the object from the original image
        cropped_object = image[y:y+h, x:x+w]

        # Resize the object to a standard size
        standard_size = (100, 100)  # Adjust the size as needed
        normalized_object = cv2.resize(cropped_object, standard_size)

        return normalized_object

    else:
        print("No contours found.")
        return None


def sobel_operator(image):
    # Apply the Sobel operator separately in the horizontal and vertical directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Convert radians to degrees

    return gradient_magnitude, gradient_direction


def decompose_direction(gradient_direction):
    # Define the standard chaincode directions
    num_directions = 8
    bins = np.linspace(0, 360, num_directions, endpoint=False)
    
    # Initialize arrays for directional maps
    directional_maps = []

    # Iterate over bins and compute directional maps
    for i in range(num_directions):
        lower_bound = bins[i]
        upper_bound = bins[i + 1] if i < num_directions - 1 else 360

        # Create a binary mask for the current direction
        direction_mask = np.logical_and(gradient_direction >= lower_bound, gradient_direction < upper_bound)

        # Apply the parallelogram rule to decompose the direction into two adjacent chaincode directions
        parallelogram_mask = np.logical_or(
            np.roll(direction_mask, shift=1, axis=0),  # Shift vertically
            np.roll(direction_mask, shift=1, axis=1)   # Shift horizontally
        )

        # Apply the mask to the gradient direction
        direction_map = gradient_direction * parallelogram_mask

        directional_maps.append(direction_map)

    return directional_maps


def read_dgrl(file_path):
    # Read the binary data from the file
    with open(file_path, 'rb') as file:
        print(os.path.getsize(file_path))

        header_size = file.read(4)
        format_code = file.read(8) # "DGRL"
        illus_len = int.from_bytes(header_size, byteorder='little') - 36
        illus = file.read(illus_len)
        code_type = file.read(20) # "ASCII, GB, etc."
        code_length = int.from_bytes(file.read(2), byteorder='little')
        bits_per_pixel = struct.unpack('<H', file.read(2))[0]

        # Image records
        image_height = int.from_bytes(file.read(4), byteorder='little')
        image_width = int.from_bytes(file.read(4), byteorder='little')
        image_lines = int.from_bytes(file.read(4), byteorder='little')

        # Line records
        char_number = int.from_bytes(file.read(4), byteorder='little')
        label = file.read(code_length * char_number)
        top_left_coords = file.read(8)
        height = file.read(4)
        width = file.read(4)
        bitmap = file.read(height * width)

    # Convert the byte data to a NumPy array
    numpy_array = np.frombuffer(bitmap, dtype=np.uint8)

    # Decode the NumPy array into an image using OpenCV
    image = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)

    return image


# Read an image
image_path = 'data\\dgrl\\HWDB1.0trn\\001.mpf'
image = read_dgrl(image_path)

cv2.imshow('penis', image)


"""
# Preprocess the image
preprocessed_image = preprocess_image(original_image)

# Shape normalization
normalized_object = shape_normalization(preprocessed_image)

if normalized_object is not None:
    # Apply the Sobel operator
    gradient_magnitude, gradient_direction = sobel_operator(normalized_object)

    # Decompose the direction into 8 equal-sized directions around 360 degrees
    directional_maps = decompose_direction(gradient_direction)

    # Display the original, preprocessed, normalized object, gradient magnitude, and directional maps
    plt.figure(figsize=(20, 5))
    

    plt.subplot(1, 12, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 12, 2)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')

    plt.subplot(1, 12, 3)
    plt.imshow(normalized_object, cmap='gray')
    plt.title('Normalized')
    plt.axis('off')

    plt.subplot(1, 12, 4)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Mag.')
    plt.axis('off')

    for i, direction_map in enumerate(directional_maps):
        plt.subplot(1, 12, i + 5)
        plt.imshow(direction_map, cmap='gray', vmin=0, vmax=360)
        plt.title(f'Map {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Shape normalization did not find any contours.")
    """

