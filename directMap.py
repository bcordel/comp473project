"""
What are we doing?
Normalization-cooperated method to generate directMaps:
1. Reverse the gray levels, background as 0 and foreground in [1,255]
2. Foreground gray levels are non-linearly normalized to a specified range [19]
   Shape normalization uses line density projection interpolation method
3. Compute the gradient by the Sobel operator
4. Then decompose the direction of gradient into its two adjacent standard chaincode directions by parallelogram rule see [60]
**Gradient elements of the original images are directly mapped to directMaps of standard image size
"""
import numpy as np
from PIL import Image, ImageFilter
import os
from io import BytesIO


# Takes in a byte array and returns image with reversed gray levels
def reverse_gray_levels(byte_array):
    return bytearray([255 - pixel for pixel in byte_array])


def sobel_operator(image):
    # Apply the Sobel operator to compute the gradient
    gradient_x = image.filter(ImageFilter.FIND_EDGES).convert('L')
    gradient_y = image.filter(ImageFilter.FIND_EDGES).convert('L').transpose(Image.Transpose.ROTATE_270)

    return gradient_x, gradient_y


def direction_decomposition(gradient_x, gradient_y, output_size=(64, 64)):
    # Compute the direction maps by decomposing the gradient
    direction_maps = []
    for i in range(8):
        angle = i * (360 / 8)  # Compute the angle for each chaincode direction
        direction_x = gradient_x * np.cos(np.radians(angle))
        direction_y = gradient_y * np.sin(np.radians(angle))
        direction_map = np.sqrt(direction_x**2 + direction_y**2)
        direction_map = Image.fromarray(direction_map)
        direction_map = direction_map.resize(output_size)
        direction_map.show()
        direction_maps.append(direction_map)

    return direction_maps


def resize_image(image, new_size=(64, 64)):
    # Normalize image (needed?)
    # normalized_image = np.array(transposed_image) / 255.0
    return image.resize(new_size)


def gnt_read_images(file_name):
    image_data = []
    samples = []
    labels = []
    num_classes = 0

    try:
        with open(file_name, 'rb') as image_file:

            file_length = os.path.getsize(file_name)

            # While current cursor spot is less than length
            while image_file.tell() < file_length:
                # skip length of image (we get this from w x h
                int.from_bytes(image_file.read(4), byteorder='little')
                # image label
                label = image_file.read(2)
                # image dimensions
                width = int.from_bytes(image_file.read(2), byteorder='little')
                height = int.from_bytes(image_file.read(2), byteorder='little')
                # byte array of gray-scale image
                byte_array = bytearray(image_file.read(width * height))
                byte_array = reverse_gray_levels(byte_array)

                # Convert to image
                image = Image.frombytes('L', (width, height), byte_array)

                # Resize and normalize image
                resized_image = resize_image(image)

                # Generate directMap from image
                gradient_x, gradient_y = sobel_operator(resized_image)
                directmap_images = direction_decomposition(gradient_x, gradient_y)

                # Append to the sample and label arrays
                samples.append(directmap_images)
                labels.append(label)

            num_classes = len(set(labels))
            image_data.append(num_classes)
            image_data.append(samples)
            image_data.append(labels)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return image_data


def open_images(image_data, num):
    for i in range(num):
        image_data[1][0][i].show()
    print(len(image_data[1]))


file_path = "./data/competition-gnt/C001-f-f.gnt"
images = gnt_read_images(file_path)
open_images(images, 8)






