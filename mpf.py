import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Takes in a np array and returns image with reversed gray levels
def reverse_gray_levels(image_array):
    reversed_array = 255 - image_array
    return reversed_array


def resize_image(image, new_size=(32, 32)):
    return image.resize(new_size)


def gnt_convert_images(file_name, save_name):
    try:
        with open(file_name, 'rb') as image_file:
            file_length = os.path.getsize(file_name)

            # While current cursor spot is less than length
            while image_file.tell() < file_length:
                header = int.from_bytes(image_file.read(4), byteorder='little')
                # Skip format code
                image_file.read(8)
                # Get Illustration
                illus_len = header - 62
                image_file.read(illus_len)
                # Get code type
                code_type = image_file.read(20)
                # Get code length
                code_length = int.from_bytes(image_file.read(2), byteorder='little')
                # Get data type
                data_type = image_file.read(20)
                # Get dimensionality
                dimensionality = int.from_bytes(image_file.read(4), byteorder='little')
                # Get label
                label = image_file.read(code_length)
                # Get image by reading dimensionality * size of data_type (int, char, etc)
                byte_array = image_file.read(dimensionality * 33)


                image_array = np.frombuffer(byte_array, dtype=np.uint8)

                # Save as JPG with same num as label
                #if not os.path.exists(f"./data/gnt-jpg/train/{save_name}-{num}.jpg"):
                 #   resized_image.save(f"./data/gnt-jpg/train/{save_name}-{num}.jpg")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return


if __name__ == "__main__":

    # Convert Files If Needed
    for file in os.listdir("./data/mpf/test"):
        file_path = f"./data/mpf/test/{file}"
        save_name = file[:4]
        gnt_convert_images(file_path, save_name) # convert gnt to jpg







