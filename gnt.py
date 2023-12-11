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
            num = 0
            file_length = os.path.getsize(file_name)

            # While current cursor spot is less than length
            while image_file.tell() < file_length:
                # skip length of image (we get this from w x h
                image_file.read(4)
                # image label
                image_file.read(2)
                # image dimensions
                width = int.from_bytes(image_file.read(2), byteorder='little')
                height = int.from_bytes(image_file.read(2), byteorder='little')
                # byte array of gray-scale image
                byte_array = image_file.read(width * height)
                image_array = np.frombuffer(byte_array, dtype=np.uint8)
                image_array = image_array.reshape((height, width))
                reversed_array = reverse_gray_levels(image_array)

                image = Image.fromarray(reversed_array)

                resized_image = resize_image(image)

                # Save as JPG with same num as label
                if not os.path.exists(f"./data/gnt-jpg/train/{save_name}-{num}.jpg"):
                    resized_image.save(f"./data/gnt-jpg/train/{save_name}-{num}.jpg")

                num+=1

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return


if __name__ == "__main__":

    labels = []

    # Convert Files If Needed
    for file in os.listdir("./data/competition-gnt"):
        file_path = f"./data/competition-gnt/{file}"
        save_name = file[:4]
        labels.append(gnt_convert_images(file_path, save_name)) # convert gnt to jpg
        # labels.append(read_labels(file_path))







