"""
gnt
This code reads and extracts label and image information from .gnt files to .jpg and dictionary type objects in python. 
This code must be run first if you want to train the network to generate the .jpgs. 
"""

import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class GNT():
    def __init__(self, label_dict):
        self.label_dict = label_dict
    
    # Get labels from GNT file
    def create_label_dict(self, target_dir):
        self.label_dict = {}

        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            num = 0
            try:
                with open(file_path, 'rb') as image_file:
                    # Get length of file
                    file_length = os.path.getsize(file_path)

                    # While current cursor spot is less than length
                    while image_file.tell() < file_length:
                        # Skip length of image
                        image_file.read(4)

                        # Get label
                        label = image_file.read(2)

                        # Get image dimensions
                        width = int.from_bytes(image_file.read(2), byteorder='little')
                        height = int.from_bytes(image_file.read(2), byteorder='little')

                        # Skip byte array of gray-scale image
                        image_file.read(width * height)

                        # Save label
                        # Each entry_name corresponds to the first 4 characters of the .gnt file and a number
                        # The images are saved with the same name, this allows us to know which label corresponds to which image
                        entry_name = f"{os.path.basename(file_path)[:4]}-{num}.jpg"
                        # Convert label to something readable
                        self.label_dict[entry_name] = label.decode("GBK")
                        num += 1

            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        return
    

    # Reverse the gray levels of the image array
    def reverse_gray_levels(image_array):
        reversed_array = 255 - image_array
        return reversed_array


    # Resize image
    def resize_image(self, image, new_size=(32, 32)):
        return image.resize(new_size)


    # Convert GNT to JPG
    def gnt_convert_images(self, file_path, target_dir):
        try:
            with open(file_path, 'rb') as image_file:
                num = 0
                file_length = os.path.getsize(file_path)
                save_name = os.path.basename(file_path)[:4]

                # While current cursor spot is less than length
                while image_file.tell() < file_length:
                    # skip length of image and label
                    image_file.read(6)

                    # image dimensions
                    width = int.from_bytes(image_file.read(2), byteorder='little')
                    height = int.from_bytes(image_file.read(2), byteorder='little')

                    # byte array of gray-scale image
                    byte_array = image_file.read(width * height)
                    # turn into an np image array
                    image_array = np.frombuffer(byte_array, dtype=np.uint8)
                    # reshape into 2d array
                    image_array = image_array.reshape((height, width))
                    # reverse gray levels
                    reversed_array = self.reverse_gray_levels(image_array)
                    # turn into PIL image
                    image = Image.fromarray(reversed_array)
                    # resize image to 32x32
                    resized_image = self.resize_image(image)

                    # Save as JPG
                    if not os.path.exists(f"{target_dir}{save_name}-{num}.jpg"):
                        resized_image.save(f"{target_dir}{save_name}-{num}.jpg")

                    # incr counter
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







