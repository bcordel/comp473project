from PIL import Image
import numpy as np
import struct

def read_image(file):
    return np.frombuffer(file.read(), dtype=np.uint8)

def restore_page_image(lineImage, lineHei, lineWid):
    page_image = np.zeros((sum(lineHei), max(lineWid)), dtype=np.uint8)

    current_height = 0
    for img, height in zip(lineImage, lineHei):
        page_image[current_height:current_height + height, :lineWid] |= np.array(img).reshape((height, lineWid))
        current_height += height

    return page_image

def dgrl_to_jpg(file_path, output_path):
    with open(file_path, 'rb') as file:
        # Read the header
        dgrlhsize, dgrlcode = struct.unpack('<I8s', file.read(12))
        illuslen = dgrlhsize - 36
        illustr = file.read(illuslen)
        codetype, codelen, bitspp = struct.unpack('<20sHH', file.read(24))

        # Read image data
        pageHei, pageWid, lineNumber = struct.unpack('<III', file.read(12))
        charNumber = [0] * 100
        lineLabel = [b''] * 100
        lineTop = [0] * 100
        lineLeft = [0] * 100
        lineHei = [0] * 100
        lineWid = [0] * 100
        lineImage = [None] * 100

        for n in range(lineNumber):
            print(n)
            charNumber[n] = struct.unpack('<I', file.read(4))[0]
            lineLabel[n] = file.read(charNumber[n] * codelen)
            lineLabel[n] = lineLabel[n].replace(b'\x00', b' ')

            lineTop[n] = struct.unpack('<I', file.read(4))[0]
            lineLeft[n] = struct.unpack('<I', file.read(4))[0]
            lineHei[n] = struct.unpack('<I', file.read(4))[0]
            lineWid[n] = struct.unpack('<I', file.read(4))[0]

            lineImage[n] = file.read(lineHei[n] * lineWid[n])

    # Save as JPEG
    # img = Image.fromarray(page_image)
    # img.save(output_path)


# Example usage:
dgrl_to_jpg("./data/dgrl/HWDB1.0trn/001.mpf", "./data/dgrl/HWDB1.0trn_images/")
