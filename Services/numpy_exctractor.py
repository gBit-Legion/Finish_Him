import numpy as np
import cv2
from matplotlib import pyplot as plt


class NumpyExtractor:
    def __init__(self, path):
        self.np_array = None
        self.path = path

    def extract_from_file(self):
        self.np_array = np.load(self.path)
        print(self.np_array)

    def visualisation_numpy_array(self):
        print("Array shape:", self.np_array.shape)
        print("Array data type:", self.np_array.dtype)

        if len(self.np_array.shape) != 4:
            raise ValueError("self.np_array must be a 4D array")

        number_of_massive, channels, height, width = self.np_array.shape

        if channels not in [1, 3, 4]:
            raise ValueError(f"Invalid number of channels: {channels}")

        # Normalize and scale the image to the 0-255 range
        image_to_show = self.np_array[0].astype(float)
        print("first", image_to_show, "\n", "__________________________")
        #image_to_show -= image_to_show.min()
        print("second", image_to_show, "\n", "__________________________")
        # if image_to_show.max() != 0:
        #     image_to_show *= (255.0 / image_to_show.max())
        print("third", image_to_show, "\n", "__________________________")
        # image_to_show = image_to_show.astype(np.uint8)
        print("fourth", image_to_show, "\n", "__________________________")
        # If the image has one channel, normalize and apply CLAHE

        # If the image has one channel, normalize and apply CLAHE
        if channels == 1:
            image_to_show = image_to_show[0, :, :]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_to_show = clahe.apply(image_to_show)
            # Stack to form a 3-channel grayscale image for display
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2BGR)

        # If the image has three channels, normalize and apply CLAHE
        elif channels == 3:
            image_to_show = np.transpose(image_to_show, (1, 2, 0))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for c in range(3):
                image_to_show[:, :, c] = clahe.apply(image_to_show[:, :, c])
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)

        # If the image has four channels, process as needed
        elif channels == 4:
            image_to_show = np.transpose(image_to_show, (1, 2, 0))
            # Assuming that we need to apply CLAHE on the RGB channels only
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for c in range(3):  # Apply CLAHE to the RGB channels
                image_to_show[:, :, c] = clahe.apply(image_to_show[:, :, c])
            # No need to convert colors if it is already in RGBA

        # Resize image for better viability (optional)
        image_to_show = cv2.resize(image_to_show, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
        # Display the image
        cv2.imshow("Enhanced Image", image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ne = NumpyExtractor("wrfout_d01_2019-01-01_00%3A00%3A00.npy")
    ne.extract_from_file()
    ne.visualisation_numpy_array()
