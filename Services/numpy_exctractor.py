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

        number_of_massive, channels, height, width = self.np_array.shape

        # Normalize and scale the image to the 0-255 range
        min_val = np.min(self.np_array)
        max_val = np.max(self.np_array)
        scaled_array = (self.np_array - min_val) / (max_val - min_val) * 255.0
        scaled_array = scaled_array.astype(np.uint8)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fourcc = -1
        video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 24, (width, height))

        for i in range(number_of_massive):
            image_to_show = scaled_array[i]

            image_to_show = np.transpose(image_to_show, (1, 2, 0))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for c in range(3):
                image_to_show[:, :, c] = clahe.apply(image_to_show[:, :, c])
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)

            # Resize image for better visibility (optional)
            image_to_show = cv2.resize(image_to_show, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

            # Write current frame to video file
            video_writer.write(image_to_show)

        # Release video writer
        video_writer.release()


if __name__ == '__main__':
    ne = NumpyExtractor("wrfout_d01_2019-01-01_00%3A00%3A00.npy")
    ne.extract_from_file()
    ne.visualisation_numpy_array()
