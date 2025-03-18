import cv2
import numpy as np

from camera import Camera


def main():
    cam_1 = Camera("calibration_data_1")

    cam_1.calibration()
    cam_1.save_calib()


if __name__ == "__main__":
    main()

