import cv2
import numpy as np

from camera import Camera


def main():
    cam_1 = Camera("calibration_data_1", 2)
    cam_2 = Camera("calibration_data_2", 3)

    cam_1.calibration()
    cam_2.calibration()

    cam_1.save_calib()
    cam_2.save_calib()

if __name__ == "__main__":
    main()

