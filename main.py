import cv2
import numpy as np

from camera import Camera


def main():
    cam_1 = Camera("calibration_data_1", 0)

    #cam_1.take_photo(n=2)
    cam_1.calibration()
    cam_1.save_calib()


    # Test de la rectification stéréo avec deux images prises avec la caméra 1
    img1_path = "calib_0.jpg"  # Exemple de première image
    img2_path = "calib_1.jpg"  # Exemple de deuxième image

    cam_1.stereo_rectify(img1_path, img2_path)

    # Appeler la méthode de feature matching
    matches = cam_1.feature_matching(img1_path, img2_path)

if __name__ == "__main__":
    main()

