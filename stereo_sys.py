import numpy as np
import cv2
import glob

from camera import Camera 

pattern_size = (7, 9)
square_size = 2

class StereoSys():

    def __init__(self, data_name, cap_id_1, cap_id_2):
        self.cam_1 = Camera(data_name+"_1", cap_id_1)
        self.cam_2 = Camera(data_name+"_2", cap_id_2)

    def stereo_calibration(self, name_file_1="calib_1", name_file_2= "calib_2"):
        # Préparation des points 3D réels
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        obj_points = []  # Points 3D réels
        img_points_1 = []  # Points 2D détectés pour caméra 1
        img_points_2 = []  # Points 2D détectés pour caméra 2

        # Charger les images de calibration des deux caméras
        images_1 = glob.glob(f"{name_file_1}_*.jpg")
        images_2 = glob.glob(f"{name_file_2}_*.jpg")

        for fname1, fname2 in zip(images_1, images_2):
            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Détection des coins du damier
            ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

            if ret1 and ret2:
                obj_points.append(objp)
                img_points_1.append(corners1)
                img_points_2.append(corners2)

        # Calibration stéréo
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_1,
            img_points_2,
            self.cam_1.camera_matrix,
            self.cam_1.dist_coeffs,
            self.cam_2.camera_matrix,
            self.cam_2.dist_coeffs,
            gray1.shape[::-1],
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        if ret:
            print("Calibration stéréo terminée !")
            print("Matrice de rotation :\n", R)
            print("Vecteur de translation :\n", T)
            print("Matrice essentielle :\n", E)
            print("Matrice fondamentale :\n", F)
        else:
            print("Échec de la calibration stéréo.")

