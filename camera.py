import cv2
import numpy as np
import glob

pattern_size = (7, 9)
square_size = 2

class Camera():

    def __init__(self, data_name):
        self.data_name = data_name
        self.camera_matrix = None
        self.dist_coeffs = None



    def take_photo(self):
        cap = cv2.VideoCapture(2)  # 0 pour la caméra principale

        i = 0
        while i < 10:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture")
                break

            cv2.imshow("Capture du damier", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):  # Appuyer sur 's' pour sauvegarder l'image
                cv2.imwrite(f"calib_{i}.jpg", frame)
                print(f"Image calib_{i}.jpg enregistrée")
                i += 1

            elif key == ord('q'):  # Appuyer sur 'q' pour quitter
                break

        cap.release()
        cv2.destroyAllWindows()

    #####################################################
    #############    Camera Calibration    ##############
    #####################################################

    def calibration(self):

        # Préparation des points 3D réels
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Stockage des points
        obj_points = []  # Points 3D réels
        img_points = []  # Points 2D détectés

        # Charger les images du damier
        images = glob.glob("calib_*.jpg")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Détection des coins du damier
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                obj_points.append(objp)
                img_points.append(corners)

                # Affichage des coins détectés
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)

        # Calibration de la caméra
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        print("Calibration terminée !")
        print("Matrice de la caméra :\n", self.camera_matrix,)
        print("Coefficients de distorsion :\n", self.dist_coeffs)

    def save_calib(self):
        np.savez(self.data_name, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
        print("Sauvegarde terminée !")



