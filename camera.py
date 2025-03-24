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
        cap = cv2.VideoCapture(0)  # 0 pour la caméra principale

        i = 0
        while i < 1:
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

    def stereo_rectify(self, img1_path, img2_path):
        """
        Fonction pour effectuer une rectification stéréo avec deux images de la même caméra.
        Cette méthode applique la rectification stéréo aux deux images, en utilisant la calibration de la caméra.
        """
        # Charger les images de la caméra
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Convertir les images en niveaux de gris
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Trouver les coins du damier pour les deux images
        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

        if ret1 and ret2:
            # Terminer la détection des coins avec plus de précision
            cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            # Effectuer la stéréo rectification
            # Obtenez la matrice de rotation et de translation de la calibration
            R = np.eye(3)  # Identité pour une seule caméra
            T = np.zeros((3, 1))  # Pas de translation entre deux caméras

            # Appliquer la stéréo rectification
            image_size = gray1.shape[::-1]
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.camera_matrix, self.dist_coeffs, self.camera_matrix, self.dist_coeffs,
                image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

            # Calculer les cartes de transformation
            map1x, map1y = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, R1, P1, image_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, R2, P2, image_size, cv2.CV_32FC1)

            # Appliquer la rectification
            rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

            # Afficher les images rectifiées
            cv2.imshow("Rectified Image 1", rectified_img1)
            cv2.imshow("Rectified Image 2", rectified_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("Erreur de détection des coins du damier dans les images.")


