import cv2
import numpy as np
import glob

pattern_size = (7, 9)
square_size = 2

class Camera():

    def __init__(self, data_name, cap_id):
        self.data_name = data_name
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cap = cv2.VideoCapture(cap_id)


    def take_photo(self, n=10):

        i = 0
        while i < n:
            ret, frame = self.cap.read()
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

        self.cap.release()
        cv2.destroyAllWindows()

    #####################################################
    #############    Camera Calibration    ##############
    #####################################################

    def calibration(self, ):

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

    #####################################################
    #############   Stereo Rectification   ##############
    #####################################################

    def stereo_rectify(self, img1_path, img2_path):
        # Charger les images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Si les images ont des tailles différentes, redimensionner
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Charger les matrices de calibration et les coefficients de distorsion
        camera_matrix_1 = self.camera_matrix
        camera_matrix_2 = self.camera_matrix
        dist_coeffs_1 = self.dist_coeffs
        dist_coeffs_2 = self.dist_coeffs

        # Rotation et translation entre les deux caméras (exemple)
        R = np.eye(3)  # La matrice de rotation entre les caméras (par exemple une matrice identité)
        T = np.array([1, 0, 0], dtype=np.float32)  # Le vecteur de translation (exemple)
        
        # Taille de l'image : on prend uniquement la largeur et la hauteur (sans les canaux)
        image_size = img1.shape[:2][::-1]  # inverse pour (largeur, hauteur)

        # Rectification stéréo
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_1, dist_coeffs_1,
            camera_matrix_2, dist_coeffs_2,
            image_size, R, T
        )

        # Vous pouvez également afficher ou utiliser Q ici, qui est la matrice de transformation 4x4
        print("Matrice Q :\n", Q)
        
        # Calcul des images rectifiées
        map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, image_size, cv2.CV_32FC1)
        
        # Appliquer la rectification stéréo sur les images
        rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        # Afficher les images rectifiées
        cv2.imshow("Rectified Image 1", rectified_img1)
        cv2.imshow("Rectified Image 2", rectified_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #####################################################
    #############    Feature Matching (SIFT)    ##########
    #####################################################

    def feature_matching(self, img1, img2):
        # Convertir les images en niveaux de gris
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialiser l'algorithme SIFT
        sift = cv2.SIFT_create()

        # Extraire les points caractéristiques et leurs descripteurs
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Utiliser un matcheur de descripteurs avec KNN
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # crossCheck=False pour le KNN
        matches = bf.knnMatch(des1, des2, k=2)

        # Filtrer les bonnes correspondances en utilisant le ratio de Lowe
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # ratio de Lowe
                good_matches.append(m)

        # Dessiner les correspondances filtrées
        result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Afficher l'image avec les correspondances
        cv2.imshow('Good Matches', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return good_matches







