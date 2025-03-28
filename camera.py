import cv2
import numpy as np
import glob
import os

pattern_size = (7, 9)
square_size = 2

class Camera():

    def __init__(self, data_name, cap_id):
        self.data_name = data_name
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cap_id = cap_id



    def take_photo(self, n, name_file, folder_name=None): 
        save_path = name_file
        if folder_name:
            save_path = os.path.join(folder_name, name_file)
        
        sucess = 0
        cap = cv2.VideoCapture(self.cap_id)
        i = 0
        while i < n:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture")
                sucess = 1
                break

            cv2.imshow("Capture du damier", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):  # Appuyer sur 's' pour sauvegarder l'image
                img_path = f"{save_path}_{i}.jpg"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Crée le dossier si nécessaire
                cv2.imwrite(img_path, frame)
                print(f"Image {img_path} enregistrée")
                i += 1

            elif key == ord('q'):
                sucess = 1
                break

        cap.release()
        cv2.destroyAllWindows()
        return sucess

    #####################################################
    #############    Camera Calibration    ##############
    #####################################################

    def calibration(self, name_file, folder_name=None):
        # Préparation des points 3D réels
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Stockage des points
        obj_points = []  # Points 3D réels
        img_points = []  # Points 2D détectés

        # Construire le chemin pour la recherche des images
        search_path = f"{name_file}_*.jpg"
        if folder_name:
            search_path = os.path.join(folder_name, f"{name_file}_*.jpg")

        # Charger les images du damier
        images = glob.glob(search_path)
        gray = None

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
        cv2.destroyAllWindows()

        # Calibration de la caméra
        try :
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        except Exception as e:
            print(f"Erreur lors de la calibration : {e}")
            print(f"Vérifier la taille de votre damier : {pattern_size}")
            return 1

        print("Calibration terminée !")
        print("Matrice de la caméra :\n", self.camera_matrix,)
        print("Coefficients de distorsion :\n", self.dist_coeffs)

        return 0



    def save_calib(self, folder_name=None):
        save_path = self.data_name
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            save_path = os.path.join(folder_name, self.data_name)
        np.savez(save_path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
        print(f"Sauvegarde terminée dans {save_path}.npz !")

    def load_calib(self, folder_name=None):
        load_path = self.data_name
        if folder_name:
            load_path = os.path.join(folder_name, self.data_name)
        npzfile = np.load(f"{load_path}.npz")
        self.camera_matrix = npzfile['camera_matrix']
        self.dist_coeffs = npzfile['dist_coeffs']
        print(f"Fichier {load_path}.npz chargé !")



