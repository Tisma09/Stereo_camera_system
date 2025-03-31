import cv2
import numpy as np
import glob
import os

from config import *


class Camera():
    def __init__(self, data_name, cap_id):
        self.cap_id = cap_id
        self.data_name = data_name
        self.K = None
        self.dist = None

    def take_photo_calib(self, n, name_file, folder_name=None): 
        save_path = name_file
        if folder_name:
            save_path = os.path.join(folder_name, name_file)
        
        cap = cv2.VideoCapture(self.cap_id)
        sucess = 0
        i = 0
        while i < n:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture")
                break

            cv2.imshow("Capture du damier", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                img_path = f"{save_path}_{i}.jpg"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(img_path, frame)
                print(f"Image {img_path} enregistrée")
                i += 1

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    

    #####################################################
    #############    Camera Calibration    ##############
    #####################################################

    def calibration(self, name_file, folder_name=None, debug_mode=False):
        # Préparation des points 3D réels
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Stockage des points
        obj_points = [] 
        img_points = [] 

        # Recherche des images
        search_path = f"{name_file}_*.jpg"
        if folder_name:
            search_path = os.path.join(folder_name, f"{name_file}_*.jpg")
        images = glob.glob(search_path)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Détection des coins du damier
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                obj_points.append(objp)
                img_points.append(corners)

        # Calibration de la caméra
        try :
            ret, self.K, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        except Exception as e:
            print(f"Erreur lors de la calibration : {e}")
            print(f"Vérifier la taille de votre damier : {pattern_size}")
            return

        if debug_mode:
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            print("Calibration terminée !")
            print("Coefficients de distorsion :\n", self.dist)
            print("Matrice de la caméra :\n", self.K,)
            print("Matrice de la caméra sans distortion :\n", self.K_undist,)


    #####################################################
    ###########    Sauvegarde et chargement    ##########
    #####################################################


    def save_calib(self, folder_name=None):
        save_path = self.data_name
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            save_path = os.path.join(folder_name, self.data_name)
        np.savez(save_path, K=self.K, dist=self.dist)
        print(f"Sauvegarde terminée dans {save_path}.npz !")

    def load_calib(self, folder_name=None):
        load_path = self.data_name
        if folder_name:
            load_path = os.path.join(folder_name, self.data_name)
        npzfile = np.load(f"{load_path}.npz")
        self.K = npzfile['K']
        self.dist = npzfile['dist']
        print(f"Fichier {load_path}.npz chargé !")



