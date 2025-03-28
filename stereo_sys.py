import numpy as np
import cv2
import glob

from camera import Camera 

pattern_size = (7, 9)
square_size = 2

class StereoSys():

    def __init__(self, data_name, cap_id_1, cap_id_2):
        self.data_name = data_name
        self.cam_1 = Camera(data_name+"_1", cap_id_1)
        self.cam_2 = Camera(data_name+"_2", cap_id_2)
        self.R = None
        self.T = None
        self.E = None
        self.F = None

    def stereo_calibration(self):
        i = 0
        while i < 1:
            ret1, frame1 = self.cam_1.cap.read()
            ret2, frame2 = self.cam_2.cap.read()
            if not ret1 or not ret2:
                print("Erreur de capture")
                break

            cv2.imshow("Capture 1", frame1)
            cv2.imshow("Capture 2", frame2)

            key = cv2.waitKey(1)
            if key == ord('s'):  # Appuyer sur 's' pour sauvegarder l'image
                cv2.imwrite(f"Image_cam_1.jpg", frame1)
                print(f"Image 1 enregistrée")
                cv2.imwrite(f"Image_cam_2.jpg", frame2)
                print(f"Image 2 enregistrée")
                i+= 1

        self.cam_1.cap.release()
        self.cam_2.cap.release()
        cv2.destroyAllWindows()

        image_1 = cv2.imread("Image_cam_1.jpg", cv2.IMREAD_COLOR)
        image_2 = cv2.imread("Image_cam_2.jpg", cv2.IMREAD_COLOR)

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        # Détection de point d'intérets SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray_1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_2, None)

        # Association des points
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)  # Trier par distance

        # Montre les meilleiurs matches
        matched_img = cv2.drawMatches(gray_1, keypoints1, gray_2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Correspondances SIFT", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(matches) < 8:
            print("Pas assez de points d'intérêt correspondants trouvés.")
            return

        # Extraction des points correspondants
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches[:10]])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches[:10]])

        # Matrice fondamentale avec RANSAC
        self.F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Matrice essentielle
        K1 = self.cam_1.camera_matrix
        K2 = self.cam_2.camera_matrix
        self.E = K2.T @ self.F @ K1

        # Rotation et translation
        _, self.R, self.T, mask = cv2.recoverPose(self.E, pts1, pts2, K1)

        print("Calibration stéréo terminée !")
        print("Matrice de rotation :\n", self.R)
        print("Vecteur de translation :\n", self.T)
        print("Matrice essentielle :\n", self.E)
        print("Matrice fondamentale :\n", self.F)



    def save_calibration(self):
        np.savez(self.data_name, R=self.R, T=self.T, E=self.E, F=self.F)
        print("Sauvegarde terminée !")



    def load_calibration(self):
        npzfile = np.load(self.data_name + ".npz")
        self.R = npzfile['R']
        self.T = npzfile['T']
        self.R = npzfile['E']
        self.T = npzfile['F']
        print("Sauvegarde chargé !")
