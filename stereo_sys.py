import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import struct

from camera import Camera 
from config import *

class StereoSys():

    def __init__(self, data_name, cap_id_1, cap_id_2, debug_mode=False):
        self.debug_mode = debug_mode

        self.data_name = data_name
        self.cam_1 = Camera(data_name+"_1", cap_id_1)
        self.cam_2 = Camera(data_name+"_2", cap_id_2)
        self.image_1, self.image_2 = None, None
        self.gray_1, self.gray_2 = None, None
        self.R, self.T, self.E, self.F = None, None, None, None
        self.Q = None
        self.disparity = None
        self.point3d, self.color = None, None
        
    #####################################################
    #############    Photo to process      ##############
    #####################################################

    def take_photos(self, folder_name=None):
        cap_1 = cv2.VideoCapture(self.cam_1.cap_id)
        cap_2 = cv2.VideoCapture(self.cam_2.cap_id)
        i = 0

        # Préparation chemins de sauvegarde
        save_path_1 = "Image_cam_1.jpg"
        save_path_2 = "Image_cam_2.jpg"
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            save_path_1 = os.path.join(folder_name, "Image_cam_1.jpg")
            save_path_2 = os.path.join(folder_name, "Image_cam_2.jpg")

        while i < 1:
            ret1, frame1 = cap_1.read()
            ret2, frame2 = cap_2.read()
            if not ret1 or not ret2:
                print("Erreur de capture")
                break

            cv2.imshow("Capture 1", frame1)
            cv2.imshow("Capture 2", frame2)

            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(save_path_1, frame1)
                print(f"Image 1 enregistrée dans {save_path_1}")
                cv2.imwrite(save_path_2, frame2)
                print(f"Image 2 enregistrée dans {save_path_2}")
                i+= 1
            elif key == ord('q'):
                break

        cap_1.release()
        cap_2.release()
        cv2.destroyAllWindows()

        # Lecture des images sauvegardées
        self.image_1 = cv2.imread(save_path_1)
        self.image_2 = cv2.imread(save_path_2)
        self.gray_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        self.gray_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)


    #####################################################
    ########## Feature Detection and Matching ###########
    #####################################################

    def feature_matching(self):
        # Détection de point d'intérets SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(self.gray_1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(self.gray_2, None)

        # Association des points
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Montre quelques matches
        if self.debug_mode:
            matched_img = cv2.drawMatches(self.gray_1, keypoints1, self.gray_2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Correspondances SIFT", matched_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return matches, keypoints1, keypoints2

    #####################################################
    #############    Stereo Calibration    ##############
    #####################################################

    def stereo_calibration(self):
        matches, keypoints1, keypoints2 = self.feature_matching()

        if len(matches) < 8:
            print("Pas assez de points d'intérêt correspondants trouvés.")
            return

        # Extraction des points correspondants
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Matrice fondamentale avec RANSAC
        self.F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Matrice essentielle
        K1 = self.cam_1.camera_matrix
        K2 = self.cam_2.camera_matrix
        self.E = K2.T @ self.F @ K1

        # Rotation et translation
        _, self.R, self.T, mask = cv2.recoverPose(self.E, pts1, pts2, K2)

        

        # Verif lignes épipolaires
        if self.debug_mode:
            print("Calibration stéréo terminée !")
            print("Matrice de rotation :\n", self.R)
            print("Vecteur de translation :\n", self.T)
            print("Matrice essentielle :\n", self.E)
            print("Matrice fondamentale :\n", self.F)

            img1_with_lines = self.show_lines(self.image_1, pts1)
            img2_with_lines = self.show_lines(self.image_2, pts2)
            cv2.imshow("Lignes Epipolaires - Image 1", img1_with_lines)
            cv2.imshow("Lignes Epipolaires - Image 2", img2_with_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    #####################################################
    #############   Stereo Rectification   ##############
    #####################################################

    def stereo_rectify(self):
        image_size = self.image_1.shape[:2][::-1] 
        # Pour image_size : les deux font la même taille donc peu importe 1 ou 2

        # Rectification stéréo
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.cam_1.camera_matrix, self.cam_1.dist_coeffs,
            self.cam_2.camera_matrix, self.cam_2.dist_coeffs,
            image_size, self.R, self.T
        )
        # Calcul des images rectifiées
        map1x, map1y = cv2.initUndistortRectifyMap(self.cam_1.camera_matrix, self.cam_1.dist_coeffs, R1, self.P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.cam_2.camera_matrix, self.cam_2.dist_coeffs, R2, self.P2, image_size, cv2.CV_32FC1)
        # Appliquer la rectification stéréo sur les images
        self.image_1 = cv2.remap(self.image_1, map1x, map1y, cv2.INTER_LINEAR)
        self.image_2 = cv2.remap(self.image_2, map2x, map2y, cv2.INTER_LINEAR)
        self.gray_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        self.gray_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)

        
        if self.debug_mode:
            
            matches, keypoints1, keypoints2 = self.feature_matching()
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
            img1_with_lines = self.show_lines(self.image_1, pts1)
            img2_with_lines = self.show_lines(self.image_2, pts2)
            cv2.imshow("Lignes Epipolaires - Image 1", img1_with_lines)
            cv2.imshow("Lignes Epipolaires - Image 2", img2_with_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def find_disparity(self):
        stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities,preFilterCap = 1, blockSize = 5, uniquenessRatio = 2, speckleWindowSize = 50, speckleRange = 2, disp12MaxDiff = 1, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2,mode = 4)
        disparity = stereo.compute(self.gray_1,self.gray_2).astype(np.float32)
        self.disparity = cv2.medianBlur(disparity, 5)
        #self.disparity = cv.bilateralFilter(disparity,9,75,75)

        if self.debug_mode:
            plt.imshow(disparity,"jet")
            plt.show()


    def disparity_to_pointcloud(self, image_num):
        image = self.image_1 if image_num == 0 else self.image_2
        color = cv2.imread(image)
        point_cloud = cv2.reprojectImageTo3D(self.disparity, self.Q)
        mask = self.disparity > self.disparity.min()
        xp=point_cloud[:,:,0]
        yp=point_cloud[:,:,1]
        zp=point_cloud[:,:,2]
        color = color[mask]
        self.color = color.reshape(-1,3)
        xp = xp[np.where(mask== True)[0],np.where(mask== True)[1]]
        yp = yp[np.where(mask== True)[0],np.where(mask== True)[1]]
        zp = zp[np.where(mask== True)[0],np.where(mask== True)[1]]

        xp=xp.flatten().reshape(-1,1)
        yp=yp.flatten().reshape(-1,1)
        zp=zp.flatten().reshape(-1,1)
        self.point3d = np.hstack((xp,yp,zp))

    
    def write_pointcloud(self, filename):
        assert self.xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
        if rgb_points is None:
            rgb_points = np.ones(self.xyz_points.shape).astype(np.uint8)*255
        assert self.xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
        # Écrire l'en-tête du fichier .ply
        print("opening file")
        fid = open(filename,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%self.xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Écrire des points 3D dans un fichier .ply
        for i in range(self.xyz_points.shape[0]):
            if(i%5000 == 0):
                print(i)
            fid.write(bytearray(struct.pack("fffccc",self.xyz_points[i,0],self.xyz_points[i,1],self.xyz_points[i,2],
                                                rgb_points[i,2].tobytes(),rgb_points[i,1].tobytes(),
                                                rgb_points[i,1].tobytes())))
        fid.close()

    #####################################################
    #############         Debug Mode        #############
    #####################################################

    def show_lines(self, image, pts):
        lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 2, self.F)
        lines = lines.reshape(-1, 3)

        img_with_lines = image.copy()
        for r in lines:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img_with_lines.shape[1], -(r[2] + r[0] * img_with_lines.shape[1]) / r[1]])
            cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)

        return img_with_lines

