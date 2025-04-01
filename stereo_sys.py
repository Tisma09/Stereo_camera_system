import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import struct
import open3d as o3d

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
        save_path_1 = "Image_cam_1.png"
        save_path_2 = "Image_cam_2.png"
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            save_path_1 = os.path.join(folder_name, "Image_cam_1.png")
            save_path_2 = os.path.join(folder_name, "Image_cam_2.png")

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
        K1 = self.cam_1.K
        K2 = self.cam_2.K
        self.E = K2.T @ self.F @ K1

        # Rotation et translation
        _, self.R, self.T, mask = cv2.recoverPose(self.E, pts1, pts2, K1)

        

        # Verif lignes épipolaires
        if self.debug_mode:
            print("Calibration stéréo terminée !")
            print("Matrice de rotation :\n", self.R)
            print("Vecteur de translation :\n", self.T)
            print("Matrice essentielle :\n", self.E)
            print("Matrice fondamentale :\n", self.F)

            img1_with_lines = self.show_lines(self.image_1, pts1, 1)
            img2_with_lines = self.show_lines(self.image_2, pts2, 2)
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
            self.cam_1.K, self.cam_1.dist,
            self.cam_2.K, self.cam_2.dist,
            image_size, self.R, self.T
        )
        # Calcul des images rectifiées
        map1x, map1y = cv2.initUndistortRectifyMap(self.cam_1.K, self.cam_1.dist, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.cam_2.K, self.cam_2.dist, R2, P2, image_size, cv2.CV_32FC1)
        # Appliquer la rectification stéréo sur les images
        self.image_1 = cv2.remap(self.image_1, map1x, map1y, cv2.INTER_LINEAR)
        self.image_2 = cv2.remap(self.image_2, map2x, map2y, cv2.INTER_LINEAR)
        self.gray_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        self.gray_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)

        
        if self.debug_mode:
            
            matches, keypoints1, keypoints2 = self.feature_matching()
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
            img1_with_lines = self.show_lines(self.image_1, pts1, 1)
            img2_with_lines = self.show_lines(self.image_2, pts2, 2)
            cv2.imshow("Lignes Epipolaires - Image 1", img1_with_lines)
            cv2.imshow("Lignes Epipolaires - Image 2", img2_with_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def find_disparity(self):
        stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities,preFilterCap = 1, blockSize = 5, uniquenessRatio = 2, speckleWindowSize = 50, speckleRange = 2, disp12MaxDiff = 1, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2,mode = 4)
        self.disparity = stereo.compute(self.gray_1,self.gray_2).astype(np.float32)
        self.disparity = cv2.medianBlur(self.disparity, 5)
        #self.disparity = cv2.bilateralFilter(self.disparity,9,75,75)

        if self.debug_mode:
            print("Disparité calculée !")
            plt.imshow(self.disparity,"jet")
            plt.show()


    def disparity_to_pointcloud(self, image_num):
        color = self.image_1 if image_num == 0 else self.image_2
        point_cloud = cv2.reprojectImageTo3D(self.disparity, self.Q)
        mask = self.disparity > self.disparity.min()
        xp = point_cloud[:,:,0]
        yp = point_cloud[:,:,1]
        zp = point_cloud[:,:,2]
        color = color[mask]
        self.color = color.reshape(-1,3)
        xp = xp[np.where(mask == True)[0],np.where(mask == True)[1]]
        yp = yp[np.where(mask == True)[0],np.where(mask == True)[1]]
        zp = zp[np.where(mask == True)[0],np.where(mask == True)[1]]

        xp=xp.flatten().reshape(-1,1)
        yp=yp.flatten().reshape(-1,1)
        zp=zp.flatten().reshape(-1,1)
        self.point3d = np.hstack((xp,yp,zp))
        print(self.point3d.shape)
        print(self.color.shape)


    def display_point_cloud(self):
        # Vérification des dimensions
        assert self.point3d.shape[1] == 3, "self.point3d doit avoir une forme (N, 3)"
        assert self.color.shape[1] == 3, "self.color doit avoir une forme (N, 3)"
        assert self.point3d.shape[0] == self.color.shape[0], "self.point3d et self.color doivent avoir le même nombre de points"

        # Filtrer les points invalides
        valid_mask = np.isfinite(self.point3d).all(axis=1)
        self.point3d = self.point3d[valid_mask]
        self.color = self.color[valid_mask]

        # Créer un nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point3d)
        pcd.colors = o3d.utility.Vector3dVector(self.color.astype(np.float32) / 255.0)

        # Afficher le nuage de points
        o3d.visualization.draw_geometries([pcd])

    
    def write_pointcloud_ply(self, filename):
        assert self.point3d.shape[1] == 3,'Input XYZ points should be Nx3 float array'
        if self.color is None:
            self.color = np.ones(self.point3d.shape).astype(np.uint8)*255
        assert self.point3d.shape == self.color.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
        # Écrire l'en-tête du fichier .ply
        print("opening file")
        fid = open(filename,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%self.point3d.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Écrire des points 3D dans un fichier .ply
        for i in range(self.point3d.shape[0]):
            if(i%5000 == 0):
                print(i)
            fid.write(bytearray(struct.pack("fffccc",self.point3d[i,0],self.point3d[i,1],self.point3d[i,2],
                                            self.color[i,2].tobytes(),self.color[i,1].tobytes(),
                                            self.color[i,1].tobytes())))
        fid.close()

  

    #####################################################
    #############         Debug Mode        #############
    #####################################################

    def show_lines(self, image, pts, num):
        lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), num, self.F)
        lines = lines.reshape(-1, 3)

        img_with_lines = image.copy()
        for r in lines:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img_with_lines.shape[1], -(r[2] + r[0] * img_with_lines.shape[1]) / r[1]])
            cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)

        return img_with_lines

