import os
import cv2

from stereo_sys import StereoSys


def main():
    # Param de sauvegarde 
    folder_image_1 = "images_calib"
    folder_image_2 = "images_calib"
    folder_image_stereo = "images_calib_stereo"
    folder_calibration = "data_calib"

    #list_cameras = find_cameras()
    list_cameras = [0, 2]

    # Initialisation du système stéréo
    stereo_sys = StereoSys("calibration_data", list_cameras[0], list_cameras[1], debug_mode=True)

    # Calibration des caméras
    load_path = os.path.join(folder_calibration, "calibration_data_1.npz")
    if os.path.exists(load_path) :
        req_calib_1 = input("Voulez-vous refaire la calibration de la camera 1 ? (y/n)")
    else :
        req_calib_1 = "y"

    if req_calib_1 == "y" :
        req_n = int(input("Avec combien d'images voulez-vous calibrer la caméras 1 ?"))
        stereo_sys.cam_1.take_photo_calib(req_n, "calib_1", folder_name=folder_image_1)
        stereo_sys.cam_1.calibration("calib_1", folder_name=folder_image_1, debug_mode=True)
        stereo_sys.cam_1.save_calib(folder_calibration)
    else :
        stereo_sys.cam_1.load_calib(folder_calibration)

    load_path = os.path.join(folder_calibration, "calibration_data_2.npz")
    if os.path.exists(load_path) :
        req_calib_2 = input("Voulez-vous refaire la calibration de la camera 2 ? (y/n)")
    else :
        req_calib_2 = "y"

    if req_calib_2 == "y" :
        req_n = int(input("Avec combien d'images voulez-vous calibrer la caméras 2 ?"))
        stereo_sys.cam_2.take_photo_calib(req_n, "calib_2", folder_name=folder_image_2)
        stereo_sys.cam_2.calibration("calib_2", folder_name=folder_image_2, debug_mode=True)
        stereo_sys.cam_2.save_calib(folder_calibration)
    else :
        stereo_sys.cam_2.load_calib(folder_calibration)


    req_calibration = input("Voulez-vous commencer la prise de photo stereo ? (y/n)")
    if req_calibration == "y" :
        stereo_sys.take_photos(folder_name=folder_image_stereo)
        stereo_sys.stereo_calibration()
        stereo_sys.stereo_rectify()
        stereo_sys.find_disparity()
        stereo_sys.disparity_to_pointcloud(0)
        stereo_sys.display_point_cloud()


def find_cameras():
    # Fonction pour trouver les caméras disponibles
    liste_cameras = []
    for i in range(10):  # Vérifie jusqu'à 10 caméras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            liste_cameras.append(i)
            cap.release()

    if len(liste_cameras) == 0:
        print("Aucune caméra trouvée")
        return []
    elif len(liste_cameras) == 1:
        print("Branchez une deuxieme camera")
    elif len(liste_cameras) == 3:
        camera_defaut = input("Avez-vous une caméra par défaut ? (y/n)")
        if camera_defaut == "y":
            del liste_cameras[0]
    elif len(liste_cameras) > 3:
        print("Trop de cameras trouvées")

    print(f"Caméras trouvées : {liste_cameras}")
    return liste_cameras


if __name__ == "__main__":
    main()

