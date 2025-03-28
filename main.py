import os

from stereo_sys import StereoSys


def main():
    # Param de sauvegarde 
    folder_image_1 = "images_calib"
    folder_image_2 = "images_calib"
    folder_image_stereo = "images_calib_stereo"
    folder_calibration = "data_calib"

    # Initialisation du système stéréo
    stereo_sys = StereoSys("calibration_data", 2, 1, debug_mode=True)

    # Calibration des caméras
    load_path = os.path.join(folder_calibration, "calibration_data_1.npz")
    if os.path.exists(load_path) :
        req_calib_1 = input("Voulez-vous refaire la calibration de la camera 1 ? (y/n)")
    else :
        req_calib_1 = "y"

    if req_calib_1 == "y" :
        req_n = int(input("Avec combien d'images voulez-vous calibrer la caméras 1 ?"))
        x_photo = stereo_sys.cam_1.take_photo(req_n, "calib_1", folder_name=folder_image_1)
        x_calib = stereo_sys.cam_1.calibration("calib_1", folder_name=folder_image_1)
        if x_photo == 1 or x_calib == 1:
            print("Erreur lors de la calibration de la caméra 1")
            return
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
        x_photo = stereo_sys.cam_2.take_photo(req_n, "calib_2", folder_name=folder_image_2)
        x_calib = stereo_sys.cam_2.calibration("calib_2", folder_name=folder_image_2)
        if x_photo == 1 or x_calib == 1:
            print("Erreur lors de la calibration de la caméra 2")
            return
        stereo_sys.cam_2.save_calib(folder_calibration)
    else :
        stereo_sys.cam_2.load_calib(folder_calibration)


    req_calibration = input("Voulez-vous commencer la calibration stenreo ? (y/n)")
    if req_calibration == "y" :
        x_calib = stereo_sys.stereo_calibration(folder_name=folder_image_stereo)
        if x_calib == 1:
            print("Erreur lors de la calibration stéréo")
            return
        stereo_sys.save_calibration(folder_calibration)
    else :
        try :
            stereo_sys.load_calibration(folder_calibration)
        except :
            print("Error : Pas de donnée de calibration trouvé")
            return


if __name__ == "__main__":
    main()

