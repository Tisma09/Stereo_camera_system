import os

from stereo_sys import StereoSys


def main():
    stereo_sys = StereoSys("calibration_data", 2, 1)
    if os.path.exists("calibration_data_1.npz") :
        req_calib_1 = input("Voulez-vous refaire la calibration de la camera 1 ? (y/n)")
    else :
        req_calib_1 = "y"

    if req_calib_1 == "y" :
        req_n = int(input("Avec combien d'images voulez-vous calibrer la caméras 1 ?"))
        stereo_sys.cam_1.take_photo(req_n, "calib_1")
        stereo_sys.cam_1.calibration("calib_1")
        stereo_sys.cam_1.save_calib()
    else :
        stereo_sys.cam_1.load_calib()


    if os.path.exists("calibration_data_2.npz") :
        req_calib_2 = input("Voulez-vous refaire la calibration de la camera 2 ? (y/n)")
    else :
        req_calib_2 = "y"

    if req_calib_2 == "y" :
        req_n = int(input("Avec combien d'images voulez-vous calibrer la caméras 2 ?"))
        stereo_sys.cam_2.take_photo(req_n, "calib_2")
        stereo_sys.cam_2.calibration("calib_2")
        stereo_sys.cam_2.save_calib()
    else :
        stereo_sys.cam_2.load_calib()


    req_calibration = input("Voulez-vous commencer la calibration stenreo ? (y/n)")
    if req_calibration == "y" :
        stereo_sys.stereo_calibration()
        stereo_sys.save_calibration()
    else :
        try :
            stereo_sys.load_calibration()
        except :
            print("Error : Pas de donnée de calibration trouvé")


if __name__ == "__main__":
    main()

