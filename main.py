import os

from stereo_sys import StereoSys


def main():
    stereo_sys = StereoSys("calibration_data", 2, 3)
    if os.path.exists("calibration_data_1.npz") :
        req_calib_1 = input("Voulez-vous refaire la calibration pour la caméra 1 ? (y/n)")
    else :
        req_calib_1 = "y"
    if os.path.exists("calibration_data_2.npz") :
        req_calib_2 = input("Voulez-vous refaire la calibration pour la caméra 2 ? (y/n)")
    else :
        req_calib_2 = "y"

    if req_calib_1 == "y" :
        req_n = input("Avec combien d'images voulez-vous calibrer la caméra 1 ?")
        stereo_sys.cam_1.take_photo(req_n, "calib_1")
        stereo_sys.cam_1.calibration("calib_1")
        stereo_sys.cam_1.save_calib()
    else :
        stereo_sys.cam_1.load_calib()

    if req_calib_2 == "y" :
        req_n = input("Avec combien d'images voulez-vous calibrer la caméra 2 ?")
        stereo_sys.cam_2.take_photo(req_n, "calib_2")
        stereo_sys.cam_2.calibration("calib_2")
        stereo_sys.cam_2.save_calib()
    else :
        stereo_sys.cam_1.load_calib()

if __name__ == "__main__":
    main()

