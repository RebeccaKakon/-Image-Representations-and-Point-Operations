"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from __future__ import division
from __future__ import print_function

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

import argparse

import cv2
import numpy as np
# from _future_ import print_function

import argparse

from ex1_utils import LOAD_GRAY_SCALE


# i have comment in my read me for this function
def gammaDisplay(img_path: str, rep: int):
    """
        GUI for gamma correction
        :param img_path: Path to the image
        :param rep: grayscale(1) or RGB(2)
        :return: None
        """

    gamma_slider_max = 200
    if rep == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # open the photo in gray
    else:
        img = cv2.imread(img_path)  # open the photo in RGB

    def on_trackbar(param):  # this function calculate thegamma every time we move in the scayl and show the new imge
        if rep == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # open the photo in gray
        else:
            img = cv2.imread(img_path)  # open the photo in RGB
        temp = img  # copy of the imge
        gamma = cv2.getTrackbarPos('Gamma', 'GammaCorrection')  # get the position of the scayl
        gamma = float(gamma)
        gamma = gamma / 100  # calculate the gamma
        img = np.uint8(255 * np.power((temp / 255), gamma))  # calculate the new image
        cv2.imshow('GammaCorrection', img)
        pass

    cv2.namedWindow('GammaCorrection')
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, 'GammaCorrection', 0, gamma_slider_max,
                       on_trackbar)  # create the window with a scayl
    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv2.waitKey()


def nothing(x: int):
    pass


def main():
    gammaDisplay(('C:/Users/User/Desktop/vision/beach.jpg', LOAD_RGB))


if __name__ == '__main__':
    main()
