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
from typing import List
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from scipy import misc
from PIL import Image
from typing import List
from imageio import imread

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
import cv2
import numpy as np

import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 204901417


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = imread(filename)

    img_fl = img.astype(np.float64)

    if np.max(img_fl) > 1:
        img_fl /= 255  # normalization

    if representation == 1:
        img_fl = rgb2gray(img_fl)

    return img_fl


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert(filename, representation)

    plt.imshow(img, cmap=plt.cm.gray)

    plt.axis('off')

    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    trans = get_YIQ_trans()
    me = np.tensordot(imgRGB, trans, axes=(2, 1))
    return np.tensordot(imgRGB, trans, axes=(2, 1))


def get_YIQ_trans():
    mat = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
    return np.array(mat)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    trans = get_YIQ_trans()
    return np.tensordot(imgYIQ, np.linalg.inv(trans).copy(), axes=(2, 1))


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    if not isGrayScale(imOrig):
        img2 = transformRGB2YIQ(imOrig)
        # nis= img2
        img1 = img2[:, :, 0] #working on the y channel
    if isGrayScale(imOrig):
        img1 = imOrig.copy()
    hist_origi = np.histogram(img1.flatten(), 256)[0]
    cum_sum = np.cumsum(hist_origi)

    cum_sum = cum_sum / cum_sum[-1]
    cum_sum = cum_sum * 255
    cum_sum = np.round(cum_sum)

    imgnew = np.interp(img1 * 255, np.arange(256), cum_sum) #apllaying all the information we collect to the new y channel

    hist_eq = np.histogram(imgnew.flatten(), 256)[0]
    imgnew = imgnew / 255
    if not isGrayScale(imOrig): #if its not grey scal we need all 3 channel
        img2[:, :, 0] = imgnew
        img2 = transformYIQ2RGB(img2)
        imgnew = img2

    imgnew[imgnew < 0] = 0
    imgnew[imgnew > 1] = 1
    img3 = imgnew * 255
    img4 = img3.astype(int)
    imOrig = img4
    if isGrayScale(imOrig):
        imOrig = imgnew

    # print(np.max((nis-img2)),np.min(nis-img2))

    return imOrig, hist_origi, hist_eq




def isGrayScale(img):
    if (len(img.shape) < 3):
        return True
    else:
        return False


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int):
    if isGrayScale(imOrig):
        channel = np.copy(imOrig)
    if not isGrayScale(imOrig):
        im = transformRGB2YIQ(np.copy(imOrig))
        channel = np.copy(im[:, :, 0])  # working on the y channel
    K = 256
    # error = []
    my_list = []
    my_list_error = []
    hist, bounds = np.histogram(channel, K, range=(0, 1))  # chake range hist for yiq
    z, q = initialize_zq(hist, nQuant)
    index = len(q)
    #print("size of channel : ", np.size(channel))

    for i in range(nIter):
        z_prev = z.copy()
        optimize_z_q(z, q, hist) #chacking for optimize z,q
        save = apply_quantization(z, q, channel) # applying the quant
        #if np.array_equal(channel, save): #test for me
            #print("not gooodd at alll")
        error = np.sqrt(np.sum((save - channel) ** 2)) / (
            channel.size)
        if not isGrayScale(imOrig): # before insert to the list need all channel and transform to rgb
            im[:, :, 0] = save
            save = transformYIQ2RGB(np.copy(im))

        my_list.insert(i, save)
        my_list_error.insert(i, error)

        if np.array_equal(z, z_prev):
            break

    return my_list, my_list_error


def initialize_zq(hist, n_quant):
    z = np.zeros((n_quant + 1,))
    z[n_quant] = 255
    c_hist = np.cumsum(hist)
    for i in range(1, n_quant):
        z[i] = np.argmax(c_hist >= (i * c_hist[-1]) / n_quant)
        q = (z[:-1] + z[1:]) / 2
    return z, q


def optimize_z_q(z, q, hist):
    for i in range(len(q)):
        first, last = np.rint(z[i]).astype(np.int64), np.rint(z[i + 1] + 1).astype(np.int64)
        g = np.arange(first, last)
        g_h = (hist[first:last])
        q[i] = np.sum(g * g_h) / np.sum(g_h)
    for i in range(1, len(q)):
        z[i] = (q[i - 1] + q[i]) / 2
    return z, q


def apply_quantization(z, q, im_orig):
    K = 255

    copy_im = np.copy(im_orig)
    for i in range(len(q)):
        low = im_orig >= (z[i] / K)
        if i == len(q) - 1:
            high = im_orig <= z[i + 1] / K
        else:
            high = im_orig < z[i + 1] / K
        # replacing pixel values with quantized value:
        copy_im[(np.logical_and(low, high))] = q[i] / K
    return copy_im
