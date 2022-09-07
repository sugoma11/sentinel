import numpy as np
import cv2
import os


def bitwise(img, port, out_port):
    graph = cv2.imread(f'C:/Users/gradu/Desktop/python/sentinel_remastered/pic_features/temp/distr_graphic.png')
    blank_port, blank_out_port = np.zeros((225, 225, 3), dtype='uint8'), np.zeros((225, 225, 3), dtype='uint8')
    blank_port[:, :, 0], blank_out_port[:, :, 0] = port.b_ch, out_port.b_ch
    blank_port[:, :, 1], blank_out_port[:, :, 1] = port.g_ch, out_port.g_ch
    blank_port[:, :, 2], blank_out_port[:, :, 2] = port.r_ch, out_port.r_ch
    img[img.shape[0] - blank_port.shape[0]:img.shape[0], img.shape[1] - blank_port.shape[0]:img.shape[1], :] = blank_port
    img[img.shape[0] - 2 * blank_port.shape[0]:img.shape[0] - blank_port.shape[0], img.shape[1] - blank_port.shape[0]:img.shape[1], :] = blank_out_port
    img = np.vstack((img, graph))
    return img