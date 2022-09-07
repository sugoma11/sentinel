import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


name_list = [["Hue", "Saturation", "Value"], ["B", "G", "R"]]
f, ax = plt.subplots(2, 3)
f.set_size_inches(19.2, 10)


def normalize_hsv(img):
    img[:, :, 0] = img[:, :, 0] * 2
    img[:, :, 1] = img[:, :, 1] / 2.55
    img[:, :, 2] = img[:, :, 2] / 2.55
    return img


def create_distr(port, out_port, l, j):
    if l == 0:
        port = normalize_hsv(cv2.cvtColor(port, cv2.COLOR_BGR2HSV))
        out_port = normalize_hsv(cv2.cvtColor(out_port, cv2.COLOR_BGR2HSV))
    port = port[:, :, j]
    out_port = out_port[:, :, j]
    x = [i for i in range(256)]
    y_port = [0 for i in range(256)]
    y_out = [0 for i in range(256)]

    for i in x:
        y_port[i] = (port == i).sum()
        y_out[i] = (out_port == i).sum()

    ax[l][j].xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax[l][j].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax[l][j].yaxis.set_major_locator(ticker.MultipleLocator(50000))
    ax[l][j].yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax[l][j].plot(x, y_out, label='out_port')
    ax[l][j].plot(x, y_port, label='port')
    ax[l][j].set_title(name_list[l][j] + ' channel')
    ax[l][j].legend(fontsize=10)


def plotter(port, out_port):
    for l in range(2):
        for j in range(3):
            create_distr(port, out_port, l, j)
    plt.savefig(f'C:/Users/gradu/Desktop/python/sentinel_remastered/pic_features/temp/distr_graphic')

    for l in range(2):
        for j in range(3):
            ax[l][j].clear()