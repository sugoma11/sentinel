from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
import cv2


def normalize_hsv(img):
    img[:, :, 0] = img[:, :, 0] * 2
    img[:, :, 1] = img[:, :, 1] / 2.55
    img[:, :, 2] = img[:, :, 2] / 2.55
    return img


def add_moments(port, out_port):
    port_hsv = cv2.cvtColor(port.image, cv2.COLOR_BGR2HSV)
    out_port_hsv = cv2.cvtColor(out_port.image, cv2.COLOR_BGR2HSV)

    port_hsv = normalize_hsv(port_hsv)
    out_port_hsv = normalize_hsv(out_port_hsv)

    port.h_ch = round(np.mean(port_hsv[:, :, 0]), 2)
    out_port.h_ch = round(np.mean(out_port_hsv[:, :, 0]), 2)

    port.s_ch = round(np.mean(port_hsv[:, :, 1]), 2)
    out_port.s_ch = round(np.mean(out_port_hsv[:, :, 1]), 2)

    port.v_ch = round(np.mean(port_hsv[:, :, 2]), 2)
    out_port.v_ch = round(np.mean(out_port_hsv[:, :, 2]), 2)

    port.r_ch = round(np.mean(port.image[:, :, 2]), 2)
    out_port.r_ch = round(np.mean(out_port.image[:, :, 2]), 2)
    
    port.g_ch = round(np.mean(port.image[:, :, 1]), 2)
    out_port.g_ch = round(np.mean(out_port.image[:, :, 1]), 2)
    
    port.b_ch = round(np.mean(port.image[:, :, 0]), 2)
    out_port.b_ch = round(np.mean(out_port.image[:, :, 0]), 2)
    
    port.mean = round(np.mean(port_hsv[:, :, 0]), 2)
    out_port.mean = round(np.mean(out_port_hsv[:, :, 0]), 2)
    
    port.disp = round(np.var(port_hsv[:, :, 0]), 2)
    out_port.disp = round(np.var(out_port_hsv[:, :, 0]), 2)
    
    port.kurtosis = round(kurtosis(port_hsv[:, :, 0].reshape(1, -1)[0]), 2)
    out_port.kurtosis = round(kurtosis(out_port_hsv[:, :, 0].reshape(1, -1)[0]), 2)
    
    port.assym = round(float(skew(port_hsv[:, :, 0].reshape(-1, 1))), 2)
    out_port.assym = round(float(skew(out_port_hsv[:, :, 0].reshape(-1, 1))), 2)
    