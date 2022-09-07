from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
import cv2


def add_text(background, port, out_port):
    string_one = f'Port: M:{port.mean}; D:{port.disp}; E:{port.kurtosis}; A:{port.assym};'
    string_two = f'R:{port.r_ch}; G:{port.g_ch}; B:{port.b_ch}; H:{port.h_ch}; S:{port.s_ch}; V:{port.v_ch}'
    string_thr = f'Out: M:{out_port.mean}; D:{out_port.disp}; E:{out_port.kurtosis}; A:{out_port.assym};'
    string_fourth = f'R:{out_port.r_ch}; G:{out_port.g_ch}; B:{out_port.b_ch}; H:{out_port.h_ch}; S:{out_port.s_ch}; V:{out_port.v_ch}'
    string_fifth = f'Delta: M:{round(abs(out_port.mean - port.mean), 2)}; D:{round(abs(out_port.disp - port.disp), 2)};' \
                   f' E:{round(abs(out_port.kurtosis - port.kurtosis), 2)}; A:{round(abs(out_port.assym - port.assym), 2)}; '

    str_list = [string_one, string_two, string_thr, string_fourth, string_fifth]

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = [1000, 75]
    fontScale = 1
    fontColor = (0, 255, 255)
    thickness = 2
    lineType = 2

    for i in range(5):
        cv2.putText(background, str_list[i], bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        bottomLeftCornerOfText[1] += 30

