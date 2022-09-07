import numpy as np
import cv2


def dot_edges(image, port_points, out_points):
    contours = np.array([[[port_points[0]]], [[port_points[1]]], [[port_points[2]]], [[port_points[3]]]], dtype=int)
    contours_outs = np.array([[[out_points[0]]], [[out_points[1]]], [[out_points[2]]], [[out_points[3]]]], dtype=int)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 7)
    cv2.drawContours(image, contours_outs, -1, (0, 0, 255), 7)
    return image