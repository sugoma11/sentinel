import cv2
import numpy as np
import os

colors = {
    0: (0, 255, 0), # clear
    1: (0, 0, 255), # dirty
    2: (255, 0, 0), # ship
    3: (0, 255, 255) # cloud
}

def mouse_click(event, x, y, flags, param):
    global poly, current_polygon, selected_class, img, sea_mask
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        poly[selected_class][current_polygon].append((x, y))
    for cur_class in range(len(poly)):
        for cur_polygon in range(len(poly[cur_class])):
            if len(poly[cur_class][cur_polygon]) > 0:
                cv2.circle(img2, poly[cur_class][cur_polygon][-1], 1, (0, 255, 255), -1)
            if len(poly[cur_class][cur_polygon]) > 1:
                for i in range(len(poly[cur_class][cur_polygon]) - 1):
                    cv2.circle(img2, poly[cur_class][cur_polygon][i], 1, (0, 255, 255), -1)
                    cv2.line(img2, poly[cur_class][cur_polygon][i],
                             poly[cur_class][cur_polygon][i + 1],
                             (255, 0, 0), 2)
                cv2.line(img2, poly[cur_class][cur_polygon][0],
                     poly[cur_class][cur_polygon][-1],
                     (255, 0, 0), 2)
                mask = np.zeros(img2.shape, np.uint8)
                points = np.array(poly[cur_class][cur_polygon], np.int32)
                points = points.reshape((-1, 1, 2))

                if cur_class == 0:
                    color = (0, 255, 0)
                elif cur_class == 1:
                    color = (0, 0, 255)
                elif cur_class == 2:
                    color = (255, 0, 0)
                elif cur_class == 3:
                    color = (255, 255, 255)

                mask = cv2.fillPoly(mask, [points], color)
                img2 = cv2.addWeighted(src1=img2, alpha=1, src2=mask, beta=.2, gamma=0)
    if event == cv2.EVENT_RBUTTONDOWN:
        poly[selected_class][current_polygon].pop()

    cv2.imshow('DrawContours', img2)
    cv2.imshow('SeaMask', sea_mask)
    # print(event, x, y, flags, param)

poly = [[[]] for i in range(4)]


def fill_clouds(image, sea_mask):
    clouds_mask = np.where(image[:, :, 2] < 240, 0, 255).astype('uint8')
    contours, hierarchy = cv2.findContours(clouds_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cv2.drawContours(sea_mask, contours, i, (255, 0, 0), -1)


def define_coor(pic):
    global img, poly, cur_class, selected_class, current_polygon, sea_mask
    img = pic
    selected_class = 0
    current_polygon = 0

    cv2.namedWindow('DrawContours')
    cv2.setMouseCallback('DrawContours', mouse_click)

    cv2.namedWindow('SeaMask')

    while True:
        key = cv2.waitKey(0)

        if key == 120 or key == 247: # X
            if current_polygon == len(poly[selected_class]) - 1:
                poly[selected_class].append([])
            current_polygon += 1

        if key == 122 or key == 255: # Z
            if current_polygon > 0:
                current_polygon -= 1

        if key == 118: # C
            fill_clouds(img, sea_mask)

        if key == 48: # 0
            selected_class = 0
            current_polygon = len(poly[selected_class]) - 1

        if key == 32: # Space
            pass

        if key == 49: # 1
            selected_class = 1
            current_polygon = len(poly[selected_class]) - 1

        if key == 50: # 2
            selected_class = 2
            current_polygon = len(poly[selected_class]) - 1

        if key == 51: # 3
            selected_class = 3
            current_polygon = len(poly[selected_class]) - 1

        if key != -1:
            pass
        if key == 27: # ESCape
            break
    cv2.destroyAllWindows()


def fill_pollys(image, poly):
    for i, cls in enumerate(poly):
        for zone in cls:
            if len(zone):
                cont = []
                for pair in zone:
                    cont.append([[pair[0], pair[1]]])
                cont = np.asarray(cont, dtype='int32')
                cont = cont.reshape((-1, 1, 2))
                cv2.drawContours(image, [cont], 0, colors[i], -1)

    return image


img = cv2.imread('2022-07-12-00_00_2022-07-12-23_59_Sentinel-2_L2A_True_color.png')
sea_mask = img.copy()
water_mask = np.load('cont.npy')
cv2.drawContours(sea_mask, [water_mask], 0, (0, 255, 0), -1)

define_coor(img)

drawed = fill_pollys(sea_mask, poly)
cv2.imshow('drawed', drawed)
cv2.waitKey(0)
