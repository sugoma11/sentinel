import cv2
import numpy as np
import os

colors = {
    0: (0, 0, 0), # clear
    1: (0, 0, 255), # dirty
    2: (255, 0, 0), # ship
    3: (255, 255, 255) # cloud
}

def mouse_click(event, x, y, flags, param):
    global poly, current_polygon, selected_class, img
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

    cv2.imshow('tst', img2)
    # print(event, x, y, flags, param)

poly = [[[]] for i in range(4)]


def define_coor(pic):
    global img, poly, cur_class, selected_class, current_polygon
    img = pic
    selected_class = 0
    current_polygon = 0
    cv2.namedWindow('tst')
    cv2.setMouseCallback('tst', mouse_click)
    while True:
        key = cv2.waitKey(0)

        if key == 120 or key == 247: # X
            if current_polygon == len(poly[selected_class]) - 1:
                poly[selected_class].append([])
            current_polygon += 1

        if key == 122 or key == 255: # Z
            if current_polygon > 0:
                current_polygon -= 1

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


img = cv2.imread('mask.jpg')

#water_mask = np.load('cont.npy')
#define_coor(img)
#cv2.drawContours(img, [water_mask], 0, (0, 255, 255), -1)


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

define_coor(img)
drawed = fill_pollys(img, poly)
cv2.imwrite('drawed.jpg', drawed)
cv2.imshow('drawed', drawed)
cv2.waitKey(0)
