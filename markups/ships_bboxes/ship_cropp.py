import time

import cv2
import numpy as np
import os
from pathlib import Path

colors = {
    0: (0, 255, 0), # clear
    1: (0, 0, 255), # dirty
    2: (255, 0, 0), # ship
    3: (0, 255, 255), # cloud
    4: (255, 0, 0), # ship
    5:  (0, 255, 255) # cloud
}

names = {
    0: 'clear',
    1: 'dirty',
    2: 'ship',
    3: 'cloud',
    4: 'ship contour',
    5: 'cloud contour'
}

colors_marks = {
    (0, 255, 0): 0,
    (0, 0, 255): 1,
    (255, 0, 0): 2,
    (0, 255, 255): 3,
}

PORT_NAME = 'gelendzh'
# delta of xxyy
DXDY = 3

data_dir = f'../../data/raw/{PORT_NAME}'

if not os.path.isdir(f'{data_dir}/marked'):
    os.mkdir(f'{data_dir}/marked')

if len(list(Path(f'{data_dir}/marked').glob('*.png'))) == 0:
    num = 0

else:
    num = len((sorted(Path(f'{data_dir}/marked').glob('*.png'), key=os.path.getmtime)))

def mouse_click(event, x, y, flags, param):
    global poly, current_polygon, selected_class, img, sea_mask, ships_masks
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:

        if selected_class == 4 or selected_class == 5:
            for xxyy in ships_masks:
                if (xxyy[0][0] < x < xxyy[2][0]) and (xxyy[3][1] < y < xxyy[1][1]):
                    # print(xxyy)
                    cn = []
                    for i, pair in enumerate(xxyy):
                        cn.append((pair[0], pair[1]))
            try:
                poly[selected_class][current_polygon] = cn
            except UnboundLocalError:
                pass

        else:
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

                mask = cv2.fillPoly(mask, [points], colors[cur_class])
                img2 = cv2.addWeighted(src1=img2, alpha=1, src2=mask, beta=.2, gamma=0)
    if event == cv2.EVENT_RBUTTONDOWN:
        poly[selected_class][current_polygon].pop()

    cv2.putText(img2, str(names[selected_class]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.putText(img2, f'current_polygon: {current_polygon}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('DrawContours', img2)

    # print(event, x, y, flags, param)


def ships_finder(image):
    global ships_masks
    # temp = image.copy()
    # temp = np.where(temp == 0, 255, 0).astype('uint8')
    # temp = cv2.bitwise_or(image, image, mask=temp[:, :, 0])
    ships_mask = np.where(image[:, :, 2] > 60, 0, 255).astype('uint8')

    ships_contours, hierarchy = cv2.findContours(ships_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ships_contours = sorted(ships_contours, key=len)
    ships_contours.pop()

    ships_masks = []

    for cont in ships_contours:
        cont = cont.reshape(cont.shape[0], cont.shape[2])

        x_min = np.min(cont[:, 0])
        x_max = np.max(cont[:, 0])
        y_min = np.min(cont[:, 1])
        y_max = np.max(cont[:, 1])

        ships_masks.append([[x_min - 10, y_min - 10], [x_min - 10, y_max + 10], [x_max + 10, y_max + 10], [x_max + 10, y_min - 10]])


def define_coor(pic):
    global img, poly, cur_class, selected_class, current_polygon, sea_mask
    img = pic
    selected_class = 0
    current_polygon = 0

    cv2.namedWindow('DrawContours')
    cv2.setMouseCallback('DrawContours', mouse_click)

    while True:
        key = cv2.waitKey(0)
         # print(key)

        if key == 120 or key == 247: # X
            if current_polygon == len(poly[selected_class]) - 1:
                poly[selected_class].append([])
            current_polygon += 1

        if key == 100: # D - Delete
            if current_polygon >= 1:
                current_polygon -= 1
                poly[selected_class].clear()

        if key == 122 or key == 255: # Z
            if current_polygon > 0:
                current_polygon -= 1

        if key == 99: # C - Cloud contours
            selected_class = 5
            current_polygon = len(poly[selected_class]) - 1

        if key == 48: # 0
            selected_class = 0
            current_polygon = len(poly[selected_class]) - 1

        if key == 115: # S - Ships and contours
            selected_class = 4
            current_polygon = len(poly[selected_class]) - 1


        if key == 117: # U - upscale box
            tmp = [list(ele) for ele in poly[selected_class][current_polygon]]

            tmp[0][0] -= 5
            tmp[0][1] -= 5
            tmp[1][0] -= 5
            tmp[1][1] += 5

            tmp[2][0] += 5
            tmp[2][1] += 5
            tmp[3][1] -= 5
            tmp[3][0] += 5

            poly[selected_class][current_polygon] = tmp

        if key == 100: # D - Downscale box
            tmp = [list(ele) for ele in poly[selected_class][current_polygon]]

            tmp[0][0] += 5
            tmp[0][1] += 5
            tmp[1][0] += 5
            tmp[1][1] -= 5

            tmp[2][0] -= 5
            tmp[2][1] -= 5
            tmp[3][1] += 5
            tmp[3][0] -= 5

            poly[selected_class][current_polygon] = tmp


        if key == 103 and selected_class == 4:
            poly[3][current_polygon] = poly[selected_class][current_polygon]
            poly[selected_class][current_polygon] = []
            # poly[selected_class][current_polygon].clear()
            # poly[3][len(poly[3]) - 1] = tmp

        # Save image
        if key == 32: # Space
            for i, cls in enumerate(poly):
                for zone in cls:
                    if len(zone) == 4:
                        crop_img = img[zone[0][1]: zone[1][1], zone[0][0]: zone[2][0]]
                        tst = cv2.imwrite(f'marked/{time.time()}.jpg', crop_img)
            break


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

        if key == 98: # B - Break
            exit()

        if key == 27: # ESCape
            break

    cv2.destroyAllWindows()



ls = os.listdir(data_dir)
os.chdir(data_dir)

# print(len(ls))

for i in range(num, len(ls)):
    poly = [[[]] for i in range(6)]

    if ls[i].endswith('jpg') or ls[i].endswith('png'):
        img = cv2.imread(ls[i])
    else:
        continue

    sea_mask = img.copy()
    ships_finder(img)
    define_coor(img)

