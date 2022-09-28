import cv2 as cv2
import numpy as np
import os


def crop_port(image, p1, p2, p3, p4):
    h,w,_ = np.shape(image)
    input_points = np.float32([p1,p2,p3,p4])
    output_points = np.float32([(0,0),(w,0),(w,h),(0,h)])
    a = cv2.getPerspectiveTransform(input_points,output_points)
    res = cv2.warpPerspective(image, a, (w,h), flags=cv2.INTER_LINEAR)
    return res


def mouse_click(event, x, y, flags, param):
    global poly, current_polygon, selected_class, img
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(poly[selected_class][current_polygon]) < 4:
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
                    color = (125, 30, 200)
                elif cur_class == 1:
                    color = (14, 88, 0)
                mask = cv2.fillPoly(mask, [points], color)
                img2 = cv2.addWeighted(src1=img2, alpha=1, src2=mask, beta=.2, gamma=0)
    if event == cv2.EVENT_RBUTTONDOWN:
        poly[selected_class][current_polygon].pop()

    cv2.imshow('tst', img2)
    # print(event, x, y, flags, param)


file = os.listdir('raw')[0]


def define_coor(pic):
    global img, poly, cur_class, selected_class, current_polygon
    img = pic
    selected_class = 0
    current_polygon = 0
    poly = [[[]] for i in range(2)]
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
            # for cur_class in range(len(poly)):
            #    for cur_polygon in range(len(poly[cur_class])):
            #        if len(poly[cur_class][cur_polygon]) == 4:
            #            h, w, _ = img.shape
            #            input_pt = np.array(poly[cur_class][cur_polygon])
            #            output_pt = np.array([(0, 0), (w, 0), (w, h), (0, h)])
            #            hom, _ = cv2.findHomography(input_pt, output_pt)
            #            res_img = cv2.warpPerspective(img.copy(), hom, (w,h))
            #            if cur_class == 0:
            #                res_img = cv2.resize(res_img, (380, 300), interpolation = cv2.INTER_AREA)
            #                cv2.imwrite(f"dirty/{time.time()}.jpg", res_img)
            #            elif cur_class == 1:
            #                res_img = cv2.resize(res_img, (380, 300), interpolation=cv2.INTER_AREA)
            #                cv2.imwrite(f"clear/{time.time()}.jpg", res_img)
            #os.remove(f'images/{file}')
            #break
        if key == 49: # 1
            selected_class = 1
            current_polygon = len(poly[selected_class]) - 1
        if key != -1:
            pass
        if key == 27: # ESCape
            break
    cv2.destroyAllWindows()
    return poly[0][0], poly[0][1]

