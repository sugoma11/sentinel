import numpy as np
import cv2
import sys
from keras.models import model_from_json
from os import name

if name == 'nt':
    sys.path.append("..\\pic_features\\modules")
elif name == 'posix':
    sys.path.append("../pic_features/modules")

from get_coords import define_coor

# in this directory must be situated model (optionally with its weights)

# read model settings
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
# load weights into new model
cnn.load_weights("model2.h5")

# init image splitting (HEIGHT_DIVxWIDTH_DIV pieces)
HEIGHT_DIV = 23
WIDTH_DIV = 23


def crop_port(image, p1, p2, p3, p4):
    """
    :param image: ndarray;
    :param p1, p2, p3, p4: int: edges of roi
    :return:
    res: ndarray; Cropped and perspectively transformed image
    inv: ndarray; Matrix for inverse transformation
    h, w: int; height and width of transformed image (also necessary for inverse transformation)
    """
    h, w, _ = np.shape(image)
    input_points = np.float32([p1, p2, p3, p4])
    output_points = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    a = cv2.getPerspectiveTransform(input_points, output_points)
    inv = np.linalg.inv(a)
    res = cv2.warpPerspective(image, a, (w, h), flags=cv2.INTER_LINEAR)

    return res, inv, h, w


def split(img_height, img_width, m, n):
    cr_matrix = []
    for y in range(0, img_height, m):
        for x in range(0, img_width, n):
            cr_matrix.append([(x, y), (x + n, y), (x + n, y + m), (x, y + m)])

    return cr_matrix


def fill_zone(img, predict, coordinates):
    """
    :param img: ndarray; Image to draw
    :param predict: int; Prediction for zone
    :param coordinates: list[int]; Coordinates of current split
    :return: ndarray; Drawed img
    """
    coordinates = np.array(coordinates, np.int32)
    coordinates = coordinates.reshape((-1, 1, 2))
    mask = np.zeros(roi.shape, np.uint8)
    cl = {1: (0, 0, 255), 2: (255, 0, 0), 0: (0, 255, 0)}
    for i in range(len(coordinates) - 1):
        cv2.circle(img, coordinates[i][0], 4, (0, 0, 255), -1)
        cv2.line(img,
                 coordinates[i][0],
                 coordinates[i + 1][0],
                 (255, 0, 0), 1)
    cv2.line(img,
             coordinates[0][0],
             coordinates[-1][0],
             (255, 0, 0), 1)
    cv2.fillPoly(mask, [coordinates], color=cl[predict])
    img = cv2.addWeighted(src1=img, alpha=1, src2=mask, beta=0.2, gamma=0)
    return img


# path to image to test
pth = 'test_im/taman_deep_GB_and_ships.png'

source = cv2.imread(pth)
# for some models necessary change RGB2BGR
source_cnn = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_RGB2BGR)
# get edges of roi
roi_points = define_coor(source)
# get roi for prediction and displaying
roi_dt, inv, h, w = crop_port(source_cnn, *roi_points)
roi, _, _, _ = crop_port(source, *roi_points)

# grid matrix of split-coordinates 
coor_matrix = split(roi.shape[0], roi.shape[1], roi.shape[0] // HEIGHT_DIV, roi.shape[1] // WIDTH_DIV)
# list for predictions
preds = []

for coord in coor_matrix:
    tile, _, _, _ = crop_port(roi_dt, *coord)
    # resize for input layer of NN
    tile = cv2.resize(tile, (300, 300)).reshape(1, 300, 300, 3)
    # normalize
    tile = tile / 255
    # get prediction
    pred = cnn.predict(tile)
    pred = np.argmax(pred, axis=-1)[0]
    preds.append(pred)

for l, coord in enumerate(coor_matrix):
    pred = preds[l]
    # draw split zone
    roi = fill_zone(roi, pred, coord)

# inverse transform of drawed image
roi_inv = cv2.warpPerspective(roi, inv, (w, h), flags=cv2.INTER_LINEAR)
# binary mask where roi == 0
_, mask = cv2.threshold(roi_inv, 0, 255, cv2.THRESH_BINARY_INV)
# source image without roi, i.e roi == 0 (black)
source_without_roi = cv2.bitwise_and(source, mask)
# merging drawed image and back
img_recovered = cv2.bitwise_or(source_without_roi, roi_inv)

cv2.imshow('test', img_recovered)
cv2.waitKey(300000)
