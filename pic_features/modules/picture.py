import numpy as np
import cv2
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Picture:
    def __init__(self, raw, port_points, out_points):
        """
        :param raw (ndarray): BGR image
        :param port_points (list): list of tuples with xy coords of port-ROI
        :param out_points (list): list of tuples with xy coords of out port-ROI (to analyse difference)
        """
        self.raw = raw

        self.port = Picture.crop_port(raw, *port_points)

        self.out = Picture.crop_port(raw, *out_points)

        self.port_hsv = cv2.cvtColor(self.port, cv2.COLOR_BGR2HSV)

        self.out_hsv = cv2.cvtColor(self.out, cv2.COLOR_BGR2HSV)

        # means of HSV and RGB channels
        self.port_ch = (round(np.mean(self.port[:, :, 2]), 2),
                        round(np.mean(self.port[:, :, 1]), 2),
                        round(np.mean(self.port[:, :, 0]), 2),
                        round(np.mean(self.port_hsv[:, :, 0] * 2), 2),
                        round(np.mean(self.port_hsv[:, :, 1] / 2.55), 2),
                        round(np.mean(self.port_hsv[:, :, 2] / 2.55), 2))

        self.out_ch = (round(np.mean(self.out[:, :, 2]), 2),
                       round(np.mean(self.out[:, :, 1]), 2),
                       round(np.mean(self.out[:, :, 0]), 2),
                       round(np.mean(self.out_hsv[:, :, 0] * 2), 2),
                       round(np.mean(self.out_hsv[:, :, 1] / 2.55), 2),
                       round(np.mean(self.out_hsv[:, :, 2] / 2.55), 2))

        # statistical estiamtors
        self.port_moments = (self.port_ch[3], round(self.port[:, :, 2].var(), 2),
                             round(kurtosis(self.port_hsv[:, :, 0].reshape(1, -1)[0]), 2),
                             round(float(skew(self.port_hsv[:, :, 0].reshape(-1, 1))), 2))

        self.out_moments = (self.out_ch[3], round(self.out[:, :, 2].var(), 2),
                            round(kurtosis(self.out_hsv[:, :, 0].reshape(1, -1)[0]), 2),
                            round(float(skew(self.out_hsv[:, :, 0].reshape(-1, 1))), 2))

        # approximated by mean of split image
        self.port_pix = None

    def create_distr(self, l, j):
        """
        Creating histograms of image channels
        :param l (int): column index for multi-ax figure
        :param j (int): row index for multi-ax figure
        """
        if l != 0:
            port = self.port[:, :, j]
            out_port = self.out[:, :, j]
        else:
            if j == 0:
                port = self.port_hsv[:, :, j] * 2
                out_port = self.out_hsv[:, :, j] * 2
            else:
                port = self.port_hsv[:, :, j] // 2.55
                out_port = self.out_hsv[:, :, j] // 2.55

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

    def plotter(self):
        """
        Calls creating histograms in loop and saving image in temp directory
        """
        global name_list
        global f, ax
        name_list = [["Hue", "Saturation", "Value"], ["B", "G", "R"]]
        f, ax = plt.subplots(2, 3)
        f.set_size_inches(19.2, 10)

        for l in range(2):
            for j in range(3):
                Picture.create_distr(self, l, j)
        plt.savefig(f'temp/distr_graphic')

        for l in range(2):
            for j in range(3):
                ax[l][j].clear()

    def add_text(self):
        """
        Add text with statistical information on image to analysing
        """
        string_one = f'Port: M:{self.port_moments[0]}; D:{self.port_moments[1]}; E:{self.port_moments[2]}; A:{self.port_moments[3]};'
        string_two = f'R:{self.port_ch[0]}; G:{self.port_ch[1]}; B:{self.port_ch[2]}; H:{self.port_ch[3]}; S:{self.port_ch[4]}; V:{self.port_ch[5]}'
        string_thr = f'Out: M:{self.out_moments[0]}; D:{self.out_moments[1]}; E:{self.out_moments[2]}; A:{self.out_moments[3]};'
        string_fourth = f'R:{self.out_ch[0]}; G:{self.out_ch[1]}; B:{self.out_ch[2]}; H:{self.out_ch[3]}; S:{self.out_ch[4]}; V:{self.out_ch[5]}'
        string_fifth = f'Delta: M:{round(abs(self.port_moments[0] - self.out_moments[0]), 2)}; D:{round(abs(self.port_moments[1] - self.out_moments[1]), 2)};' \
                       f' E:{round(abs(self.port_moments[2] - self.out_moments[2]), 2)}; A:{round(abs(self.port_moments[3] - self.out_moments[3]), 2)};'

        str_list = [string_one, string_two, string_thr, string_fourth, string_fifth]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = [10, 75]
        fontScale = 1
        fontColor = (0, 255, 255)
        thickness = 2
        lineType = 2


        for i in range(5):
            cv2.putText(self.raw, str_list[i], bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)
            bottomLeftCornerOfText[1] += 30

    def aver_split(self):
        """
        Creating approximated image by means of splits of raw image.
        Necessary to show how different image in its parts if we use mean like an estimator
        """
        im = cv2.resize(self.port, (500, 500))
        blank = np.zeros_like(im)

        img_height = im.shape[0]
        img_width = im.shape[1]

        y_one = 0

        M = img_height // 4
        N = img_width // 4

        for y in range(0, img_height, M):
            for x in range(0, img_width, N):
                y_one = y + M
                x_one = x + N
                tiles = im[y:y + M, x:x + N]
                blank[y:y + M, x:x + N] = np.array([tiles[:, :, 0].mean(), tiles[:, :, 1].mean(),
                                                    tiles[:, :, 2].mean()])

        blank = cv2.resize(blank, (225, 225))

        self.port_pix = blank

    def bitwise(self):
        """
        Joins matplotlib graphics with image
        """
        graph = cv2.imread(f'temp/distr_graphic.png')
        blank_port, blank_out_port = np.zeros((225, 225, 3), dtype='uint8'), np.zeros((225, 225, 3), dtype='uint8')
        if self.raw.shape[0:2] != (1000, 1920):
            self.raw = cv2.resize(self.raw, (1920, 1000), interpolation=cv2.INTER_LINEAR)

        blank_port[:, :, 0], blank_out_port[:, :, 0] = self.port_ch[2], self.out_ch[2]
        blank_port[:, :, 1], blank_out_port[:, :, 1] = self.port_ch[1], self.out_ch[1]
        blank_port[:, :, 2], blank_out_port[:, :, 2] = self.port_ch[0], self.out_ch[0]

        self.raw[self.raw.shape[0] - blank_port.shape[0]:self.raw.shape[0], self.raw.shape[1] - blank_port.shape[0]:self.raw.shape[1], :] = blank_port
        self.raw[self.raw.shape[0] - 2 * blank_port.shape[0]:self.raw.shape[0] - blank_port.shape[0], self.raw.shape[1] - blank_port.shape[0]:self.raw.shape[1], :] = blank_out_port
        self.raw[self.raw.shape[0] - 3 * blank_port.shape[0]:self.raw.shape[0] - 2 * blank_port.shape[0], self.raw.shape[1] - blank_port.shape[0]:self.raw.shape[1], :] = self.port_pix
        self.raw = np.vstack((self.raw, graph))

    def dot_edges(self, port_points, out_points):
        """
        Draws points on edges of ROIs
        """
        contours = np.array([[[port_points[0]]], [[port_points[1]]], [[port_points[2]]], [[port_points[3]]]], dtype=int)
        contours_outs = np.array([[[out_points[0]]], [[out_points[1]]], [[out_points[2]]], [[out_points[3]]]],
                                 dtype=int)
        cv2.drawContours(self.raw, contours, -1, (0, 255, 0), 7)
        cv2.drawContours(self.raw, contours_outs, -1, (0, 0, 255), 7)

    @staticmethod
    def crop_port(image, p1, p2, p3, p4):
        """
        :param image (ndarray): image in RGB or BGR
        :param p1, p2, p3, p4: edge points of ROI
        :return res (ndarray): cropped and transformed ROI
        """
        h, w, _ = np.shape(image)
        input_points = np.float32([p1, p2, p3, p4])
        output_points = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
        a = cv2.getPerspectiveTransform(input_points, output_points)
        res = cv2.warpPerspective(image, a, (w, h), flags=cv2.INTER_LINEAR)
        return res

