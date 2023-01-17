import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from numpy import any
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# for debug in IDE and live rotation in 3dims
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class ImageAnnotations3D():
    """Class for displaying images in ax3d"""
    def __init__(self, xyz, imgs, ax3d, ax2d, flag=None):
        """
        :param xyz: coordinates for imgs, in this case = R, G, B values
        :param imgs: images to display
        :param ax3d: main 3d ax
        :param ax2d: dummy ax to projections
        :param flag: flag to off/on images display
        """
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        if flag == 'images':
            for s, im in zip(self.xyz, self.imgs):
                x, y = self.proj(s)
                self.annot.append(self.image(im, [x, y], ax2d))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event", self.update)

        self.funcmap = {"button_press_event": self.ax3d._button_press,
                        "motion_notify_event": self.ax3d._on_move,
                        "button_release_event": self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                    for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1,
            calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self, arr, xy, ax):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=1.5)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy,
                                      xycoords='data', boxcoords="offset points", pad=0.01)
        self.ax2d.add_artist(ab)
        return ab

    def update(self, event):
        if any(self.ax3d.get_w_lims() != self.lim) or \
                any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s, ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)


class VisExtract():

    """Class to encapsulate images reading, getting statistical estimator like mode, median or mean
    and etc dumb stuff"""

    def filer(self):
        """Uses path to data file and num of images to display,
        fills xs, ys, zs with RGB values,
        marks clear, dirty and ships pics with
        green, red and blue colors"""
        filelist = np.random.choice(os.listdir(f'{self.file_name}'), size=self.num)
        for file in filelist:
            img = cv2.imread(f'{self.file_name}/{file}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)

            if file[0] == '1':
                if self.aver == 'mean':
                    self.xs.append(img[:, :, 0].mean())
                    self.ys.append(img[:, :, 1].mean())
                    self.zs.append(img[:, :, 2].mean())
                else:
                    modes = []
                    for i in range(0, 3):
                        vals, counts = np.unique(img[:, :, i], return_counts=True)
                        modes.append(vals[np.argmax(counts)])
                    self.xs.append(modes[0])
                    self.ys.append(modes[1])
                    self.zs.append(modes[2])

                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (255, 0, 0), thickness=35,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.y.append(int(file[0]))

            if file[0] == '0':
                if self.aver == 'mean':
                    self.xs.append(img[:, :, 0].mean())
                    self.ys.append(img[:, :, 1].mean())
                    self.zs.append(img[:, :, 2].mean())
                else:
                    modes = []
                    for i in range(0, 3):
                        vals, counts = np.unique(img[:, :, i], return_counts=True)
                        modes.append(vals[np.argmax(counts)])
                    self.xs.append(modes[0])
                    self.ys.append(modes[1])
                    self.zs.append(modes[2])

                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (0, 255, 0), thickness=35,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.y.append(int(file[0]))

            if self.ships is True:
                if file[0] == '2':
                    if self.aver == 'mean':
                        self.xs.append(img[:, :, 0].mean())
                        self.ys.append(img[:, :, 1].mean())
                        self.zs.append(img[:, :, 2].mean())
                    else:
                        modes = []
                        for i in range(0, 3):
                            vals, counts = np.unique(img[:, :, i], return_counts=True)
                            modes.append(vals[np.argmax(counts)])
                        self.xs.append(modes[0])
                        self.ys.append(modes[1])
                        self.zs.append(modes[2])

                    cv2.drawMarker(img, (img.shape[0] // 3, img.shape[1] // 2), (0, 255, 255), thickness=35,
                                   markerType=cv2.MARKER_CROSS)
                    img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                    self.imgs.append(img)
                    self.y.append(int(file[0]))

        self.xs, self.ys, self.zs = np.array(self.xs),  np.array(self.ys), np.array(self.zs)

        self.y = np.array(self.y).reshape(-1, 1)
        self.data = np.c_[self.xs, self.ys, self.zs]

    def __init__(self, file_name, num, action='show', ships=False, aver='mean'):
        """
        :param file_name (str): file name (relative path) of file with data
        :param num (int): number of images to display
        :param action(str, optional): what to do: show 3d-graph or save it in WD: save or show
        :param ships(bool, optional): display or not ships images
        :param aver(str, optional): statistical estimator to be used: mode, median or mean
        """
        np.random.seed(41)
        self.imgs = []
        self.action = action
        self.data = None
        self.ships = ships
        self.aver = aver

        # R, G, B
        self.xs = []
        self.ys = []
        self.zs = []

        # targets
        self.y = []

        self.file_name = file_name
        self.num = num

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=Axes3D.name)

        # fill arrays with predictors and targets, prepare images
        self.filer()
        self.plotter()

    @staticmethod
    def f_xy(a, b, c, d, x, y):
        """Convert hyperplane from canonical form to the explicit form"""
        return - x * a / c - y * b / c - d / c

    def add_hyperplane(self, params):
        """
        Plots data and hyperplane on 3d graph
        :param params (list): where first element - x of eq, second element y of eq
        third element z of eq, last element - bias of hyperplane
        """
        x = np.arange(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 0.5)
        y = np.arange(np.min(self.data[:, 1]), np.max(self.data[:, 1]), 0.5)

        x, y = np.meshgrid(x, y)
        eq = VisExtract.f_xy(params[0], params[1], params[2], params[3], x, y)

        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.plot_surface(x, y, eq, cmap='spring', alpha=0.95)

        ax_3d.set_xlabel('R channel')
        ax_3d.set_ylabel('G channel')
        ax_3d.set_zlabel('B channel')

        ax_3d.scatter(self.xs[self.pos[0]], self.ys[self.pos[0]], self.zs[self.pos[0]], alpha=1, color='r', marker='x',
                      label='dirty')
        ax_3d.scatter(self.xs[self.neg[0]], self.ys[self.neg[0]], self.zs[self.neg[0]], alpha=1, color='g', marker='o',
                      label='clear')
        ax_3d.legend()
        # to change point of view
        # ax_3d.view_init(30, 150)
        plt.show()

    def plotter(self, flag=None):

        self.pos = np.where(self.y == 1)
        self.neg = np.where(self.y == 0)

        if self.ships is True:
            ships = np.where(self.y == 2)
            self.ax.scatter(self.xs[ships[0]], self.ys[ships[0]], self.zs[ships[0]], alpha=0, color='b', marker='*')

        self.ax.scatter(self.xs[self.pos[0]], self.ys[self.pos[0]], self.zs[self.pos[0]], alpha=0, color='r', marker='x')
        self.ax.scatter(self.xs[self.neg[0]], self.ys[self.neg[0]], self.zs[self.neg[0]], alpha=0, color='g', marker='o')

        self.ax2 = self.fig.add_subplot(111, frame_on=False)
        self.ax2.axis("off")
        self.ax2.axis([0, 1, 0, 1])

        ia = ImageAnnotations3D(self.data, self.imgs, self.ax, self.ax2, 'images')

        self.ax.set_xlabel('R channel')
        self.ax.set_ylabel('G channel')
        self.ax.set_zlabel('B channel')

        if self.action == 'show':
            plt.show()

        elif self.action == 'save':
            plt.savefig(f'{np.random.randint(0, 10)}')

    def two_dim_im_graph(self, xy, im, flag=None, alpha=1):
        """
        :param xy (ndarray): coords of imgs, RGB vals
        :param im (list): list with images
        :param flag (str, optional): what class of images necessary to display - clr, drt
        :param alpha (float, optional): level of image transparency (to analyse and debug)
        """
        if flag == 'clr':
            cl = np.where(self.y == 0)

        if flag == 'drt':
            dr = np.where(self.y == 1)

        for i, image in enumerate(im):

            fig = plt.gcf()
            ax = plt.subplot(111)
            arr_hand = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
            imagebox = offsetbox.OffsetImage(arr_hand, filterrad=1.0, dpi_cor=0, zoom=0.0875, alpha=alpha)

            if flag is not None:

                if flag == 'clr':
                    if i in cl[0]:
                        ab = offsetbox.AnnotationBbox(imagebox, [xy[i][0], xy[i][1]], frameon=0)
                        ax.add_artist(ab)

                if flag == 'drt':
                    if i in dr[0]:
                        ab = offsetbox.AnnotationBbox(imagebox, [xy[i][0], xy[i][1]], frameon=0)
                        ax.add_artist(ab)
            else:
                ab = offsetbox.AnnotationBbox(imagebox, [xy[i][0], xy[i][1]], frameon=0)
                ax.add_artist(ab)

        ax.scatter(xy[:, 0], xy[:, 1])


# for debug and 3d rotation
tst = VisExtract('data', len(os.listdir('data')), 'show', ships=True)
(trainData, testData, trainLabels, testLabels) = train_test_split(tst.data, tst.y, test_size=0.25, random_state=9)
model = LogisticRegression(random_state=0, solver='lbfgs').fit(trainData, trainLabels)
tst.add_hyperplane(list((model.coef_[0][0], model.coef_[0][1], model.coef_[0][2], model.intercept_[0])))
