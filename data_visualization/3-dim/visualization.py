import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from random import choice
from numpy import any
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


class ImageAnnotations3D():
    def __init__(self, xyz, imgs, ax3d, ax2d):
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
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


class Visualization():

    def filer(self):
        cnt = self.num
        filelist = os.listdir(f'{self.file_name}')
        while len(filelist) != 0:
            file = choice(filelist)
            img = cv2.imread(f'{self.file_name}/{file}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_CUBIC)

            if file[0] == '1':
                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (255, 0, 0), thickness=50,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.xs.append(img[:, :, 0].mean())
                self.ys.append(img[:, :, 1].mean())
                self.zs.append(img[:, :, 2].mean())

            else:
                cv2.drawMarker(img, (img.shape[0] // 2, img.shape[1] // 2), (0, 255, 0), thickness=50,
                               markerType=cv2.MARKER_DIAMOND)
                img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
                self.imgs.append(img)
                self.xs.append(img[:, :, 0].mean())
                self.ys.append(img[:, :, 1].mean())
                self.zs.append(img[:, :, 2].mean())
            cnt -= 1
            # os.remove(f'{file_name}/{file}')
            filelist.remove(file)
            if cnt == 0:
                break

    def __init__(self, file_name, num):

        self.imgs = []

        # R, G, B
        self.xs = []
        self.ys = []
        self.zs = []

        self.file_name = file_name
        self.num = num

        self.plotter()


    def plotter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        self.filer()

        ax.scatter(self.xs, self.ys, self.zs, alpha=0)

        ax2 = fig.add_subplot(111, frame_on=False)
        ax2.axis("off")
        ax2.axis([0, 1, 0, 1])

        ia = ImageAnnotations3D(np.c_[self.xs, self.ys, self.zs], self.imgs, ax, ax2)

        ax.set_xlabel('R channel')
        ax.set_ylabel('G channel')
        ax.set_zlabel('B channel')

        # ax.view_init(30, 180)

        # plt.show()
        plt.savefig(f'{np.random.randint(0, 10)}')
        plt.clf()


test = Visualization('data', 10)
