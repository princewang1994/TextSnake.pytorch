import copy
import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
from skimage.draw import polygon as drawpoly
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin

from dataset.data_util import pil_load_img

class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        self.points = []

        # remove point if area is almost unchanged after removing
        ori_area = cv2.contourArea(points)
        for p in range(len(points)):
            index = list(range(len(points)))
            index.remove(p)
            area = cv2.contourArea(points[index])
            if np.abs(ori_area - area) / ori_area > 0.017:
                self.points.append(points[p])
        self.points = np.array(self.points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TotalText(data.Dataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.transform = transform

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        self.polygons = [None] * len(self.image_list)  # polygon cache

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4: # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius, \
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.2, shrink=2):

        # TODO: shrink 1/2 * radius at two line end
        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)


    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # print(image_path)

        if self.polygons[item]:
            polygons = self.polygons[item]
        else:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_mat(annotation_path)

            for i, polygon in enumerate(polygons):
                if polygon.text != '#':
                    polygon.find_bottom_and_sideline()
            self.polygons[item] = polygons

        # print(image_path, annotation_path)
        # Read image data
        image = pil_load_img(image_path)

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover()
                self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map)
        tr_mask, train_mask = self.make_text_region(image, polygons)
        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform
    transform = BaseTransform(
        size=512, mean=0.5, std=0.5
    )
    ds = TotalText(
        data_root='/home/prince/ext_data/dataset/text-detection/total-text',
        ignore_list='/data/prince/project/TextSnake/ignore_list.txt',
        is_training=False,
        transform=transform
    )
    for i in range(len(ds)):
        try:
            image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = ds[i]
        except:
            print(ds.image_list[i])

    # loader = data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    # for image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map in loader:
    #     print()
    #     print(image.size(), train_mask.size(), tr_mask.size(), tcl_mask.size(), radius_map.size(), sin_map.size(), cos_map.size())