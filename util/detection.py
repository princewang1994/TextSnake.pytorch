import numpy as np
import cv2
from util.misc import fill_hole, regularize_sin_cos

class TextDetector(object):

    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh

    def find_innerpoint(self, cont):

        xmean = cont[:, 0, 0].mean()
        ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
        found = False
        found_y = []
        for i in np.arange(ymin - 1, ymax + 1, 0.5):
            in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
            if in_poly > 0:
                found = True
                found_y.append(i)
            if in_poly < 0 and found:
                break
        if len(found_y) > 0:
            return (xmean, np.array(found_y).mean())
        else: # if cannot find use above method, try each point's neighbor
            for p in range(len(cont)):
                point = cont[p, 0]
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        test_pt = point + [i, j]
                        if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
                            return test_pt

    def centerlize(self, x, y, tangent_cos, tangent_sin, mask, stride=1):
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        _x, _y = x, y
        while mask[int(_y), int(_x)]:
            _x = _x + normal_cos * stride
            _y = _y + normal_sin * stride
        end1 = np.array([_x, _y])

        _x, _y = x, y
        while mask[int(_y), int(_x)]:
            _x = _x - normal_cos * stride
            _y = _y - normal_sin * stride
        end2 = np.array([_x, _y])
        center = (end1 + end2) / 2

        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_mask, init_xy, direct=1, max_pts=50):
        """
        :param pred_sin:
        :param pred_cos:
        :param tcl_mask:
        :param init_xy:
        :param t:
        :param direct:
        :return:
        """
        x_init, y_init = init_xy

        sin = pred_sin[int(y_init), int(x_init)]
        cos = pred_cos[int(y_init), int(x_init)]
        radii = pred_radii[int(y_init), int(x_init)]

        x_shift, y_shift = self.centerlize(x_init, y_init, cos, sin, tcl_mask)
        result = []

        while tcl_mask[int(y_shift), int(x_shift)] and len(result) < max_pts:
            result.append([x_shift, y_shift, radii])
            x, y = x_shift, y_shift

            sin = pred_sin[int(y), int(x)]
            cos = pred_cos[int(y), int(x)]

            x_c, y_c = self.centerlize(x, y, cos, sin, tcl_mask)

            sin_c = pred_sin[int(y_c), int(x_c)]
            cos_c = pred_cos[int(y_c), int(x_c)]
            radii = pred_radii[int(y_c), int(x_c)]

            # shift
            t = 0.5 * radii
            x_shift = x_c + cos_c * t * direct
            y_shift = y_c + sin_c * t * direct
            # print(x_shift, y_shift)

        return result

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Find TCL's center points and radii of each point
        :param tcl_pred: output tcl mask, (512, 512)
        :param sin_pred: output sin map, (512, 512)
        :param cos_pred: output cos map, (512, 512)
        :param radii_pred: output radii map, (512, 512)
        :return: (list), tcl array: (n, 3) 3 denote (x, y, radii)
        """
        all_tcls = []

        # find disjoint regions
        mask = fill_hole(tcl_pred)
        _, conts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in conts:
            if cv2.contourArea(cont) < 20:
                continue
            init = self.find_innerpoint(cont)

            if init is None:
                continue

            x_init, y_init = init

            # find left tcl
            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init))
            tcl_left = np.array(tcl_left)
            # find right tcl
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init), direct=-1)
            tcl_right = np.array(tcl_right)
            # concat
            tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
            all_tcls.append(tcl)

        return all_tcls

    def detect(self, output):

        tr_pred = output[0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[2:4].softmax(dim=0).data.cpu().numpy()

        # multiply TR and TCL
        tcl = tcl_pred * tr_pred

        # thresholding
        tcl_pred_mask = tcl[1] > self.conf_thresh

        sin_pred = output[4].data.cpu().numpy()
        cos_pred = output[5].data.cpu().numpy()
        radii_pred = output[6].data.cpu().numpy()

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        # find tcl in each predicted mask
        tcl_result = []
        tcl = self.build_tcl(tcl_pred_mask, sin_pred, cos_pred, radii_pred)
        tcl_result.append(tcl)

        return tcl_result