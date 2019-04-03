import numpy as np
import cv2
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2

class TextDetector(object):

    def __init__(self, tr_thresh=0.4, tcl_thresh=0.6):
        self.tr_thresh = tr_thresh
        self.tcl_thresh = tcl_thresh

    def find_innerpoint(self, cont):
        """
        generate an inner point of input polygon using mean of x coordinate by:
        1. calculate mean of x coordinate(xmean)
        2. calculate maximum and minimum of y coordinate(ymax, ymin)
        3. iterate for each y in range (ymin, ymax), find first segment in the polygon
        4. calculate means of segment
        :param cont: input polygon
        :return:
        """

        xmean = cont[:, 0, 0].mean()
        ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
        found = False
        found_y = []
        #
        for i in np.arange(ymin - 1, ymax + 1, 0.5):
            # if in_poly > 0, (xmean, i) is in `cont`
            in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
            if in_poly > 0:
                found = True
                found_y.append(i)
            # first segment found
            if in_poly < 0 and found:
                break

        if len(found_y) > 0:
            return (xmean, np.array(found_y).mean())

        # if cannot find using above method, try each point's neighbor
        else:
            for p in range(len(cont)):
                point = cont[p, 0]
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        test_pt = point + [i, j]
                        if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
                            return test_pt

    def centerlize(self, x, y, tangent_cos, tangent_sin, mask, stride=1):
        """
        centralizing (x, y) using tangent line and normal line.
        :return:
        """

        H, W = mask.shape

        # calculate normal sin and cos
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        # find upward
        _x, _y = x, y
        while mask[int(_y), int(_x)]:
            _x = _x + normal_cos * stride
            _y = _y + normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end1 = np.array([_x, _y])

        # find downward
        _x, _y = x, y
        while mask[int(_y), int(_x)]:
            _x = _x - normal_cos * stride
            _y = _y - normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end2 = np.array([_x, _y])

        # centralizing
        center = (end1 + end2) / 2

        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_mask, init_xy, direct=1):
        """
        Iteratively find center line in tcl mask using initial point (x, y)
        :param pred_sin: predict sin map
        :param pred_cos: predict cos map
        :param tcl_mask: predict tcl mask
        :param init_xy: initial (x, y)
        :param direct: direction [-1|1]
        :return:
        """

        H, W = pred_sin.shape
        x_init, y_init = init_xy

        sin = pred_sin[int(y_init), int(x_init)]
        cos = pred_cos[int(y_init), int(x_init)]
        radii = pred_radii[int(y_init), int(x_init)]

        x_shift, y_shift = self.centerlize(x_init, y_init, cos, sin, tcl_mask)
        result = []
        max_attempt = 200
        attempt = 0

        while tcl_mask[int(y_shift), int(x_shift)]:

            attempt += 1

            result.append(np.array([x_shift, y_shift, radii]))
            x, y = x_shift, y_shift

            sin = pred_sin[int(y), int(x)]
            cos = pred_cos[int(y), int(x)]

            x_c, y_c = self.centerlize(x, y, cos, sin, tcl_mask)

            sin_c = pred_sin[int(y_c), int(x_c)]
            cos_c = pred_cos[int(y_c), int(x_c)]
            radii = pred_radii[int(y_c), int(x_c)]

            # shift stride
            for shrink in [1/2., 1/4., 1/8., 1/16., 1/32.]:
                t = shrink * radii   # stride = +/- 0.5 * [sin|cos](theta), if new point is outside, shrink it until shrink < 0.1, hit ends
                x_shift_pos = x_c + cos_c * t * direct  # positive direction
                y_shift_pos = y_c + sin_c * t * direct  # positive direction
                x_shift_neg = x_c - cos_c * t * direct  # negative direction
                y_shift_neg = y_c - sin_c * t * direct  # negative direction

                # if first point, select positive direction shift
                if len(result) == 1:
                    x_shift, y_shift = x_shift_pos, y_shift_pos
                else:
                    # else select point further with second last point
                    dist_pos = norm2(result[-2][:2] - (x_shift_pos, y_shift_pos))
                    dist_neg = norm2(result[-2][:2] - (x_shift_neg, y_shift_neg))
                    if dist_pos > dist_neg:
                        x_shift, y_shift = x_shift_pos, y_shift_pos
                    else:
                        x_shift, y_shift = x_shift_neg, y_shift_neg
                # if out of bounds, skip
                if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
                    continue
                # found an inside point
                if tcl_mask[int(y_shift), int(x_shift)]:
                    break
            # if out of bounds, break
            if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
                break
            if attempt > max_attempt:
                break
        return result

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Find TCL's center points and radii of each point
        :param tcl_pred: output tcl mask, (512, 512)
        :param sin_pred: output sin map, (512, 512)
        :param cos_pred: output cos map, (512, 512)
        :param radii_pred: output radii map, (512, 512)
        :return: (list), tcl array: (n, 3), 3 denotes (x, y, radii)
        """
        all_tcls = []

        # find disjoint regions
        mask = fill_hole(tcl_pred)
        _, conts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in conts:

            # find an inner point of polygon
            init = self.find_innerpoint(cont)

            if init is None:
                continue

            x_init, y_init = init

            # find left tcl
            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init), direct=1)
            tcl_left = np.array(tcl_left)
            # find right tcl
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, tcl_pred, (x_init, y_init), direct=-1)
            tcl_right = np.array(tcl_right)
            # concat
            tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
            all_tcls.append(tcl)

        return all_tcls

    def detect(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Input: FCN output, Output: text detection after post-processing

        :param tr_pred: (tensor), text region prediction, (2, H, W)
        :param tcl_pred: (tensor), text center line prediction, (2, H, W)
        :param sin_pred: (tensor), sin prediction, (H, W)
        :param cos_pred: (tensor), cos line prediction, (H, W)
        :param radii_pred: (tensor), radii prediction, (H, W)

        :return:
            (list), tcl array: (n, 3), 3 denotes (x, y, radii)
        """

        # thresholding
        tr_pred_mask = tr_pred[1] > self.tr_thresh
        tcl_pred_mask = tcl_pred[1] > self.tcl_thresh

        # multiply TR and TCL
        tcl = tcl_pred_mask * tr_pred_mask

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        # find tcl in each predicted mask
        detect_result = self.build_tcl(tcl, sin_pred, cos_pred, radii_pred)

        return detect_result