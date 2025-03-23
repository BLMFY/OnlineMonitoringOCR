import copy
import cv2
import torch
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from pyclipper import PyclipperOffset
import math
import operator
from functools import reduce

support_post_process = ['DBPostProcess', 'DetModel','pse_postprocess','DistillationDBPostProcess','FCEPostProcess']

def build_post_process(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    post_process_type = copy_config.pop('type')
    assert post_process_type in support_post_process, f'{post_process_type} is not developed yet!, only {support_post_process} are support now'
    post_process = eval(post_process_type)(**copy_config)
    return post_process

def points2polygon(points):
    """Convert k points to 1 polygon.
    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.
    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return Polygon(point_mat)


def poly_intersection(poly_det, poly_gt, buffer=0.0001):
    """Calculate the intersection area between two polygon.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
    Returns:
        intersection_area (float): The intersection area between two polygons.
    """
    assert isinstance(poly_det, Polygon)
    assert isinstance(poly_gt, Polygon)

    if buffer == 0:
        poly_inter = poly_det & poly_gt
    else:
        poly_inter = poly_det.buffer(buffer) & poly_gt.buffer(buffer)
    return poly_inter.area, poly_inter


def poly_union(poly_det, poly_gt):
    """Calculate the union area between two polygon.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
    Returns:
        union_area (float): The union area between two polygons.
    """
    assert isinstance(poly_det, Polygon)
    assert isinstance(poly_gt, Polygon)

    area_det = poly_det.area
    area_gt = poly_gt.area
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    return area_det + area_gt - area_inters

def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True
    return False

def boundary_iou(src, target):
    """Calculate the IOU between two boundaries.
    Args:
       src (list): Source boundary.
       target (list): Target boundary.
    Returns:
       iou (float): The iou between two boundaries.
    """
    assert valid_boundary(src, False)
    assert valid_boundary(target, False)
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)

    return poly_iou(src_poly, target_poly)

def poly_iou(poly_det, poly_gt):
    """Calculate the IOU between two polygons.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, Polygon)
    assert isinstance(poly_gt, Polygon)
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    if area_union == 0:
        return 0.0
    return area_inters / area_union


def poly_nms(polygons, threshold):
    assert isinstance(polygons, list)

    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))

    keep_poly = []
    index = [i for i in range(polygons.shape[0])]

    while len(index) > 0:
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)
        iou_list = np.zeros((len(index), ))
        for i in range(len(index)):
            B = polygons[index[i]][:-1]
            iou_list[i] = boundary_iou(A, B)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly

def clockwise_sort_points(_point_coordinates):
    """
        以左上角为起点的顺时针排序
        原理就是将笛卡尔坐标转换为极坐标，然后对极坐标的φ进行排序
    Args:
        _point_coordinates:  待排序的点[(x,y),]
    Returns:    排序完成的点
    """
    center_point = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), _point_coordinates),
            [len(_point_coordinates)] * 2))
    return sorted(_point_coordinates, key=lambda coord: (180 + math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center_point))[::-1]))) % 360)


class DistillationDBPostProcess(object):
    def __init__(self, model_name=None,
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        if model_name is None:
            model_name = ["student"]
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(thresh=thresh,
                                          box_thresh=box_thresh,
                                          max_candidates=max_candidates,
                                          unclip_ratio=unclip_ratio,
                                          use_dilation=use_dilation,
                                          score_mode=score_mode)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k].detach().cpu().numpy(), shape_list=shape_list)
        return results


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.6,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        bitmap = (bitmap * 255).astype(np.uint8)
        # structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, structure_element)

        # cv2.imwrite('bin.jpg',bitmap)

        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise NotImplementedError(f'opencv {cv2.__version__} not support')

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]

            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            # score = self.box_score_slow(pred, contour)
            if score < self.box_thresh:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)

            # try:
            #     poly = contours[index]
            #     # cv2.drawContours(debug_mat, poly, -1, (111, 90, 255), -1)
            #
            #     epsilon = 0.001 * cv2.arcLength(poly, True)
            #     approx = cv2.approxPolyDP(poly, epsilon, True)
            #     points = approx.reshape((-1, 2))
            #     if points.shape[0] < 4:
            #         continue
            #     score = self.box_score_fast(pred, points)
            #     if score < self.box_thresh:
            #         continue
            #     poly = self.unclip(points)
            #     if len(poly) == 0 or isinstance(poly[0], list):
            #         continue
            #     poly = poly.reshape(-1, 2)
            #
            #     # box, sside = self.get_mini_boxes(poly)
            #     # if sside < self.min_size + 2:
            #     #     continue
            #     # box = np.array(box)
            #     box=np.array(poly)
            #
            #     box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            #     box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            #     boxes.append(box.astype(np.int16).flatten().tolist())
            #     scores.append(score)
            # except:
            #     print('1')
            #     pass

        return boxes, scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        try:
            rotated_box = cv2.minAreaRect(contour)
        except:
            print(len(contour))
            return None, 0
        box_points = cv2.boxPoints(rotated_box)
        rotated_points = clockwise_sort_points(box_points)
        rotated_points = list(rotated_points)
        return rotated_points, min(rotated_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = np.zeros_like(pred, dtype=np.float32)
        np.putmask(segmentation, pred > self.thresh, pred)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h, )
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch
    
def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


def fourier2poly(fourier_coeff, num_reconstr_points=50):
    """ Inverse Fourier transform
        Args:
            fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1),
                with n and k being candidates number and Fourier degree
                respectively.
            num_reconstr_points (int): Number of reconstructed polygon points.
        Returns:
            Polygons (ndarray): The reconstructed polygons shaped (n, n')
        """

    a = np.zeros((len(fourier_coeff), num_reconstr_points), dtype='complex')
    k = (len(fourier_coeff[0]) - 1) // 2

    a[:, 0:k + 1] = fourier_coeff[:, k:]
    a[:, -k:] = fourier_coeff[:, :k]

    poly_complex = np.fft.ifft(a) * num_reconstr_points
    polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2))
    polygon[:, :, 0] = poly_complex.real
    polygon[:, :, 1] = poly_complex.imag
    return polygon.astype('int32').reshape((len(fourier_coeff), -1))


class FCEPostProcess(object):
    """
    The post process for FCENet.
    """

    def __init__(self,
                 scales,
                 fourier_degree=5,
                 num_reconstr_points=50,
                 decoding_type='fcenet',
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 box_type='poly',
                 **kwargs):

        self.scales = scales
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.box_type = box_type

    def __call__(self, preds, shape_list):
        score_maps = []
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy()
            cls_res = value[:, :4, :, :]
            reg_res = value[:, 4:, :, :]
            score_maps.append([cls_res, reg_res])

        return self.get_boundary(score_maps, shape_list)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.
        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
            with size 2k+1 with k>=4.
            scale_factor(ndarray): The scale factor of size (4,).
        Returns:
            boundaries (list[list[float]]): The scaled boundaries.
        """
        boxes = []
        scores = []
        for b in boundaries:
            sz = len(b)
            valid_boundary(b, True)
            scores.append(b[-1])
            b = (np.array(b[:sz - 1]) *
                 (np.tile(scale_factor[:2], int(
                     (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
            boxes.append(np.array(b).reshape([-1, 2]))

        return np.array(boxes, dtype=np.float32), scores

    def get_boundary(self, score_maps, shape_list):
        assert len(score_maps) == len(self.scales)
        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(score_map,scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)
        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])

        # boxes_batch = [dict(points=boundaries, scores=scores)]
        return boundaries.tolist(),scores

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return self.fcenet_decode(
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            box_type=self.box_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)

    def fcenet_decode(self,
                      preds,
                      fourier_degree,
                      num_reconstr_points,
                      scale,
                      alpha=1.0,
                      beta=2.0,
                      box_type='poly',
                      score_thr=0.3,
                      nms_thr=0.1):
        """Decoding predictions of FCENet to instances.
        Args:
            preds (list(Tensor)): The head output tensors.
            fourier_degree (int): The maximum Fourier transform degree k.
            num_reconstr_points (int): The points number of the polygon
                reconstructed from predicted Fourier coefficients.
            scale (int): The down-sample scale of the prediction.
            alpha (float) : The parameter to calculate final scores. Score_{final}
                    = (Score_{text region} ^ alpha)
                    * (Score_{text center region}^ beta)
            beta (float) : The parameter to calculate final score.
            box_type (str):  Boundary encoding type 'poly' or 'quad'.
            score_thr (float) : The threshold used to filter out the final
                candidates.
            nms_thr (float) :  The threshold of nms.
        Returns:
            boundaries (list[list[float]]): The instance boundary and confidence
                list.
        """
        assert isinstance(preds, list)
        assert len(preds) == 2
        assert box_type in ['poly', 'quad']

        cls_pred = preds[0][0]
        tr_pred = cls_pred[0:2]
        tcl_pred = cls_pred[2:]

        reg_pred = preds[1][0].transpose([1, 2, 0])
        x_pred = reg_pred[:, :, :2 * fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * fourier_degree + 1:]

        score_pred = (tr_pred[1]**alpha) * (tcl_pred[1]**beta)
        tr_pred_mask = (score_pred) > score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(
            tr_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, fourier_degree] = c[:, fourier_degree] + dxy
            c *= scale

            polygons = fourier2poly(c, num_reconstr_points)
            score = score_map[score_mask].reshape(-1, 1)
            polygons = poly_nms(np.hstack((polygons, score)).tolist(), nms_thr)

            boundaries = boundaries + polygons

        boundaries = poly_nms(boundaries, nms_thr)

        if box_type == 'quad':
            new_boundaries = []
            for boundary in boundaries:
                poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
                score = boundary[-1]
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_boundaries.append(points.reshape(-1).tolist() + [score])
                boundaries = new_boundaries

        return boundaries