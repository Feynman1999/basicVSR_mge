import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F
np.set_printoptions(threshold=np.inf) # 加上这一句

def get_xy_ctr_np2(score_size):
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = np.concatenate((y_list, x_list), 1)
    xy_ctr = mge.tensor(xy_list.astype(np.float32))
    return xy_ctr

def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = score_offset + np.concatenate((y_list, x_list), 1) * total_stride
    xy_ctr = mge.tensor(xy_list.astype(np.float32), requires_grad=False)
    return xy_ctr

def get_cls_reg_ctr_targets(points, gt_bboxes, bbox_scale = 0.25):
    """
        Compute regression, classification targets for points in multiple images.
        Args:
            points (Tensor): (1, 2, 19, 19).
            gt_bboxes (Tensor): Ground truth bboxes of each image, (B,4), in [tl_x, tl_y, br_x, br_y] format.
        Returns:
            cls_labels (Tensor): Labels. (B, 1, 19, 19)   0 or 1, 0 means background, 1 means in the box.
            bbox_targets (Tensor): BBox targets. (B, 4, 19, 19)  only consider the foreground, for the background should set loss as 0!
            centerness_targets (Tensor): (B, 1, 19, 19)  only consider the foreground, for the background should set loss as 0!
    """
    gt_bboxes = F.expand_dims(gt_bboxes, axis=-1)
    gt_bboxes = F.expand_dims(gt_bboxes, axis=-1)  # (B,4,1,1)
    # cls_labels
    # 计算四个值以确定是否在内部，由于template比较大，于是缩小bbox为之前的1/2
    gap = (gt_bboxes[:, 2, ...] - gt_bboxes[:, 0, ...]) * (1-bbox_scale) / 2
    up_bound = points[:, 0, ...] > gt_bboxes[:, 0, ...] + gap
    left_bound = points[:, 1, ...] > gt_bboxes[:, 1, ...] + gap
    down_bound = points[:, 0, ...] < gt_bboxes[:, 2, ...] - gap
    right_bound = points[:, 1, ...] < gt_bboxes[:, 3, ...] - gap
    cls_labels = up_bound * left_bound * down_bound * right_bound
    cls_labels = F.expand_dims(cls_labels, axis=1)  # (B,1,19,19)

    # bbox_targets
    # 对于points中的每个坐标，计算偏离情况（这里每个坐标都会计算，所以会有负数）
    up_left = points - gt_bboxes[:, 0:2, ...]  # (B, 2, 19, 19)
    bottom_right = gt_bboxes[:, 2:4, ...] - points
    bbox_targets = F.concat([up_left, bottom_right], axis = 1)  # (B, 4, 19, 19)

    # centerness_targets
    up_bottom = F.minimum(up_left[:, 0, ...], bottom_right[:, 0, ...]) / F.maximum(up_left[:, 0, ...], bottom_right[:, 0, ...])
    left_right = F.minimum(up_left[:, 1, ...], bottom_right[:, 1, ...]) / F.maximum(up_left[:, 1, ...], bottom_right[:, 1, ...])
    centerness_targets = F.sqrt(F.abs(up_bottom * left_right))
    return cls_labels, bbox_targets, centerness_targets

a = get_xy_ctr_np2(5)
print(F.expand_dims(a, axis=-1).shape)
# gt_bboxes = mge.tensor(np.array([[0,0,255,255], [100,100,355,355]]).astype(np.float32))

# cls_labels, bbox_targets, centerness_targets = get_cls_reg_ctr_targets(a, gt_bboxes)
# print(cls_labels.shape)
# print(cls_labels)
# print(bbox_targets.shape)
# print(bbox_targets.numpy())
# print(centerness_targets.shape)
# print(centerness_targets)


# class BCELoss(M.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         pass

#     def forward(self, pred, target, weight=None):
#         losses = -1.0 * (target * F.log(pred) + (1.0 - target) * F.log(1 - pred))
#         if weight is not None:
#             return (losses * weight).sum()
#         else:
#             return losses.sum()

# bce = BCELoss()

# for i in range(1, 10):
#     for j in range(1,10):
#         print(j,i)
#         print(bce(j/10, i/10))