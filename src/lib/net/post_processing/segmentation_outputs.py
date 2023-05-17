import numpy as np
import cv2
import torch
from torch.nn import functional as F
from src.lib import color_stuff


class SegmentationOutput:

  def __init__(self, seg_pred, hparams):
    self.seg_pred = seg_pred
    self.is_numpy = False
    self.hparams = hparams

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    if not self.is_numpy:
      self.seg_pred = np.ascontiguousarray(self.seg_pred.detach().cpu().numpy())
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.seg_pred = torch.from_numpy(np.ascontiguousarray(self.seg_pred)).long()
    self.is_numpy = False

  def get_visualization_img(self, left_image):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_segmentation_mask(left_image, self.seg_pred[0])
  
  def get_visualization_img_gt(self, left_image):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_segmentation_mask_gt(left_image, self.seg_pred, num_classes=np.unique(self.seg_pred).shape[0])

  def get_prediction(self):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return self.seg_pred[0]

  def compute_loss(self, seg_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    seg_target_stacked = []
    for seg_target in seg_targets:
      seg_target_stacked.append(seg_target.seg_pred)
    seg_target_batch = torch.stack(seg_target_stacked)
    seg_target_batch=seg_target_batch.cuda()
    seg_loss = F.cross_entropy(self.seg_pred, seg_target_batch, reduction="mean", ignore_index=-100)
    log['segmentation'] = seg_loss
    return self.hparams.loss_seg_mult * seg_loss

  def compute_metrics(self, seg_targets, log, threshold=0.5,  mode = 'Train'):
    smooth = 1
    IoUs = []
    mAPs = []
    num_class=0

    for i, seg_target in enumerate(seg_targets):
      num_class=seg_target.seg_pred.max()+1
      seg_target.convert_to_numpy_from_torch()
      self.convert_to_numpy_from_torch()
      self.seg_pred[i]=self.seg_pred[i].astype(int)
      seg_target.seg_pred=seg_target.seg_pred.astype(int)

      seg_out_pred=np.argmax(self.seg_pred[i], axis=0)

      IoU = [[self.computeIoU(seg_out_pred.astype(int) == k, seg_target.seg_pred.astype(int) == j) 
              for k in range(seg_target.seg_pred.astype(int).max()+1)] 
              for j in range(seg_target.seg_pred.astype(int).max()+1)]

      IoUs.append([IoU[j][j] for j in range(seg_target.seg_pred.astype(int).max()+1)])

      positive = [np.array(IoU[j]) > threshold for j in range(seg_target.seg_pred.astype(int).max()+1)]
      TP = np.array([positive[j][j] for j in range(seg_target.seg_pred.astype(int).max()+1)]).sum()

      num_iou=seg_target.seg_pred.astype(int).max()+1
      mAPs.append(TP/num_iou)

    IoU = np.array(IoUs).sum() / num_class / (i+1)
    mAP = np.array(mAPs).sum() / (i+1)

    log[f'{mode}_seg_IoU'] = IoU
    log[f'{mode}_seg_mAP'] = mAP
    return IoU, mAP

  def computeIoU(self,pred, ground):
    assert (pred.max() <=1) and (ground.max()<=1) ,f'Incorrect behaviour {pred.max()}, {ground.max()}'
    assert (pred.min() <=1) and (ground.min()<=1) ,f'Incorrect behaviour {pred.min()}, {ground.min()}'
    smooth = 1
    intersection = (pred * ground).sum()
    union = (pred + ground).sum()
    return (intersection + smooth)/(union + smooth) 


def draw_segmentation_mask_gt(color_img, seg_mask, num_classes=5):
  assert len(seg_mask.shape) == 2
  seg_mask = seg_mask.astype(np.uint8)
  colors = color_stuff.get_panoptic_colors()
  color_img = color_img_to_gray(color_img)
  for ii, color in zip(range(num_classes), colors):
    colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
    colored_mask[seg_mask == ii, :] = color
    color_img = cv2.addWeighted(
        color_img.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
    )
  return cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


def color_img_to_gray(image):
  gray_scale_img = np.zeros(image.shape)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for i in range(3):
    gray_scale_img[:, :, i] = img
  gray_scale_img[:, :, i] = img
  return gray_scale_img


def draw_segmentation_mask(color_img, seg_mask):
  assert len(seg_mask.shape) == 3
  num_classes = seg_mask.shape[0]
  # Convert to predictions
  seg_mask_predictions = np.argmax(seg_mask, axis=0)
  return draw_segmentation_mask_gt(color_img, seg_mask_predictions, num_classes=num_classes)
