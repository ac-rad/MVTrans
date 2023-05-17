# Copyright 2019 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.lib import datapoint
from src.lib.net.post_processing import obb_outputs, depth_outputs, segmentation_outputs


def extract_left_numpy_img(anaglyph, mode = None):
  if mode == 'simnet':
    anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
    anaglyph_np = anaglyph_np.transpose((1, 2, 0))
    left_img = anaglyph_np[..., 0:3] * 255.0
  elif mode == 'multiview':
    anaglyph_np = np.ascontiguousarray(anaglyph[0].cpu().numpy())
    anaglyph_np = anaglyph_np.transpose((1, 2, 0))
    left_img = anaglyph_np[..., 0:3] * 255.0
  return left_img


def extract_right_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 3:6] * 255.0
  return left_img


def create_anaglyph(stereo_dp):
  height, width, _ = stereo_dp.left_color.shape
  image = np.zeros([height, width, 6], dtype=np.uint8)
  cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)
  cv2.normalize(stereo_dp.right_color, stereo_dp.right_color, 0, 255, cv2.NORM_MINMAX)
  image[..., 0:3] = stereo_dp.left_color
  image[..., 3:6] = stereo_dp.right_color
  image = image * 1. / 255.0
  image = image.transpose((2, 0, 1)) # 3xHxW
  return torch.from_numpy(np.ascontiguousarray(image)).float()

def CombineMultiview(stereo_dps):
  height, width, _ = stereo_dps[0].left_color.shape
  image = np.zeros([height, width, 3*len(stereo_dps)], dtype=np.uint8)
  images_combined = []
  for index, stereo_dp in enumerate(stereo_dps):
    cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)
    image = stereo_dp.left_color.transpose((2,0,1))
    image = image * 1. / 255.0
    images_combined.append(image)
  
  return torch.from_numpy(np.ascontiguousarray(images_combined)).float()

class Dataset(Dataset):

  def __init__(self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None):
    super().__init__()

    if datapoint_dataset is None:
      datapoint_dataset = datapoint.make_dataset(dataset_uri)
    self.datapoint_handles = datapoint_dataset.list()
    print(len(self.datapoint_handles))
    # No need to shuffle, already shufled based on random uids
    self.hparams = hparams

    if preprocess_image_func is None:
      self.preprocces_image_func = create_anaglyph
    else:
      assert False
      self.preprocces_image_func = preprocess_image_func

  def __len__(self):
    return len(self.datapoint_handles)

  def getMultiviewSample(self, idx):

    try:
      dp_list = [dp.read() for dp in self.datapoint_handles[idx]]
    except:
      dp_list = [dp for dp in self.datapoint_handles[idx]]

    # Get anaglyph
    stereo_list = [dp.stereo for dp in dp_list]
    anaglyph = CombineMultiview(stereo_list)

    # Get segmentation
    segmentation_target = segmentation_outputs.SegmentationOutput(dp_list[0].segmentation, self.hparams)
    segmentation_target.convert_to_torch_from_numpy()

    scene_name = [dp.uid for dp in dp_list]
    # Check for nans, infs and large depth replace
    if np.isnan(dp_list[0].depth).any():
      depth_mask_nan = np.isnan(dp_list[0].depth)
      dp_list[0].depth[depth_mask_nan] = 3.0

    if np.isinf(dp_list[0].depth).any():
      depth_mask_inf = np.isinf(dp_list[0].depth)
      dp_list[0].depth[depth_mask_inf] = 3.0

    if (dp_list[0].depth > 3).any():
      dp_list[0].depth[dp_list[0].depth > 3] = 3.0

    # Check for nans in covariance
    for pose in dp_list[0].object_poses:
      if np.isnan(pose.cov_matrices).any():
        covariance_mask_nan = np.isnan(pose.cov_matrices)
        pose.cov_matrices[covariance_mask_nan] = 0.0001
      if np.isinf(pose.cov_matrices).any():
        covariance_mask_nan = np.isnan(pose.cov_matrices)
        pose.cov_matrices[covariance_mask_nan] = 0.0001
      if np.isnan(pose.vertex_target).any():
        mask = np.isnan(pose.vertex_target)
        pose.vertex_target[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.vertex_target[~mask])
      if np.isinf(pose.vertex_target).any():
        mask = np.isinf(pose.vertex_target)
        pose.vertex_target[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.vertex_target[~mask])
      if np.isnan(pose.z_centroid).any():
        mask = np.isnan(pose.z_centroid)
        pose.z_centroid[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.z_centroid[~mask])
      if np.isinf(pose.z_centroid).any():
        mask = np.isinf(pose.z_centroid)
        pose.z_centroid[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.z_centroid[~mask])
      if np.isnan(pose.heat_map).any():
        mask = np.isnan(pose.heat_map)
        pose.heat_map[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.heat_map[~mask])
      if np.isinf(pose.heat_map).any():
        mask = np.isinf(pose.heat_map)
        pose.heat_map[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pose.heat_map[~mask])

    depth_target = depth_outputs.DepthOutput(dp_list[0].depth, self.hparams)
    depth_target.convert_to_torch_from_numpy()
    pose_target = None
    for pose_dp in dp_list[0].object_poses:
      pose_target = obb_outputs.OBBOutput(
          pose_dp.heat_map, pose_dp.vertex_target, pose_dp.z_centroid, pose_dp.cov_matrices,
          self.hparams
      )
      pose_target.convert_to_torch_from_numpy()

    box_target = None
    kp_target = None

    # Final check to make sure there are no bad depths
    assert not np.isnan(dp_list[0].depth).any(), 'Depth should not have nan!!'
    assert not isnan(depth_target), 'Depth should not have nan!!'
    
    
    # Get the camera pose
    camera_poses = np.array([np.array(dp.camera_params['camera_extrinsic']) for dp in dp_list])
    camera_poses = torch.from_numpy(np.ascontiguousarray(camera_poses)).float()

    # Get camera intrinsics
    camera_intrinsic = np.array(dp_list[0].camera_params['camera_intrinsic']).reshape((1,3,3))
    camera_intrinsic = torch.from_numpy(np.ascontiguousarray(camera_intrinsic)).float()
    
    assert camera_poses.shape[0] == anaglyph.shape[0], f'Number of camera poses {camera_poses.shape} does not match number of images {anaglyph.shape}'
    return anaglyph, camera_poses, camera_intrinsic, segmentation_target, depth_target, pose_target, box_target, kp_target, dp_list[0].detections, scene_name
  
  def getStereoSample(self,idx):
    
    try:
      dp = self.datapoint_handles[idx].read()
    except:
      dp = self.datapoint_handles[idx]

    anaglyph = self.preprocces_image_func(dp.stereo)

    segmentation_target = segmentation_outputs.SegmentationOutput(dp.segmentation, self.hparams)
    segmentation_target.convert_to_torch_from_numpy()
    scene_name = dp.uid
    
    # Check for nans, infs and large depth replace
    if np.isnan(dp.depth).any():
      depth_mask_nan = np.isnan(dp.depth)
      dp.depth[depth_mask_nan] = 3.0

    if np.isinf(dp.depth).any():
      depth_mask_inf = np.isinf(dp.depth)
      dp.depth[depth_mask_inf] = 3.0

    if (dp.depth > 3).any():
      dp.depth[dp.depth > 3] = 3.0

    # Check for nans in covariance
    for pose in dp.object_poses:
      if np.isnan(pose.cov_matrices).any():
        covariance_mask_nan = np.isnan(pose.cov_matrices)
        pose.cov_matrices[covariance_mask_nan] = 0.0001

    depth_target = depth_outputs.DepthOutput(dp.depth, self.hparams)
    depth_target.convert_to_torch_from_numpy()
    pose_target = None
    for pose_dp in dp.object_poses:
      pose_target = obb_outputs.OBBOutput(
          pose_dp.heat_map, pose_dp.vertex_target, pose_dp.z_centroid, pose_dp.cov_matrices,
          self.hparams
      )
      pose_target.convert_to_torch_from_numpy()

    box_target = None
    kp_target = None

    # Final check to make sure there are no bad depths
    assert not np.isnan(dp.depth).any(), 'Depth should not have nan!!'
    assert not isnan(depth_target), 'Depth should not have nan!!'
    return anaglyph, segmentation_target, depth_target, pose_target, box_target, kp_target, dp.detections, scene_name
  def __getitem__(self, idx):
    if self.hparams.network_type != 'multiview':
      return self.getStereoSample(idx)
    else:
      return self.getMultiviewSample(idx)

def plot(data,index):
  from matplotlib import pyplot as plt
  plt.imshow((((data-np.amin(data))/(np.amax(data) - np.amin(data)))*256).astype(np.uint8), interpolation='nearest')
  plt.savefig(f'/h/helen/transparent-perception/tests/{index}_depth.png')
  plt.close()

def isnan(x):
  return x != x
