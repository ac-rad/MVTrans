import numpy as np
import cv2
import torch

from skimage.feature import peak_local_max

from src.lib import transform, color_stuff, camera
from src.lib.net.post_processing.epnp import optimize_for_9D
from src.lib.net.post_processing import epnp, eval3d, nms
from src.lib.net.post_processing.eval3d import measure_3d_iou, EvalMetrics
from src.lib.net import losses
import copy

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()


class PoseOutput:

  def __init__(self, heatmap, vertex_field, z_centroid_field, hparams):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.z_centroid_field = z_centroid_field
    self.is_numpy = False
    self.hparams = hparams

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
    self.vertex_field = np.ascontiguousarray(self.vertex_field.cpu().numpy())
    self.vertex_field = self.vertex_field.transpose((0, 2, 3, 1))
    self.vertex_field = self.vertex_field / 100.0
    self.z_centroid_field = np.ascontiguousarray(self.z_centroid_field.cpu().numpy())
    self.z_centroid_field = self.z_centroid_field / 100.0 + 1.0
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.vertex_field = self.vertex_field.transpose((2, 0, 1))
    self.vertex_field = 100.0 * self.vertex_field
    self.vertex_field = torch.from_numpy(np.ascontiguousarray(self.vertex_field)).float()
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
    # Normalize z_centroid by 1.
    self.z_centroid_field = 100.0 * (self.z_centroid_field - 1.0)
    self.z_centroid_field = torch.from_numpy(np.ascontiguousarray(self.z_centroid_field)).float()
    self.is_numpy = False

  def get_detections(self):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()

    poses, scores = compute_9D_poses(
        np.copy(self.heatmap[0]), np.copy(self.vertex_field[0]), np.copy(self.z_centroid_field[0])
    )

    detections = []
    for pose, score in zip(poses, scores):
      bbox = epnp.get_2d_bbox_of_9D_box(pose.camera_T_object, pose.scale_matrix)
      detections.append(
          eval3d.Detection(
              camera_T_object=pose.camera_T_object,
              bbox=bbox,
              score=score,
              scale_matrix=pose.scale_matrix
          )
      )

    detections = nms.run(detections)

    return detections

  def get_visualization_img(self, left_img):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_pose_from_outputs(
        self.heatmap[0],
        self.vertex_field[0],
        self.z_centroid_field[0],
        left_img,
        max_detections=100,
    )

  def compute_metrics(self, pose_targets, log):

    td_iou_list=[]
    IoU_list=[]
    mAP_list=[]
    num_sample=0
    ap_values = []
    ADD_list=[]
    ADD_s_list=[]
    AUC_list=[]
    less2cm_list=[]
    AUC_adds_list=[]
    less2cm_adds_list=[]
    
    # 3D bbox eval
    detection_outputs=self.get_detections()

    poses=[]
    bbox_ext=[]
    pose_item=None
    for pose_target in pose_targets:
        gt_detections = pose_target.get_detections()
        # 3D IOU
        true_matches, pred_matches, pred_scores, class_labels, ignore_labels, sorted_detections, overlaps = measure_3d_iou(copy.deepcopy(detection_outputs), copy.deepcopy(gt_detections))
        if len(overlaps) == 0:
            flag=True
            break
        td_iou_list.append(overlaps[0][0])

        # 3D mAP
        td_mAP=EvalMetrics()
        td_mAP.process_sample(true_matches=true_matches, pred_matches=pred_matches, pred_scores=pred_scores)
        ap_values.append(td_mAP.process_dataset()[0])

    def get_obj_pose_and_bbox(heatmap_output, vertex_output, z_centroid_output, cov_matrices, camera_model):
      peaks = self.extract_peaks_from_centroid(np.copy(heatmap_output), max_peaks=np.inf)
      bboxes_ext = self.extract_vertices_from_peaks(np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output))  # Shape: List(np.array([8,2])) --> y,x order
      z_centroids = self.extract_z_centroid_from_peaks(np.copy(peaks), np.copy(z_centroid_output))
      cov_matrices = self.extract_cov_matrices_from_peaks(np.copy(peaks), np.copy(cov_matrices))
      poses = []
      for bbox_ext, z_centroid, cov_matrix, peak in zip(bboxes_ext, z_centroids, cov_matrices, peaks):
          bbox_ext_flipped = bbox_ext[:, ::-1] # Switch from yx to xy
          # Solve for pose up to a scale factor
          error, camera_T_object, scale_matrix = optimize_for_9D(bbox_ext_flipped.T, camera_model, solve_for_transforms=True) 
          abs_camera_T_object, abs_object_scale = epnp.find_absolute_scale(
              -1.0 * z_centroid, camera_T_object, scale_matrix
          )
          poses.append(transform.Pose(camera_T_object=abs_camera_T_object, scale_matrix=abs_object_scale))
      return poses, bboxes_ext
  def compute_loss(self, pose_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    vertex_target = torch.stack([pose_target.vertex_field for pose_target in pose_targets])
    z_centroid_field_target = torch.stack([
        pose_target.z_centroid_field for pose_target in pose_targets
    ])
    heatmap_target = torch.stack([pose_target.heatmap for pose_target in pose_targets])

    # Move to GPU
    heatmap_target = heatmap_target.cuda()
    vertex_target = vertex_target.cuda()
    z_centroid_field_target = z_centroid_field_target.cuda()

    vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
    log['vertex_loss'] = vertex_loss
    z_centroid_loss = _mask_l1_loss(z_centroid_field_target, self.z_centroid_field, heatmap_target)
    log['z_centroid'] = z_centroid_loss

    heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
    log['heatmap'] = heatmap_loss
    return self.hparams.loss_vertex_mult * vertex_loss + self.hparams.loss_heatmap_mult * heatmap_loss + self.hparams.loss_z_centroid_mult * z_centroid_loss


def extract_peaks_from_centroid(
    centroid_heatmap, min_distance=5, min_confidence=0.3, max_peaks=np.inf
):
  peaks = peak_local_max(
      centroid_heatmap,
      min_distance=min_distance,
      threshold_abs=min_confidence,
      num_peaks=max_peaks
  )
  peaks_old = peak_local_max(
      centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence
  )

  return peaks


def extract_vertices_from_peaks(peaks, vertex_fields, c_img, scale_factor=8):
  '''
  peaks: np.array (n_objs, 2)
  vertex_fields: np.array (h, w, 16)
  '''
  assert peaks.shape[1] == 2
  assert vertex_fields.shape[2] == 16
  height, width = c_img.shape[0:2]
  vertex_fields = vertex_fields
  vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (2 * height) - height
  vertex_fields[:, :, 1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 * width) - width
  bboxes = []
  for ii in range(peaks.shape[0]):
    bbox = get_bbox_from_vertex(vertex_fields, peaks[ii, :], scale_factor=scale_factor)
    bboxes.append(bbox)
  return bboxes # Shape: List(np.array([8,2]))


def extract_z_centroid_from_peaks(peaks, z_centroid_output, scale_factor=8):
  assert peaks.shape[1] == 2
  z_centroids = []
  for ii in range(peaks.shape[0]):
    index = np.zeros([2])
    index[0] = int(peaks[ii, 0] / scale_factor)
    index[1] = int(peaks[ii, 1] / scale_factor)
    index = index.astype(np.int)
    z_centroids.append(z_centroid_output[index[0], index[1]])
  return z_centroids


def extract_cov_matrices_from_peaks(peaks, cov_matrices_output, scale_factor=8):
  assert peaks.shape[1] == 2
  cov_matrices = []
  for ii in range(peaks.shape[0]):
    index = np.zeros([2])
    index[0] = int(peaks[ii, 0] / scale_factor)
    index[1] = int(peaks[ii, 1] / scale_factor)
    index = index.astype(np.int)
    cov_mat_values = cov_matrices_output[index[0], index[1], :]
    cov_matrix = np.array([[cov_mat_values[0], cov_mat_values[3], cov_mat_values[4]],
                           [cov_mat_values[3], cov_mat_values[1], cov_mat_values[5]],
                           [cov_mat_values[4], cov_mat_values[5], cov_mat_values[2]]])
    cov_matrices.append(cov_matrix)
  return cov_matrices


def get_bbox_from_vertex(vertex_fields, index, scale_factor=8):
  '''
  index: where the vertex is located; order is y, x
  vertex_fields: (h,w,16)
  '''
  assert index.shape[0] == 2
  index[0] = int(index[0] / scale_factor)
  index[1] = int(index[1] / scale_factor)
  bbox = vertex_fields[index[0], index[1], :]
  bbox = bbox.reshape([8, 2]) # y, x order
  bbox = scale_factor * (index) - bbox
  return bbox


def draw_peaks(centroid_target, peaks):
  centroid_target = np.clip(centroid_target, 0.0, 1.0) * 255.0
  color = (0, 0, 255)
  height, width = centroid_target.shape
  # Make a 3 Channel image.
  c_img = np.zeros([centroid_target.shape[0], centroid_target.shape[1], 3])
  c_img[:, :, 1] = centroid_target
  for ii in range(peaks.shape[0]):
    point = (int(peaks[ii, 1]), int(peaks[ii, 0]))
    c_img = cv2.circle(c_img, point, 8, color, -1)
  return cv2.resize(c_img, (width, height))


def draw_pose_from_outputs(
    heatmap_output, vertex_output, z_centroid_output, c_img, max_detections=np.inf
):
  poses, _ = compute_9D_poses(
      np.copy(heatmap_output),
      np.copy(vertex_output),
      np.copy(z_centroid_output),
      max_detections=max_detections,
  )
  return draw_9dof_cv2_boxes(c_img, poses)


def draw_pose_9D_from_detections(detections, c_img):
  successes = []
  poses = []
  for detection in detections:
    poses.append(
        transform.Pose(
            camera_T_object=detection.camera_T_object, scale_matrix=detection.scale_matrix
        )
    )
    successes.append(detection.success)
  return draw_9dof_cv2_boxes(c_img, poses, successes=successes)


def solve_for_rotation_from_cov_matrix(cov_matrix):
  assert cov_matrix.shape[0] == 3
  assert cov_matrix.shape[1] == 3
  U, D, Vh = np.linalg.svd(cov_matrix, full_matrices=True)
  d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
  if d:
    D[-1] = -D[-1]
    U[:, -1] = -U[:, -1]
  # Rotation from world to points.
  rotation = np.eye(4)
  rotation[0:3, 0:3] = U
  return rotation


def compute_9D_poses(heatmap_output, vertex_output, z_centroid_output, max_detections=np.inf):
  peaks = extract_peaks_from_centroid(np.copy(heatmap_output), max_peaks=max_detections)
  bboxes_ext = extract_vertices_from_peaks(
      np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output)
  )
  z_centroids = extract_z_centroid_from_peaks(np.copy(peaks), np.copy(z_centroid_output))
  poses = []
  scores = []
  for bbox_ext, z_centroid, peak in zip(bboxes_ext, z_centroids, peaks):
    bbox_ext_flipped = bbox_ext[:, ::-1]
    # Solve for pose up to a scale factor
    error, camera_T_object, scale_matrix = optimize_for_9D(
        bbox_ext_flipped.T, solve_for_transforms=True
    )
    # Assign correct depth factor
    abs_camera_T_object, abs_scale_matrix = epnp.find_absolute_scale(
        z_centroid, camera_T_object, scale_matrix
    )
    poses.append(transform.Pose(camera_T_object=abs_camera_T_object, scale_matrix=abs_scale_matrix))
    scores.append(heatmap_output[peak[0], peak[1]])
  return poses, scores


def draw_9dof_cv2_boxes(c_img, poses, camera_model=None, successes=None):
  boxes = []
  for pose in poses:
    # Compute the bounds of the boxes current size and location
    unit_box_homopoints = camera.convert_points_to_homopoints(epnp._WORLD_T_POINTS.T)
    morphed_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
    if camera_model == None:
      camera_model = camera.HSRCamera()
    else:
      camera_model = camera_model
    morphed_pixels = camera.convert_homopixels_to_pixels(
        camera_model.K_matrix @ morphed_homopoints
    ).T
    boxes.append(morphed_pixels[:, ::-1])
  return draw_9dof_box(c_img, boxes, successes=successes)


def draw_9dof_box(c_img, boxes, successes=None):
  if len(boxes) == 0:
    return c_img
  if successes is None:
    colors = color_stuff.get_colors(len(boxes))
  else:
    colors = []
    for success in successes:
      #TODO(michael.laskey): Move to Enum Structure
      if success == 1:
        colors.append(np.array([0, 255, 0]).astype(np.uint8))
      elif success == -1:
        colors.append(np.array([255, 255, 0]).astype(np.uint8))
      elif success == -2:
        colors.append(np.array([0, 0, 255]).astype(np.uint8))
      else:
        colors.append(np.array([255, 0, 0]).astype(np.uint8))
  c_img = cv2.cvtColor(np.array(c_img), cv2.COLOR_BGR2RGB)
  for vertices, color in zip(boxes, colors):
    vertices = vertices.astype(np.int)
    points = []
    vertex_colors = (255, 0, 0)
    line_color = (int(color[0]), int(color[1]), int(color[2]))
    circle_colors = color_stuff.get_colors(8)
    for i, circle_color in zip(range(vertices.shape[0]), circle_colors):
      color = vertex_colors
      point = (int(vertices[i, 1]), int(vertices[i, 0]))
      c_img = cv2.circle(c_img, point, 1, (0, 255, 0), -1)
      points.append(point)

    # Draw the lines
    thickness = 1

    c_img = cv2.line(c_img, points[0], points[3], line_color, thickness)
    c_img = cv2.line(c_img, points[0], points[4], line_color, thickness)
    c_img = cv2.line(c_img, points[0], points[1], line_color, thickness)
    
    c_img = cv2.line(c_img, points[5], points[1], line_color, thickness)
    c_img = cv2.line(c_img, points[5], points[4], line_color, thickness)
    c_img = cv2.line(c_img, points[5], points[6], line_color, thickness)
    
    c_img = cv2.line(c_img, points[2], points[1], line_color, thickness)
    c_img = cv2.line(c_img, points[2], points[3], line_color, thickness)
    c_img = cv2.line(c_img, points[2], points[6], line_color, thickness)
    
    c_img = cv2.line(c_img, points[7], points[6], line_color, thickness)  #9
    c_img = cv2.line(c_img, points[7], points[4], line_color, thickness)  #6
    c_img = cv2.line(c_img, points[7], points[3], line_color, thickness)  #12
  return c_img
