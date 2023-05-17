import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from glob import glob
import json
from scipy.stats import multivariate_normal
import pickle
import zstandard as zstd
import copy
import operator

from src.lib.net.post_processing import obb_outputs, depth_outputs, segmentation_outputs
from dataloaders.template import Panoptic, Stereo, OBB, LocalReadHandle, Detection
from dataloaders.tools import exr_loader


def extract_left_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
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
  image = image.transpose((2, 0, 1))
  return torch.from_numpy(np.ascontiguousarray(image)).float()

class BlenderLocalDataset:

  def __init__(self, dataset_path, multiview = False, num_views = 2, num_samples = 56):
    assert os.path.isdir(dataset_path), f'dataset_path is {dataset_path}, which does not exist'
    self.dataset_path = dataset_path
    self._PEAK_CONCENTRATION = 0.8  
    self._DOWNSCALE_VALUE = 8
    self.multiview = multiview
    self.num_views = num_views
    self.num_samples = num_samples

  def compress_datapoint(self, x):
    # Compress and save
    x = copy.deepcopy(x)
    x.compress()
    buf = pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    cctx = zstd.ZstdCompressor()
    cbuf = cctx.compress(buf)
    return cbuf

  def decompress(self, cbuf, disable_final_decompression=False):
    cctx = zstd.ZstdDecompressor()
    buf = cctx.decompress(cbuf)
    x = pickle.loads(buf)
    if not disable_final_decompression:
      x.decompress()
    return x


  def LoadColor(self, file_path, camera):
    left_img = Image.open(os.path.join(file_path, f"Camera_{camera}_EmptyVessel_Frame_0_RGB_L.jpg"))
    right_img = Image.open(os.path.join(file_path, f"Camera_{camera}_EmptyVessel_Frame_0_RGB_R.jpg"))
    left_img = np.array(left_img)
    right_img = np.array(right_img)
    return left_img, right_img

  def LoadDepth(self, file_path, camera):
    '''
    Function for loading depth given filepath and camera
    file_path: str
      path to exr file
    camera: int
      camera number
    '''
    # File name format: Camera_0_EmptyVessel_Frame_0_Depth
    depth_path = os.path.join(file_path, f"Camera_{camera}_EmptyVessel_Frame_0_Depth.exr")
    depth = exr_loader(depth_path, ndim = 1)

    # Process depth
    if np.isnan(depth).any():
      print('Has NAN')
      depth_mask_nan = np.isnan(depth)
      depth[depth_mask_nan] = 3.0
      
    if np.isinf(depth).any():
      print('Has INF')
      depth_mask_inf = np.isinf(depth)
      depth[depth_mask_inf] = 3.0

    if (depth > 3).any():
      print('HAS LARGE VALS')
      depth[depth > 3] = 3.
      
    return depth
  

  def LoadSegmentation(self, file_path, camera, dimension = (480, 640)):

    '''
    Returns a single segmentation map with all objects
    file_path: str
    camera: int
    dimension: tuple(y, x)
    '''

    def GetBWColourScheme(num_colors):
      '''
      Computes an array of evenly spaced floats
      '''
      colours = np.linspace(0, 1, num_colors)
      return colours
    
    # List all files
    segmentation_files = glob(os.path.join(file_path, f"segmentation_Camera_{camera}_*_visible_L.png"))

    # Filter the segmentation_files
    segmentation_files = [i for i in segmentation_files if ('_Content_' not in i)]  

    # Colour of non-blending pixels
    no_blend = np.array([[206,206,206], [177,0,7], [0,0,0]])
    flatten_no_blend = np.dot(no_blend.astype(np.uint32),[1,256,65536])

    # Colour of objects
    object_colour = np.array([[206,206,206]])
    flatten_object_colour = np.dot(object_colour.astype(np.uint32),[1,256,65536])

    # Colour of surface plane
    surface_colour = np.array([[0,0,0]])
    flatten_surface_colour = np.dot(surface_colour.astype(np.uint32),[1,256,65536])

    # Table Surface
    table_mask = np.full((dimension[0]*dimension[1],), True)

    # Accumulator of object masks
    obj_masks = list()

    # Store individual segmentations
    self.segmentation_ind = {}
    
    # Construct segmentation maps
    for idx, file in enumerate(segmentation_files):

      # Ignore content and full segmentation mask annotations
      if '_Content_' in file:
        continue
      if '_full_' in file:
        continue

      # Load RGB segmentation
      rgb_seg = np.array(Image.open(file))

      # Get the object name
      obj_name = file.split('/')[-1].split(f'segmentation_Camera_{camera}_')[-1].split('_visible_')[0]

      # "Flatten" third dimension to single 24-bit number
      flatten_rgb_seg = np.dot(rgb_seg.astype(np.uint32),[1,256,65536])

      # Get pixels which have blending
      no_blend_pixels = np.in1d(flatten_rgb_seg, flatten_no_blend) # Checked

      # Convert blended pixels to black
      red_pixels = (~no_blend_pixels.reshape(dimension)) & ((rgb_seg[:,:,0] > 110) & (rgb_seg[:,:,1] < 100))
      black_pixels = ~no_blend_pixels.reshape(dimension)
      white_pixels = (~no_blend_pixels.reshape(dimension)) & ((rgb_seg[:,:,0] > 110))
      rgb_seg[black_pixels] = np.array([0,0,0])
      rgb_seg[white_pixels] = np.array([206,206,206])
      rgb_seg[red_pixels] = np.array([177,0,7])

      # "Flatten" third dimension to single 24-bit number
      flatten_rgb_seg = np.dot(rgb_seg.astype(np.uint32),[1,256,65536])

      # Now use np.in1d() to get object mask
      obj_mask = np.in1d(flatten_rgb_seg,flatten_object_colour)

      # Get the table surface mask, excluding all objects
      table_mask = np.logical_and(table_mask,np.in1d(flatten_rgb_seg,flatten_surface_colour))

      # Reshape masks into images
      obj_mask = obj_mask.reshape(dimension)
      rgb_seg = rgb_seg.reshape(dimension + (3,))
      assert obj_mask.shape == rgb_seg.shape[0:2], f'Incorrect shape, expected {rgb_seg.shape[0:2]}, got {obj_mask.shape}'

      # Fill object mask with a unique number
      obj_mask = obj_mask.astype(int)*(idx + 2)
      if len(obj_name) > 38:
        obj_name = obj_name[:38]
      self.segmentation_ind[obj_name] = obj_mask.copy()
      obj_masks.append(obj_mask)

    # Reshape table mask
    table_mask = table_mask.reshape(dimension)

    # Replace table_mask with color
    table_mask[table_mask == True] = 1. #colour_scheme[-1]
    table_mask[table_mask == False] = 0.
    self.segmentation_ind['table_mask'] = table_mask
    
    # Add the mask values
    segmentation = table_mask
    for i in obj_masks:
      segmentation = segmentation + i
    semantic_segmentation = segmentation.copy()
    semantic_segmentation[semantic_segmentation>=2] = 2.
    return segmentation, semantic_segmentation # 0 = background; 1 = table; 2:n = objects
  
  def resize_with_pad(im, target_width, target_height, colour):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
      # It must be fixed by width
      resize_width = target_width
      resize_height = round(resize_width * im_ratio)
    else:
      # Fixed by height
      resize_height = target_height
      resize_width = round(resize_height / im_ratio)

    # Differentiate between single channel and 2 channel
    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    if len(im.mode) == 1:
      background = Image.new('L', (target_width, target_height), 0.)
      offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
      background.paste(image_resize, offset)
      return np.array(background.convert('L'))
    elif len(im.mode) == 3:
      background = Image.new('RGBA', (target_width, target_height), (0, 0, 0))
      offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
      background.paste(image_resize, offset)
      return np.array(background.convert('RGB'))
    else:
      raise ValueError(f'Image channels is invalid, received {im.mode}')
    
      

  def ReshapeArray(self, old_array, new_shape, color = (255,0,0)):
    old_image_height, old_image_width, channels = old_array.shape

    # Check if manipulation is needed
    if (old_image_height == new_shape[0]) and (old_image_width == new_shape[1]):
      return old_array

    # create new image of desired size and color (blue) for padding
    new_image_width = new_shape[1]
    new_image_height = new_shape[0]
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
          x_center:x_center+old_image_width] = old_array

    return result
  
  def CombineHeatmap(self, heatmaps):
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    heatmap_combined = np.zeros(heatmaps[0].shape)
    for heat_map, ii in zip(heatmaps, range(len(heatmaps))):
      mask = (heatmap_indices == ii)
      heatmap_combined[mask] = heat_map[mask]
    return heatmap_combined


  def GetHeatMap(self, mask, dimensions = (480,640)):
    
    # Check if object present in scene
    if np.sum(mask) == 0:
      return None

    # Get the coordinates of pixels corresponding to mask shape
    coords = np.indices(mask.shape)
    coords = coords.reshape([2, -1]).T

    # Flatten mask
    mask_f = mask.flatten()

    # Find indicies where an object is present
    indices = coords[np.where(mask_f != 0)]

    # Check if multiple pixels are present
    if indices.shape[0] <= 1:
      return None

    # Get the center pixel coordinates
    mean_value = np.floor(np.average(indices, axis=0))

    # Compute covariance
    cov = np.cov((indices - mean_value).T)

    # Reduce covariance by peak concentration parameter
    cov = cov * self._PEAK_CONCENTRATION

    # Construct a multivariate normal
    try:
      multi_var = multivariate_normal(mean=mean_value, cov=cov)
    except Exception as e:
      print(f'error in Multivariate Normal, mean {mean_value}, cov {cov}')
      return None
    # Compute probability density function for all coordinate values
    density = multi_var.pdf(coords)

    # Build heatmap
    heat_map = np.zeros(mask.shape)
    heat_map[coords[:, 0], coords[:, 1]] = density

    # TODO: Reshape heatmap if necessary
    if heat_map.shape != dimensions:
      new_heatmap = self.ReshapeArray(self, old_array = heat_map, new_shape = dimensions, color = (255,0,0))
      return new_heatmap / np.max(new_heatmap)
    return heat_map / np.max(heat_map)
  
  def compute_vertex_field(self,heatmaps, boxes):
    # Shape of heatmap
    H, W = heatmaps[0].shape[0], heatmaps[0].shape[1]
    disp_fields = []
    vertex_target = np.zeros([len(boxes), int(H / self._DOWNSCALE_VALUE), int(W / self._DOWNSCALE_VALUE), 16])
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    for i in range(8):
      vertex_points = []
      coords = np.indices([H, W])
      coords = coords.transpose((1, 2, 0))
      for box_idx, bbox, heatmap in zip(range(len(boxes)), boxes, heatmaps):
        disp_field = np.zeros([H, W, 2])
        vertex_point = np.array([bbox[i][1], bbox[i][0]])

        mask = (heatmap_indices == box_idx)
        disp_field[mask] = coords[mask] - vertex_point
        # Normalize by height and width
        disp_field[mask, 0] = 1.0 - (disp_field[mask, 0] + H) / (2 * H)
        disp_field[mask, 1] = 1.0 - (disp_field[mask, 1] + W) / (2 * W)
        vertex_target[box_idx, :, :,
                      (2 * i):2 * i + 2] = disp_field[::self._DOWNSCALE_VALUE, ::self._DOWNSCALE_VALUE]
    return np.max(vertex_target, axis=0)

  def compute_rotation_field(self, cov_matricies, heat_maps, threshold=0.3):
    cov_target = np.zeros([len(cov_matricies), heat_maps[0].shape[0], heat_maps[0].shape[1], 6])
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for cov_matrix, heat_map, ii in zip(cov_matricies, heat_maps, range(len(heat_maps))):
      mask = (heatmap_indices == ii)
      cov_mat_values = np.array([
          cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[2, 2], cov_matrix[0, 1], cov_matrix[0, 2],
          cov_matrix[1, 2]
      ])
      cov_target[ii, mask] = cov_mat_values
    return np.sum(cov_target, axis=0)[::self._DOWNSCALE_VALUE, ::self._DOWNSCALE_VALUE]
  
  def compute_z_centroid_field(self,z_vals, heatmaps):
    z_centroid_target = np.zeros([len(z_vals), heatmaps[0].shape[0], heatmaps[0].shape[1]])
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    for pose, heat_map, ii in zip(z_vals, heatmaps, range(len(heatmaps))):
      mask = (heatmap_indices == ii)
      z_centroid_target[ii, mask] = pose
    # Normalize z_centroid by 1. and multiply by 10 to avoid tensorrt float precision issues.
    return np.sum(z_centroid_target, axis=0)[::self._DOWNSCALE_VALUE, ::self._DOWNSCALE_VALUE]
  
  def LoadObjectPoses(self, file_path, camera, img_size = (480,640)):
    # Open json & load object poses as seen in the camera coordinates
    f = open(os.path.join(file_path, 'BoundingBox.json'))
    data = json.load(f)
    f.close()

    # Get annotations for particular camera
    camera_annotations = data[f'Camera_{camera}']
    heatmaps = list()
    z_centroids = list()
    cov_matricies = list()
    not_in_scene = list()
    boxes = list()
    names = list()
    to_ignore = ['focal_length', 'camera_intrinsic', 'camera_location', 'quaternion_WXYZ', 'camera_extrinsic', 'sensor_width', 'sensor_height', 'sensor_fit', 'resolution_y', 'resolution_x', 'baseline']
    camera_intrinsic = np.array(camera_annotations['camera_intrinsic'])
    for object in camera_annotations.keys():
      object_annotations = camera_annotations[object]

      if 'transparent' in object:
        object = object.split('transparent_')[-1]
      elif object in to_ignore:
        continue

      # Shorten object name:
      if len(object) > 38:
        object = object[:38]

      # Match to the segmentation of the object
      assert object in self.segmentation_ind.keys(), f'Object name {object} not in self.segmentation keys {self.segmentation_ind.keys()}'
      individual_segmentation = self.segmentation_ind[object]

      # Get the heatmap
      obj_heatmap = self.GetHeatMap(mask = individual_segmentation, dimensions = img_size)

      # Check if obj_heatmap is none, if none means object not in frame; remove object
      if isinstance(obj_heatmap, type(None)):
        not_in_scene.append(object)
        continue

      # Get Bounding Boxes
      object_bounding_boxes_i = np.array(object_annotations['3dbox']['image_frame'])
      object_bounding_boxes_c = np.array(object_annotations['3dbox']['camera_frame']).T
      object_bounding_boxes = camera_intrinsic @ object_bounding_boxes_c
      object_bounding_boxes = object_bounding_boxes[:2] / object_bounding_boxes[2]
      object_bounding_boxes[0,:] = img_size[1] - object_bounding_boxes[0,:]
      object_bounding_boxes = object_bounding_boxes.T

      if object_bounding_boxes.shape[0] != 8:
        not_in_scene.append(object)
        continue
      boxes.append(object_bounding_boxes)
      heatmaps.append(obj_heatmap)
      names.append(object)

      # Need to get the covariance matrix
      cov_matrix = np.array(object_annotations['covariance'])
      cov_matricies.append(cov_matrix)

      # Get z_centroid
      z_centroid = np.average(np.array(object_annotations['3dbox']['camera_frame']),axis = 0)[2]
      z_centroids.append(z_centroid)

    # Check if heatmaps are empty
    if len(heatmaps) == 0:
      return None
    # Get combined heatmap
    heatmaps_combined = self.CombineHeatmap(heatmaps)
    # Compute vertex array
    vertex_field = self.compute_vertex_field(heatmaps = heatmaps, boxes = boxes)

    # Compute covariance matrix
    covariance_matrix = self.compute_rotation_field(cov_matricies = cov_matricies, heat_maps = heatmaps, threshold=0.3)

    # Compute z_centroid matrix
    z_centroid_matrix = self.compute_z_centroid_field(z_vals = z_centroids, heatmaps = heatmaps)

    # Check if any values are nan or inf
    if np.isnan(heatmaps_combined).any() or np.isinf(heatmaps_combined).any():
      print(f'Heatmaps has problems! isnan = {np.isnan(heatmaps_combined).any()} and isinf = {np.isinf(heatmaps_combined).any()}, {heatmaps_combined}')
      return None
    if np.isnan(vertex_field).any() or np.isinf(vertex_field).any():
      print(f'vertex_field has problems! isnan = {np.isnan(vertex_field).any()} and isinf = {np.isinf(vertex_field).any()}, {vertex_field}')
      return None
    if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
      print(f'covariance_matrix has problems! isnan = {np.isnan(covariance_matrix).any()} and isinf = {np.isinf(covariance_matrix).any()}, {covariance_matrix}')
      return None
    if np.isnan(z_centroid_matrix).any() or np.isinf(z_centroid_matrix).any():
      print(f'z_centroid_matrix has problems! isnan = {np.isnan(z_centroid_matrix).any()} and isinf = {np.isinf(z_centroid_matrix).any()}, {z_centroid_matrix}')
      return None

    pose = OBB(heat_map = heatmaps_combined, vertex_target = vertex_field, z_centroid = z_centroid_matrix, cov_matrices = covariance_matrix, obj_names = names)
    poses = [pose]
    return poses

  def LoadDetections(self, file_path, camera, img_size = (512,640)):

    # Open json & load object poses as seen in the camera coordinates
    f = open(os.path.join(file_path, 'BoundingBox.json'))
    data = json.load(f)
    f.close()

    # Get annotations for particular camera
    camera_annotations = data[f'Camera_{camera}']

    not_in_scene = list()
    # boxes = list()
    detections = list()
    to_ignore = ['focal_length', 'camera_intrinsic', 'camera_location', 'quaternion_WXYZ', 'camera_extrinsic', 'sensor_width', 'sensor_height', 'sensor_fit', 'resolution_y', 'resolution_x', 'baseline']
    for object in camera_annotations.keys():
      object_annotations = camera_annotations[object]

      if 'transparent' in object:
        object = object.split('transparent_')[-1]
      elif object in to_ignore:
        continue
      
      # Shorten object name:
      if len(object) > 38:
        object = object[:38]

      # Match to the segmentation of the object
      assert object in self.segmentation_ind.keys(), f'Object name {object} not in self.segmentation keys {self.segmentation_ind.keys()}'
      individual_segmentation = self.segmentation_ind[object]
      # Get the heatmap
      obj_heatmap = self.GetHeatMap(mask = individual_segmentation, dimensions = img_size)

      # Check if obj_heatmap is none, if none means object not in frame; remove object
      if isinstance(obj_heatmap, type(None)):
        not_in_scene.append(object)
        continue
      # Bbox
      try:
        bbox = np.array([[np.amin(np.array([object_annotations['2dbox_imgframe']['x1'], 
                object_annotations['2dbox_imgframe']['x2']])), 
                np.amin(np.array([object_annotations['2dbox_imgframe']['y1'], 
                object_annotations['2dbox_imgframe']['y2']]))], 
                [np.amax(np.array([object_annotations['2dbox_imgframe']['x1'], 
                object_annotations['2dbox_imgframe']['x2']])), 
                np.amax(np.array([object_annotations['2dbox_imgframe']['y1'], 
                object_annotations['2dbox_imgframe']['y2']]))]])
      except:
        continue
      detection = Detection(camera_T_object = np.array(object_annotations['6dpose_c_obj']),
                scale_matrix = np.zeros((4,4)), object_name = object,
                bbox = bbox)
      detections.append(detection)
    x,y = np.where(self.segmentation_ind['table_mask'])
    top_left = [x.min(), y.min()]
    bottom_right = [x.max(), y.max()]
    table_bbox = np.array([top_left, bottom_right])
    # Add table bounding box
    table_detection = Detection(camera_T_object = np.array([[1,0,0,0], [0,1,0,0],[0.,0.,1.,0.], [0.,0.,0.,1.]]),
                scale_matrix = np.zeros((4,4)),
                bbox = table_bbox)
    detections = [table_detection] + detections
    return detections

  def LoadCameraParams(self, file_path, camera, img_size = (512,640)):
    # Open json & load object poses as seen in the camera coordinates
    f = open(os.path.join(file_path, 'BoundingBox.json'))
    data = json.load(f)
    f.close()

    # Get annotations for particular camera
    camera_annotations = data[f'Camera_{camera}']

    # Make the dictionary
    camera_params = {}
    camera_params_str = ['focal_length', 'camera_intrinsic', 'camera_location', 'quaternion_WXYZ', 'camera_extrinsic', 'sensor_width', 'sensor_height', 'sensor_fit', 'resolution_y', 'resolution_x', 'baseline']
    for object in camera_annotations.keys():
      object_annotations = camera_annotations[object]
      if object not in camera_params_str:
        continue
      camera_params[object] = object_annotations
    return camera_params

  def GenerateData(self, save_pkl = False, start_at = None, end_at = None):
    handles = []
    scene_paths = [f.path for f in os.scandir(self.dataset_path) if f.is_dir()]
    if start_at or end_at:
      start_at = start_at if start_at else 0
      end_at = end_at if end_at else len(scene_paths)
      scene_paths = scene_paths[start_at:end_at]
    cams_to_select = range(56)

    for path_num, path in enumerate(scene_paths):
      uid = int(path.split('/')[-1])
      for cam in cams_to_select:
        try:
          left_color, right_color = self.LoadColor(path, cam)
          stereo = Stereo(left_color = left_color, right_color = right_color, compressed = False)
          depth = self.LoadDepth(path, cam)
          segmentation, semantic_segmentation = self.LoadSegmentation(path, cam)
          object_poses = self.LoadObjectPoses(file_path = path, camera = cam, img_size = (480,640))
          if isinstance(object_poses, type(None)):
            continue
          detections = self.LoadDetections(file_path = path, camera = cam, img_size = (480,640))
          camera_params = self.LoadCameraParams(file_path = path, camera = cam)

          # Unused methods
          boxes = []
          keypoints = []
          instance_mask = None
          datapoint = Panoptic(stereo = stereo,
                                  depth = depth,
                                  segmentation = semantic_segmentation,
                                  instance_segmentation = segmentation,
                                  object_poses = object_poses,
                                  boxes = None,
                                  detections = detections,
                                  keypoints = None,
                                  instance_mask = None,
                                  scene_name = None,
                                  uid = uid*100 + int(cam),
                                  compressed = False,
                                  camera_params = camera_params)
          if save_pkl:
            self.write(datapoint)
          else:
            print(str(uid*100 + int(cam)))
            handles.append(datapoint)
        except:
          continue
    return sorted(handles, key = lambda x: x.uid)
  
  def LoadFromPkl(self, load_pkl = False):
    handles = []
    # Stereo version
    if not self.multiview:
      for path in self.dataset_path.glob('*.pickle.zstd'):
        uid = path.name.partition('.')[0]
        handles.append(LocalReadHandle(self.dataset_path, int(uid)))
      return sorted(handles, key=operator.attrgetter('uid'))
    else: # multiview
      # Opening JSON file
      f = open(os.path.join(self.dataset_path, f'{self.num_views}_{self.num_samples}.json'))
      data = json.load(f)
      groupings = data['data']
      f.close()
      
      for group in groupings:
        group_handle = []
        for element in group:
          # Get the uid
          uid = os.path.basename(element).split('.')[0]
          group_handle.append(LocalReadHandle(self.dataset_path, int(uid)))
        handles.append(group_handle)
      return sorted(handles, key= lambda x: x[0].uid)

  def list(self, save_pkl = False, start_at = None, end_at = None):
    load_pkl = True if 'pkl' in str(self.dataset_path) else False
    # Check if mode is to load pkl or generate data
    if not load_pkl:
      return self.GenerateData(save_pkl = save_pkl, start_at = start_at, end_at = end_at)
    else:
      return self.LoadFromPkl()

  def write(self, datapoint, postfix = '_pkl'):
    path = os.path.join(f'{self.dataset_path}{postfix}', f'{datapoint.uid}.pickle.zstd')
    if not os.path.exists(f'{self.dataset_path}{postfix}'):
      os.mkdir(f'{self.dataset_path}{postfix}')
    buf = self.compress_datapoint(datapoint)
    print('Writing to: ', path)
    with open(path, 'wb') as fh:
      fh.write(buf)

class BlenderDataset(Dataset):

  def __init__(self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None):
    super().__init__()

    if datapoint_dataset is None:
      datapoint_dataset = BlenderLocalDataset('/h/helen/transparent-perception/synthetic_dataset')
    self.datapoint_handles = datapoint_dataset.list()
    self.hparams = hparams
    if preprocess_image_func is None:
      self.preprocces_image_func = create_anaglyph
    else:
      assert False
      self.preprocces_image_func = preprocess_image_func

  def __len__(self):
    return len(self.datapoint_handles)

  def __getitem__(self, idx):
    dp = self.datapoint_handles[idx].read()
    anaglyph = self.preprocces_image_func(dp.stereo)
    segmentation_target = segmentation_outputs.SegmentationOutput(dp.segmentation, self.hparams)
    segmentation_target.convert_to_torch_from_numpy()
    depth_target = depth_outputs.DepthOutput(dp.depth, self.hparams)
    depth_target.convert_to_torch_from_numpy()
    pose_target = None
    for pose_dp in dp.object_poses:
      pose_target = obb_outputs.OBBOutput(
          pose_dp.heat_map, pose_dp.vertex_target, pose_dp.z_centroid, pose_dp.cov_matrices,
          self.hparams, pose_dp.names
      )
      pose_target.convert_to_torch_from_numpy()

    box_target = None
    kp_target = None
    scene_name = dp.scene_name
    return anaglyph, segmentation_target, depth_target, pose_target, box_target, kp_target, dp.detections, scene_name


def plot(data,index, postfix = 'depth'):
  from matplotlib import pyplot as plt
  print('Plotting')
  print('UNIQUE', np.unique((((data-np.amin(data))/(np.amax(data) - np.amin(data)))*255).astype(np.uint8)))
  plt.imshow((((data-np.amin(data))/(np.amax(data) - np.amin(data)))*255).astype(np.uint8), interpolation='nearest')
  plt.savefig(f'/h/helen/transparent-perception/tests/{index}_{postfix}.png')
  plt.close()

if __name__ == '__main__':
  train_ds = BlenderLocalDataset('/home/chemrobot/hdd/helen/synthetic_dataset_val')
  ds_items = train_ds.list(save_pkl = True, start_at = 4, end_at = 50)

  visualize_depth = False
  visualize_seg = False
  if visualize_depth:
    for i in range(0, len(ds_items)):
      if np.isnan(ds_items[i].depth).any():
        print('Has NAN')
      if np.isinf(ds_items[i].depth).any():
        print('Has INF')
      if (ds_items[i].depth > 3).any():
        print('HAS LARGE VALS')
        ds_items[i].depth[ds_items[i].depth > 3] = 3.

  if visualize_seg:
    for i in range(0, len(ds_items)):
      print(np.unique(ds_items[i].segmentation))
      plot(ds_items[i].segmentation, ds_items[i].uid, postfix = 'segmentation')
