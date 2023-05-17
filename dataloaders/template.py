import dataclasses
from dataclasses import field
import numpy as np
import io
import shortuuid
from PIL import Image
import os
import pickle
import zstandard as zstd

def get_uid():
  return shortuuid.uuid()

# Struct for Pose Prediction
@dataclasses.dataclass
class Pose:
  heat_map: np.ndarray
  vertex_target: np.ndarray
  z_centroid: np.ndarray


# Struct for Keypoint Prediction
@dataclasses.dataclass
class Keypoint:
  heat_map: np.ndarray


# Struct for Oriented Bounding Box Prediction
@dataclasses.dataclass
class OBB:
  heat_map: np.ndarray
  vertex_target: np.ndarray
  z_centroid: np.ndarray
  cov_matrices: np.ndarray
  compressed: bool = False
  obj_names: list = field(default_factory=list)

  def compress(self):
    if self.compressed:
      return
    # Heat map scale by 4x and quantize
    height, width = self.heat_map.shape
    self.heat_map = self.heat_map.astype(np.float16)
    self.vertex_target = self.vertex_target.transpose(2, 0, 1).astype(np.float16)
    self.compressed = True

  def decompress(self):
    if not self.compressed:
      return

    # Heat map scale by 4x and quantize
    height, width = self.heat_map.shape
    self.heat_map = self.heat_map.astype(np.float32)

    # Vertex field, quantize and transpose to make vertex field smooth in memory order (makes
    # downstream compression 50x more effective)
    self.vertex_target = self.vertex_target.astype(np.float32).transpose(1, 2, 0)
    self.compressed = False

@dataclasses.dataclass
class Detection:
  camera_T_object: np.ndarray
  scale_matrix: np.ndarray
  class_label: str = None
  size_label: str = ''
  scene_name: str = ''
  ignore: bool = False
  bbox: list = None
  score: float = 1.0
  success: int = 0
  obj_CAD: int = 0
  object_name: str = ''

def compress_color_image(img, quality=90):
  with io.BytesIO() as buf:
    img = Image.fromarray(img)
    img.save(buf, format='jpeg', quality=quality)
    return buf.getvalue()


def decompress_color_image(img_bytes):
  with io.BytesIO(img_bytes) as buf:
    img = Image.open(buf)
    return np.array(img)


#Struct for Stereo Representation
@dataclasses.dataclass
class Stereo:
  left_color: np.ndarray
  right_color: np.ndarray
  compressed: bool = False

  def compress(self):
    if self.compressed:
      return
    self.left_color = compress_color_image(self.left_color)
    self.right_color = compress_color_image(self.right_color)
    self.compressed = True

  def decompress(self):
    if not self.compressed:
      return
    self.left_color = decompress_color_image(self.left_color)
    self.right_color = decompress_color_image(self.right_color)
    self.compressed = False


# Application Specific Datapoints Should be specified here.
@dataclasses.dataclass
class Panoptic:
  stereo: Stereo
  depth: np.ndarray
  segmentation: np.ndarray
  instance_segmentation: np.ndarray
  object_poses: list
  boxes: list
  detections: list
  camera_params: list = dataclasses.field(default_factory=list)
  keypoints: list = dataclasses.field(default_factory=list)
  instance_mask: np.ndarray = None
  scene_name: str = 'sim'
  uid: str = dataclasses.field(default_factory=get_uid)
  compressed: bool = False

  def compress(self):
    self.stereo.compress()
    for object_pose in self.object_poses:
      object_pose.compress()

    if self.compressed:
      return

    # Depth scale by 4x and quantize
    height, width = self.depth.shape
    self.depth = self.depth.astype(np.float16)
    self.compressed = True

  def decompress(self):
    self.stereo.decompress()
    for object_pose in self.object_poses:
      object_pose.decompress()

    if not self.compressed:
      return

    # Depth scale by 4x and quantize
    height, width = self.depth.shape
    self.depth = self.depth.astype(np.float32)
    self.compressed = False

def decompress_datapoint(cbuf, disable_final_decompression=False):
  cctx = zstd.ZstdDecompressor()
  buf = cctx.decompress(cbuf)
  x = pickle.loads(buf)
  if not disable_final_decompression:
    x.decompress()
  return x


class LocalReadHandle:

  def __init__(self, dataset_path, uid):
    self.dataset_path = dataset_path
    self.uid = uid

  def read(self, disable_final_decompression=False):
    path = os.path.join(self.dataset_path, f'{self.uid}.pickle.zstd')
    with open(path, 'rb') as fh:
      dp = decompress_datapoint(fh.read(), disable_final_decompression=disable_final_decompression)
    if not hasattr(dp, 'uid'):
      dp.uid = self.uid
    assert dp.uid == self.uid, f'dp uid is {dp.uid}, self.uid is {self.uid}'
    return dp
