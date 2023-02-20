import sys
import os
import argparse
from importlib.machinery import SourceFileLoader
import sys
import torch
import random
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from simnet.lib.net import common
from simnet.lib import datapoint, camera
from simnet.lib.net.post_processing.eval3d import Eval3d, extract_objects_from_detections
from simnet.lib.net.panoptic_trainer import PanopticModel

os.environ['PYTHONHASHSEED'] = str(1)
random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)

_GPU_TO_USE = [1]
_NUM_NODES=1


class EvalMethod():
  def __init__(self, mode = None):
    self.eval_3d = Eval3d()
    if mode == 'blender' or mode == 'TODD':
      self.camera_model = camera.BlenderCamera()
    elif mode == 'simnet':
      self.camera_model = camera.FMKCamera()
    else:
      raise ValueError

  def process_sample(self, pose_outputs, box_outputs, seg_outputs, detections_gt, scene_name):
    detections = pose_outputs.get_detections(self.camera_model)
    if scene_name != 'sim':
      table_detection, detections_gt, detections = extract_objects_from_detections(
          detections_gt, detections
      )
    self.eval_3d.process_sample(detections, detections_gt, scene_name)
    return True

  def process_all_dataset(self, log):
    log['all 3Dmap'] = self.eval_3d.process_all_3D_dataset()

  def draw_detections(
      self, pose_outputs, box_outputs, seg_outputs, keypoint_outputs, left_image_np, llog, prefix
  ):
    pose_vis = pose_outputs.get_visualization_img(
        np.copy(left_image_np), camera_model=self.camera_model
    )
    llog[f'{prefix}/pose'] = wandb.Image(pose_vis, caption=prefix)
    seg_vis = seg_outputs.get_visualization_img(np.copy(left_image_np))
    llog[f'{prefix}/seg'] = wandb.Image(seg_vis, caption=prefix)

  def reset(self):
    self.eval_3d = Eval3d()

def GetLatestCheckpoint(out_folder):
  if len(os.listdir(out_folder)) == 0:
    return False
  else:
    max_mtime = 0
    for dirname,subdirs,files in os.walk(out_folder):
        print(files)
        for fname in files:
            if not fname.endswith('.ckpt'):
              continue
            full_path = os.path.join(dirname, fname)
            mtime = os.stat(full_path).st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
                max_dir = dirname
                max_file = fname
    try:
      return os.path.join(max_dir,max_file)
    except:
      return False


if __name__ == "__main__":
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  hparams = parser.parse_args()

  # Get the mode of training
  print(hparams.train_path)
  if 'simnet' in hparams.train_path:
    training_mode = 'simnet'
  elif 'blender'  in hparams.train_path or 'synthetic' in hparams.train_path:
    training_mode = 'blender'
  elif 'TODD' in hparams.train_path:
    training_mode = 'TODD'
  
  print(f'Making the {training_mode} dataset')
  if training_mode == 'simnet':
    train_ds = datapoint.make_dataset(hparams.train_path)
    val_ds = datapoint.make_dataset(hparams.val_path)
  elif training_mode == 'blender':
    if hparams.network_type == 'multiview':
      train_ds = datapoint.make_dataset(hparams.train_path, dataset = 'blender', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
      val_ds = datapoint.make_dataset(hparams.val_path, dataset = 'blender', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
    elif hparams.network_type == 'simnet':
      train_ds = datapoint.make_dataset(hparams.train_path, dataset = 'blender')
      val_ds = datapoint.make_dataset(hparams.val_path, dataset = 'blender')
    else:
      raise ValueError
  elif training_mode == 'TODD':
    if hparams.network_type == 'multiview':
      train_ds = datapoint.make_dataset(hparams.train_path, dataset = 'TODD', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
      val_ds = datapoint.make_dataset(hparams.val_path, dataset = 'TODD', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
    else:
      raise ValueError

  samples_per_epoch = len(train_ds.list())
  samples_per_step = hparams.train_batch_size
  steps = hparams.max_steps
  steps_per_epoch = samples_per_epoch // samples_per_step
  epochs = int(np.ceil(steps / steps_per_epoch))
  actual_steps = epochs * steps_per_epoch
  print('Samples per epoch', samples_per_epoch)
  print('Steps per epoch', steps_per_epoch)
  print('Target steps:', steps)
  print('Actual steps:', actual_steps)
  print('Epochs:', epochs)

  # Login to wandb
  wandb.login(key='WANDB_KEY')

  ckpt_path="CHECKPOINT_PATH"
  model = PanopticModel.load_from_checkpoint(ckpt_path, hparams = hparams,
                        epochs = epochs,
                        train_dataset = train_ds,
                        eval_metric = EvalMethod(mode = training_mode),
                        val_dataset = val_ds,
                        automatic_optimization = False)

  model.eval()
  model.to(torch.device('cuda:0'))
  val = model.val_dataloader()
  for val_batch_idx, val_batch in enumerate(val):
    val_out = model.test_step(val_batch)
    print(val_out)
    exit(0)
