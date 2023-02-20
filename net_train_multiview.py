import sys
import os
import argparse
from importlib.machinery import SourceFileLoader
import sys
import random
import numpy as np
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from simnet.lib.net import common
from simnet.lib import datapoint, camera
from simnet.lib.net.post_processing.eval3d import Eval3d, extract_objects_from_detections
from simnet.lib.net.panoptic_trainer import PanopticModel

os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(12345)
random.seed(12345)
torch.manual_seed(12345)

_GPU_TO_USE = -1
_NUM_NODES=2

class EvalMethod():
  def __init__(self, mode = None):

    self.eval_3d = Eval3d()
    if mode == 'blender':
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
  if 'simnet' in hparams.train_path:
    training_mode = 'simnet'
  elif 'blender' or 'synthetic' in hparams.train_path:
    training_mode = 'blender'
  
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

  model = PanopticModel(hparams = hparams, 
                        epochs = epochs, 
                        train_dataset = train_ds, 
                        eval_metric = EvalMethod(mode = training_mode), 
                        val_dataset = val_ds)

  model_checkpoint = ModelCheckpoint(dirpath=hparams.output, save_top_k=-1, mode='min', save_last = True, monitor = 'val_loss')
  wandb_logger = loggers.WandbLogger(name=hparams.wandb_name, project='simnet')

  # Make output folder if doesn't exist
  if not os.path.exists(hparams.output):
    os.mkdir(hparams.output)
  # load checkpoints if exist 
  latest_ckpt = GetLatestCheckpoint(out_folder = hparams.output)
  if not latest_ckpt:
    print("Training from scratch...")
    trainer = pl.Trainer(
      accelerator="gpu",
      max_epochs=epochs,
      gpus=_GPU_TO_USE,
      checkpoint_callback=model_checkpoint,
      default_root_dir = hparams.output,
      check_val_every_n_epoch=1,
      logger=wandb_logger,
      strategy='ddp',
      )
  else:
    print('Training from checkpoint...')
    trainer = pl.Trainer(
      accelerator="gpu",
      max_epochs=epochs,
      gpus=_GPU_TO_USE,
      checkpoint_callback=model_checkpoint,
      default_root_dir = hparams.output,
      check_val_every_n_epoch=1,
      logger=wandb_logger,
      resume_from_checkpoint = latest_ckpt,
      strategy='ddp',
    )
  
  trainer.fit(model)
