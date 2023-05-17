import os
import copy
os.environ['PYTHONHASHSEED'] = str(1)
import random
random.seed(123456)
import numpy as np
np.random.seed(123456)
import torch
torch.manual_seed(123456)
import wandb
import pytorch_lightning as pl
import csv

from src.lib.net import common
from src.lib.net.dataset import extract_left_numpy_img
from src.lib.net.functions.learning_rate import lambda_learning_rate_poly, lambda_warmup

_GPU_TO_USE = [0]


class PanopticModel(pl.LightningModule):

  def __init__(
      self, hparams, epochs=None, train_dataset=None, eval_metric=None, preprocess_func=None, val_dataset = None, automatic_optimization = True, masked_depth = True
  ):
    super().__init__()
    self.automatic_optimization = automatic_optimization
    self.hparams_model = hparams
    self.epochs = epochs
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.model = common.get_model(hparams)
    self.eval_metrics = eval_metric
    self.preprocess_func = preprocess_func
    self.masked_depth = masked_depth

  def forward(self, image, cam_poses = None, cam_intr = None, mode = 'train'):

    if self.hparams_model.network_type == 'simnet':
      seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output = self.model(
          image, self.global_step
      )
    elif self.hparams_model.network_type == 'multiview':
      seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output = self.model(
          imgs = image, cam_poses = cam_poses, cam_intr = cam_intr, mode = mode 
      )
    else:
      raise ValueError(f'Network type not supported: {self.hparams_model.network_type}')
    return seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output

  def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=None,using_native_amp=None, 
                    using_lbfgs=None):
    super().optimizer_step(epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs)
    learning_rate = 0.0
    for param_group in optimizer.param_groups:
      learning_rate = param_group['lr']
      break
    self.logger.experiment.log({'learning_rate': learning_rate})

  def get_cad_paths(self, scenes):

    if 'synthetic' in self.hparams_model.train_path:
      train_path = self.hparams_model.train_path.split('://')[-1]
      head, tail = os.path.split(train_path)
      tail_modified = tail[:-4]
      to_remove = ['Camera', 'Content', 'Cube', 'Ground', 'Light']
      to_return = []
      for scene in scenes:
        scene = int(scene[0] / 100)
        scene_path = os.path.join(head, f'{tail_modified}_models', str(scene))
        all_files = os.listdir(scene_path)
        glb_files = [file for file in all_files if file.endswith('.glb')]
        # Remove random models
        cad_paths_for_scene = []
        for path in glb_files:
          good = True
          for remove in to_remove:
            if remove in path:
              good = False
              break
          if good:
            cad_paths_for_scene.append(os.path.join(scene_path, path))
        to_return.append(cad_paths_for_scene)
      return to_return
    else:
      raise NotImplementedError

  def training_step(self, batch, batch_idx):
    if not self.automatic_optimization:
      opt = self.optimizers()
      opt.zero_grad()

    if self.hparams_model.network_type == 'simnet':
      image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
          image
      )
    elif self.hparams_model.network_type == 'multiview':
      image, camera_poses, camera_intrinsic, seg_target, depth_target, pose_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
          image = image, cam_poses = camera_poses, cam_intr = camera_intrinsic, mode = 'train'
      )
    else:
      raise ValueError(f'Network type not supported: {self.hparams_model.network_type}')

    log = {}

    # depth_loss=0
    depth_loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, 'refined_disp')
    loss = depth_loss

    # raise ValueError
    if self.masked_depth:
      if 'synthetic' in self.hparams_model.train_path:
        mask = (seg_target[0].seg_pred ==2)
      elif 'TODD' in self.hparams_model.train_path: # Keyword may need to change
        mask = (seg_target[0].seg_pred != 0)
      else:
        raise NotImplementedError

      MAE_mean, MAE_median, RMSE_mean, RMSE_median, REL_mean, REL_median = depth_output.compute_metrics(depth_target, log, masks = mask)
    else:
      MAE_mean, MAE_median, RMSE_mean, RMSE_median, REL_mean, REL_median = depth_output.compute_metrics(depth_target, log)

    if self.hparams_model.frozen_stereo_checkpoint is None:
      small_depth_loss = small_depth_output.compute_loss(depth_target, log, 'cost_volume_disp')
      loss = loss + small_depth_loss
    else:
      assert False
    seg_loss = seg_output.compute_loss(seg_target, log)
    IoU, mAP = seg_output.compute_metrics(seg_target, log)
    loss = loss + seg_loss
    if pose_targets[0] is not None:
      pose_loss = pose_outputs.compute_loss(pose_targets, log)
      loss = loss + pose_loss
      cad_list=[None]
      td_iou, td_ap, ADD, ADD_s, AUC, less2cm, AUC_adds, less2cm_adds = pose_outputs.compute_metrics(pose_targets, 
                                    log, camera_model = self.eval_metrics.camera_model, cad_list = cad_list[0]) 

      # Write to file
      stats_path = os.path.join(self.hparams_model.output,
                            f'{self.current_epoch}_{self.hparams_model.num_multiview}_{self.hparams_model.num_samples}_stats_mask.csv')
      stats_header = ['MAE_mean', 'MAE_median', 'RMSE_mean', 'RMSE_median', 'REL_mean', 'REL_median',
                      'Seg_IoU', 'Seg_mAP', 'td_iou', 'td_ap', 'ADD', 'ADD_s', 'AUC', 'less2cm', 'AUC_adds', 'less2cm_adds']

      stats = {'MAE_mean': MAE_mean, 
                'MAE_median': MAE_median, 
                'RMSE_mean': RMSE_mean, 
                'RMSE_median': RMSE_median, 
                'REL_mean': REL_mean, 
                'REL_median': REL_median, 
                'Seg_IoU': IoU.item(), 
                'Seg_mAP': mAP,
                'td_iou': td_iou,
                'td_ap': td_ap,
                'ADD': ADD,
                'ADD_s': ADD_s,
                'AUC': AUC,
                'less2cm': less2cm,
                'AUC_adds': AUC_adds,
                'less2cm_adds': less2cm_adds}
      
      with open(stats_path, 'a') as file:

          writer = csv.DictWriter(file, delimiter=',', lineterminator='\n',fieldnames=stats_header)

          # writer = csv.writer(file)
          if not os.path.exists(stats_path):
            writer.writerow(stats_header)
          writer.writerow(stats)
    
    log['train/loss/total'] = loss
    log['train/loss/depth_loss'] = depth_loss
    log['train/loss/seg_loss'] = seg_loss
    log['train/loss/small_depth_loss'] = small_depth_loss
    log['train/loss/pose_loss'] = pose_loss
    
    if torch.isnan(loss) or torch.isinf(loss):
      pose_targets_to_print = vars(pose_targets[0])
      depth_targets_to_print = vars(depth_target[0])
      seg_targets_to_print = vars(seg_target[0])

      for attribute in pose_targets_to_print.keys():
        if not torch.is_tensor(pose_targets_to_print[attribute]):
          continue
        assert not torch.isnan(pose_targets_to_print[attribute]).any(), f'{uid}: {attribute} from pose has nan {pose_targets_to_print[attribute]}'
        assert not torch.isinf(pose_targets_to_print[attribute]).any(), f'{uid}: {attribute} from pose has inf {pose_targets_to_print[attribute]}'
      for attribute in depth_targets_to_print.keys():
        if not torch.is_tensor(depth_targets_to_print[attribute]):
          continue
        assert not torch.isnan(depth_targets_to_print[attribute]).any(), f'{uid}: {attribute} from depth has nan {depth_targets_to_print[attribute]}'
        assert not torch.isinf(depth_targets_to_print[attribute]).any(), f'{uid}: {attribute} from depth has inf {depth_targets_to_print[attribute]}'

      for attribute in seg_targets_to_print.keys():
        if not torch.is_tensor(seg_targets_to_print[attribute]):
          continue
        assert not torch.isnan(seg_targets_to_print[attribute]).any(), f'{uid}: {attribute} from seg has nan {seg_targets_to_print[attribute]}'
        assert not torch.isinf(seg_targets_to_print[attribute]).any(), f'{uid}: {attribute} from seg has inf {seg_targets_to_print[attribute]}'
      
      pose_outputs_to_print = vars(pose_outputs)
      depth_outputs_to_print = vars(depth_output)
      seg_outputs_to_print = vars(seg_output)
      
      for attribute in pose_outputs_to_print.keys():
        if not torch.is_tensor(pose_outputs_to_print[attribute]):
          continue
        assert not torch.isnan(pose_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from pose has nan {pose_outputs_to_print[attribute]}'
        assert not torch.isinf(pose_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from pose has inf {pose_outputs_to_print[attribute]}'

      for attribute in depth_outputs_to_print.keys():
        if not torch.is_tensor(depth_outputs_to_print[attribute]):
          continue
        assert not torch.isnan(depth_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from depth has nan {depth_outputs_to_print[attribute]}'
        assert not torch.isinf(depth_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from depth has inf {depth_outputs_to_print[attribute]}'

      for attribute in seg_outputs_to_print.keys():
        if not torch.is_tensor(seg_outputs_to_print[attribute]):
          continue
        assert not torch.isnan(seg_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from seg has nan {seg_outputs_to_print[attribute]}'
        assert not torch.isinf(seg_outputs_to_print[attribute]).any(), f'{uid}: {attribute} from seg has inf {seg_outputs_to_print[attribute]}'

    logger = self.logger.experiment
    if (batch_idx < 50
    ):
      with torch.no_grad():
        llog = {}
        # print('*'*50, uid)
        prefix = f'train/{batch_idx}'
        left_image_np = extract_left_numpy_img(image[0], mode = self.hparams_model.network_type)
        
        seg_pred_vis = seg_output.get_visualization_img(np.copy(left_image_np))
        seg_target_vis = seg_target[0].get_visualization_img_gt(np.copy(left_image_np))
        llog[f'{prefix}/rgb'] = wandb.Image(left_image_np[...,::-1].copy(), caption=prefix)
        llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)
        llog[f'{prefix}/seg_gt'] = wandb.Image(seg_target_vis, caption=prefix + '_gt')

        if pose_targets[0] is not None:
          pose_vis = pose_outputs.get_visualization_img(
              np.copy(left_image_np), camera_model=self.eval_metrics.camera_model
          )
          pose_target_vis = pose_targets[0].get_visualization_img_gt(
              np.copy(left_image_np), camera_model=self.eval_metrics.camera_model
          )
          llog[f'{prefix}/pose'] = wandb.Image(pose_vis, caption=prefix)
          llog[f'{prefix}/pose_gt'] = wandb.Image(pose_target_vis, caption=prefix+ '_gt')

        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        depth_target_vis = depth_target[0].get_visualization_img_gt(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        llog[f'{prefix}/disparity_gt'] = wandb.Image(depth_target_vis, caption=prefix+'_gt')
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        logger.log(llog)
    logger.log(log)

    if not self.automatic_optimization:
      loss = loss.detach()
      opt.step()
      return

    return {'loss': loss, 'log': log}

  def test_step(self, batch):
    if self.hparams_model.network_type == 'simnet':
      image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image
      )
    elif self.hparams_model.network_type == 'multiview':
      image, camera_poses, camera_intrinsic, seg_target, depth_target, pose_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image=image.to(torch.device('cuda:0')), cam_poses=camera_poses.to(torch.device('cuda:0')), cam_intr=[camera_intrinsic[0].to(torch.device('cuda:0'))], mode='val'
      )

    else:
      raise ValueError(f'Network type not supported: {self.hparams_model.network_type}')
    log = {}

    depth_loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, 'refined_disp')
    loss = depth_loss

    # raise ValueError
    if self.masked_depth:
      if 'synthetic' in self.hparams_model.train_path:
        mask = (seg_target[0].seg_pred == 2)
      elif 'TODD' in self.hparams_model.train_path:  # Keyword may need to change
        mask = (seg_target[0].seg_pred != 0)
      else:
        raise NotImplementedError

      MAE_mean, MAE_median, RMSE_mean, RMSE_median, REL_mean, REL_median = depth_output.compute_metrics(depth_target,
                                                                                                        log, masks=mask)
    else:
      MAE_mean, MAE_median, RMSE_mean, RMSE_median, REL_mean, REL_median = depth_output.compute_metrics(depth_target,
                                                                                                        log)

    if self.hparams_model.frozen_stereo_checkpoint is None:
      small_depth_loss = small_depth_output.compute_loss(depth_target, log, 'cost_volume_disp')
      loss = loss + small_depth_loss
    else:
      assert False
    seg_loss = seg_output.compute_loss(seg_target, log)
    IoU, mAP = seg_output.compute_metrics(seg_target, log)
    loss = loss + seg_loss
    if pose_targets[0] is not None:
      pose_loss = pose_outputs.compute_loss(pose_targets, log)
      loss = loss + pose_loss
      cad_list = [None]
      td_iou, td_ap, ADD, ADD_s, AUC, less2cm, AUC_adds, less2cm_adds = pose_outputs.compute_metrics(pose_targets,
                                                                                                     log,
                                                                                                     camera_model=self.eval_metrics.camera_model,
                                                                                                     cad_list=cad_list[
                                                                                                       0])

      # Write to file
      stats_path = os.path.join(self.hparams_model.output,
                                f'{self.current_epoch}_{self.hparams_model.num_multiview}_{self.hparams_model.num_samples}_stats_mask.csv')
      stats_header = ['MAE_mean', 'MAE_median', 'RMSE_mean', 'RMSE_median', 'REL_mean', 'REL_median',
                      'Seg_IoU', 'Seg_mAP', 'td_iou', 'td_ap', 'ADD', 'ADD_s', 'AUC', 'less2cm', 'AUC_adds',
                      'less2cm_adds']

      stats = {'MAE_mean': MAE_mean,
               'MAE_median': MAE_median,
               'RMSE_mean': RMSE_mean,
               'RMSE_median': RMSE_median,
               'REL_mean': REL_mean,
               'REL_median': REL_median,
               'Seg_IoU': IoU.item(),
               'Seg_mAP': mAP,
               'td_iou': td_iou,
               'td_ap': td_ap,
               'ADD': ADD,
               'ADD_s': ADD_s,
               'AUC': AUC,
               'less2cm': less2cm,
               'AUC_adds': AUC_adds,
               'less2cm_adds': less2cm_adds}
    return stats
  
  def validation_step(self, batch, batch_idx):

    self.model.train()
    #if corl.sim_on_sim_overfit:
    #  # If we are overfitting on sim data set batch size to 1 and enable batch norm for val to make
    #  # it match train. this doesn't make sense unless trying to get val and train to match
    #  # perfectly on a single sample for an overfit test

    if self.hparams_model.network_type == 'simnet':
      image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, scene_name = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
          image
      )
    elif self.hparams_model.network_type == 'multiview':
      image, camera_poses, camera_intrinsic, seg_target, depth_target, pose_targets, _, scene_name = batch
      assert image.shape[1] == camera_poses.shape[1], f'dimension mismatch: num of imgs {image.shape} not equal to num of camera poses {camera_poses.shape}'
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
          image = image, cam_poses = camera_poses, cam_intr = camera_intrinsic, mode = 'val' 
      )
    else:
      raise ValueError(f'Network type not supported: {self.hparams_model.network_type}')

    log = {}
    with torch.no_grad():

      depth_loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, 'refined_disp')
      loss = depth_loss
      if self.hparams_model.frozen_stereo_checkpoint is None:
        small_depth_loss = small_depth_output.compute_loss(depth_target, log, 'cost_volume_disp')
        loss = loss + small_depth_loss
      else:
        assert False
      seg_loss = seg_output.compute_loss(seg_target, log)
      loss = loss + seg_loss
      if pose_targets[0] is not None:
        pose_loss = pose_outputs.compute_loss(pose_targets, log)
        loss = loss + pose_loss
      
      log['val/loss/total'] = loss
      log['val/loss/depth_loss'] = depth_loss
      log['val/loss/seg_loss'] = seg_loss
      log['val/loss/small_depth_loss'] = small_depth_loss
      log['val/loss/pose_loss'] = pose_loss

      logger = self.logger.experiment
      if batch_idx < 5 or scene_name[0] == 'fmk':
        llog = {}
        left_image_np = extract_left_numpy_img(image[0], mode = self.hparams_model.network_type)
        prefix = f'val/{batch_idx}'
        logger = self.logger.experiment
        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        self.eval_metrics.draw_detections(
            pose_outputs, box_outputs, seg_output, keypoint_outputs, left_image_np, llog, prefix
        )

        logger.log(llog)
    logger.log(log)
    return {'val_loss': loss, 'log': log}

  def validation_epoch_end(self, outputs):
    self.trainer.checkpoint_callback.save_best_only = False
    log = {}
    self.eval_metrics.process_all_dataset(log)
    self.eval_metrics.reset()
    return {'log': log}
  
  def test_step(self, batch):
    if self.hparams_model.network_type == 'simnet':
      image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image
      )

    elif self.hparams_model.network_type == 'multiview':
      image, camera_poses, camera_intrinsic, seg_target, depth_target, pose_targets, _, uid = batch
      seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image=image.to(torch.device('cuda:0')), cam_poses=camera_poses.to(torch.device('cuda:0')), cam_intr=[camera_intrinsic[0].to(torch.device('cuda:0'))], mode='val'
      )
    else:
      raise ValueError(f'Network type not supported: {self.hparams_model.network_type}')
    poses = pose_outputs.get_detections(self.eval_metrics.camera_model)
    seg = seg_output.get_prediction().astype(np.uint8)[0]
    depth_output.convert_to_numpy_from_torch()
    depth = depth_output.depth_pred[0]
    return poses, seg, depth

  # @pl.data_loader
  def train_dataloader(self):
    return common.get_loader(
        self.hparams_model,
        "train",
        preprocess_func=self.preprocess_func,
        datapoint_dataset=self.train_dataset
    )

  # @pl.data_loader
  def val_dataloader(self):
    return common.get_loader(self.hparams_model, 
                            "val",
                             preprocess_func=self.preprocess_func,
                             datapoint_dataset=self.val_dataset
                             )
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_model.optim_learning_rate)
    lr_lambda = lambda_learning_rate_poly(self.epochs, self.hparams_model.optim_poly_exp)
    if self.hparams_model.optim_warmup_epochs is not None and self.hparams_model.optim_warmup_epochs > 0:
      lr_lambda = lambda_warmup(self.hparams_model.optim_warmup_epochs, 0.2, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return [optimizer], [scheduler]
