import torch
import torch.nn as nn

from src.lib.net.models import simplenet
from src.lib.net.post_processing import depth_outputs
from src.lib.net.models.panoptic_net import DepthHead, OBBHead, SegmentationHead, ShapeSpec
from model.networks.psm_submodule import psm_feature_extraction
from model.networks.resnet_encoder import ResnetEncoder
from model.hybrid_depth_decoder import DepthHybridDecoder
from model.networks.layers_op import convbn_3d, convbnrelu_3d
from utils.homo_utils import homo_warping


class MultiviewBackbone(nn.Module):

  def __init__(self, hparams, in_channels=3):
    super().__init__()

    def make_rgb_stem():
      net = simplenet.NetFactory()
      x = net.input(in_dim=3, stride=1, activated=True)
      x = net.downscale(x, 32)
      x = net.downscale(x, 32)
      return net.bake()

    def make_disp_features():
      net = simplenet.NetFactory()
      x = net.input(in_dim=1, stride=1, activated=False)
      x = net.layer(x, 32, rate=5)
      return net.bake()

    self.disp_features = make_disp_features()

    def make_rgbd_backbone(num_channels=64, out_dim=64):
      net = simplenet.NetFactory()
      x = net.input(in_dim=64, activated=True, stride=4)
      x = net._lateral(x, out_dim=num_channels)
      x4 = x = net.block(x, '111')
      x = net.downscale(x, num_channels * 2)
      x8 = x = net.block(x, '1111')
      x = net.downscale(x, num_channels * 4)
      x = net.block(x, '12591259')
      net.tag(net.output(x, out_dim), 'p4')
      x = net.upsample(x, x8, out_dim)
      net.tag(x, 'p3')
      x = net.upsample(x, x4, out_dim)
      net.tag(x, 'p2')
      return net.bake()

    self.rgbd_backbone = make_rgbd_backbone()
    self.reduce_channel = torch.nn.Conv2d(256, 32, 1)   
  def forward(self, img_features, small_disp, robot_joint_angles=None):
    small_disp = small_disp #self.stereo_stem.forward(stacked_img[:, 0:3], stacked_img[:, 3:6])
    left_rgb_features = self.reduce_channel(img_features)
    disp_features = self.disp_features(small_disp)
    rgbd_features = torch.cat((disp_features, left_rgb_features), axis=1)
    outputs = self.rgbd_backbone.forward(rgbd_features)
    outputs['small_disp'] = small_disp
    return outputs


class MultiviewNet(nn.Module):

  def __init__(self, hparams, ndepths=64, depth_min=0.01, depth_max=10.0, resnet=50):
    super().__init__()
    self.hparams = hparams
    self.ndepths = ndepths
    self.depth_min = depth_min
    self.depth_max = depth_max
    self.depth_interval = (depth_max - depth_min) / (ndepths - 1)

    # the to.(torch.float32) is required, if not will be all zeros
    self.depth_cands = torch.arange(0, ndepths, requires_grad=False).reshape(1, -1).to(
            torch.float32) * self.depth_interval + self.depth_min
    self.matchingFeature = psm_feature_extraction()
    self.semanticFeature = ResnetEncoder(resnet, "pretrained")  # the features after bn and relu
    self.multiviewBackbone = MultiviewBackbone(hparams)
    self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }
    
    self.pre0 = convbn_3d(64, 32, 1, 1, 0)
    self.pre1 = convbnrelu_3d(32, 32, 3, 1, 1)
    self.pre2 = convbn_3d(32, 32, 3, 1, 1)

    self.CostRegNet = DepthHybridDecoder(self.semanticFeature.num_ch_enc,
                                             num_output_channels=1, use_skips=True,
                                             ndepths=self.ndepths, depth_max=self.depth_max,
                                             IF_EST_transformer=False)


    # self.backbone = MultiviewBackbone(hparams)
    # ResFPN used p2,p3,p4,p5 (64 channels)
    # DRN uses only p2,p3,p4 (no need for p5 since dilation increases striding naturally)
    backbone_output_shape_4x = {
        #'p0': ShapeSpec(channels=64, height=None, width=None, stride=1),
        #'p1': ShapeSpec(channels=64, height=None, width=None, stride=2),
        'p2': ShapeSpec(channels=64, height=None, width=None, stride=4),
        'p3': ShapeSpec(channels=64, height=None, width=None, stride=8),
        'p4': ShapeSpec(channels=64, height=None, width=None, stride=16),
        #'p5': ShapeSpec(channels=64, height=None, width=None, stride=32),
    }

    backbone_output_shape_8x = {
        #'p0': ShapeSpec(channels=64, height=None, width=None, stride=1),
        #'p1': ShapeSpec(channels=64, height=None, width=None, stride=2),
        #'p2': ShapeSpec(channels=64, height=None, width=None, stride=4),
        'p3': ShapeSpec(channels=64, height=None, width=None, stride=8),
        'p4': ShapeSpec(channels=64, height=None, width=None, stride=16),
        #'p5': ShapeSpec(channels=64, height=None, width=None, stride=32),
    }

    # Add depth head.
    self.depth_head = DepthHead(backbone_output_shape_4x, backbone_output_shape_8x, hparams)
    # Add segmentation head.
    self.seg_head = SegmentationHead(backbone_output_shape_4x, backbone_output_shape_8x, 3, hparams)
    # Add pose heads.
    self.pose_head = OBBHead(backbone_output_shape_4x, backbone_output_shape_8x, hparams)
  
  def get_costvolume(self, features, cam_poses, cam_intr, depth_values):
    """
    return cost volume, [ref_feature, warped_feature] concat
    :param features: middle one is ref feature, others are source features
    :param cam_poses:
    :param cam_intr:
    :param depth_values:
    :return:
    """
    num_views = len(features)
    ref_feature = features[0]
    ref_cam_pose = cam_poses[:, 0, :, :]

    ref_extrinsic = torch.inverse(ref_cam_pose)
    # step 2. differentiable homograph, build cost volume
    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, self.ndepths, 1, 1)
    costvolume = torch.zeros_like(ref_volume).to(ref_volume.dtype).to(ref_volume.device)
    for view_i in range(num_views):
        if view_i == 0:
            continue
        src_fea = features[view_i]
        src_cam_pose = cam_poses[:, view_i, :, :]
        src_extrinsic = torch.inverse(src_cam_pose)
        # warpped features
        src_proj_new = src_extrinsic.clone()
        ref_proj_new = ref_extrinsic.clone()

        ref_proj_new = ref_proj_new.to('cpu')
        cam_intr = cam_intr.to('cpu')
        ref_extrinsic = ref_extrinsic.to('cpu')
        src_extrinsic = src_extrinsic.to('cpu')

        src_proj_new[:, :3, :4] = torch.matmul(cam_intr, src_extrinsic[:, :3, :4])
        ref_proj_new[:, :3, :4] = (cam_intr @ ref_extrinsic[:, :3, :4]).clone()
        ref_proj_new = ref_proj_new.to('cuda')
        cam_intr = cam_intr.to('cuda')
        ref_extrinsic = ref_extrinsic.to('cuda')
        src_extrinsic = src_extrinsic.to('cuda')

        warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

        # it seems that ref_volume - warped_volume not good
        x = torch.cat([ref_volume, warped_volume], dim=1)
        x = self.pre0(x)
        x = x + self.pre2(self.pre1(x))

        costvolume = costvolume + x
    # aggregate multiple feature volumes by variance
    costvolume = costvolume / (num_views - 1)
    del warped_volume
    del x
    return costvolume

  def scale_cam_intr(self, cam_intr, scale):
    cam_intr_new = cam_intr[0].clone()
    cam_intr_new[:, :2, :] *= scale
    cam_intr_new[:, :2, :] *= scale


    return cam_intr_new
  def forward(self, imgs, cam_poses, cam_intr, pre_costs=None, pre_cam_poses=None, mode='train'):
    """
        input seqs (0,1,2,3,4) target view will be (1,2,3) or input three views
        :param imgs:
        :param cam_poses:
        :param cam_intr:
        :param sample:
        :return:
    """
    imgs = 2 * (imgs / 255.) - 1.
    assert len(imgs.shape) == 5, 'expected imgs to be BxVxCxHxW'
    batch_size, views_num, _, height_img, width_img = imgs.shape

    height = height_img // 4
    width = width_img // 4

    assert views_num >= 2, f'View number should be greater 1, but is {views_num}'  # the views_num should be larger than 2

    target_num = 0

    # Convert list of tensors to tensor
    assert len(cam_poses.shape) == 4, f'expected shape to be len 4, got {cam_poses.shape}'

    matching_features = self.matchingFeature(imgs.view(batch_size * views_num, 3, height_img, width_img))

    matching_features = matching_features.view(batch_size, views_num, -1, height, width)

    matching_features = matching_features.permute(1, 0, 2, 3, 4).contiguous()

    semantic_features = self.semanticFeature(
            imgs[:, 0].view(batch_size, -1, height_img, width_img))
    
    cam_intr_stage1 = self.scale_cam_intr(cam_intr, scale=1. / self.stage_infos["stage1"]["scale"])

    depth_values = self.depth_cands.view(1, self.ndepths, 1, 1
                                             ).repeat(batch_size, 1, 1, 1).to(imgs.dtype).to(imgs.device)

    target_cam_poses = []

    # Get the cost volume
    cost_volume = self.get_costvolume(matching_features,
                                              cam_poses[:, :, :, :], #bs x views x 4 x 4
                                              cam_intr_stage1,
                                              depth_values)

    outputs, cur_costs, cur_cam_poses = self.CostRegNet(costvolumes = [cost_volume],
                                                            semantic_features = semantic_features,
                                                            cam_poses = target_cam_poses,
                                                            cam_intr = cam_intr_stage1,
                                                            depth_values = depth_values,
                                                            depth_min = self.depth_min,
                                                            depth_interval = self.depth_interval,
                                                            pre_costs = pre_costs, 
                                                            pre_cam_poses = pre_cam_poses,
                                                            mode = mode)

    # Convert depth to desired shape

    # Output a small displacement output (H/4, W/4)
    small_disp_output = outputs[("depth", 0, 2)]

    assert len(small_disp_output.shape) == 4, f'Expecting depth to be Nx1xHxW, but got {small_disp_output.shape}'

    # Get RGB Features
    features = self.multiviewBackbone(img_features = semantic_features[1],
                                    small_disp = small_disp_output)

    small_disp_output = small_disp_output.squeeze(dim=1)
    if self.hparams.frozen_stereo_checkpoint is not None:
      small_disp_output = small_disp_output.detach()
      assert False
    small_depth_output = depth_outputs.DepthOutput(small_disp_output, self.hparams.loss_depth_mult)
    seg_output = self.seg_head.forward(features)
    depth_output = self.depth_head.forward(features)
    pose_output = self.pose_head.forward(features)
    box_output = None
    keypoint_output = None
    return seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output

def res_fpn(hparams):
  return MultiviewNet(hparams)

if __name__ == '__main__':
  model = MultiviewNet()