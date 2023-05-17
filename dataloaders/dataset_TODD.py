import sys
import numpy as np
import os
from PIL import Image
import json
from dataloaders.template import Panoptic, Stereo, OBB, Detection, LocalReadHandle
from dataloaders.tools import exr_loader
from dataloaders.dataset_blender import BlenderLocalDataset
import operator
from matplotlib import pyplot as plt
from pathlib import Path


class ToddLocalDataset(BlenderLocalDataset):
    def __init__(self, dataset_path, multiview=False, num_views=2, num_samples=43):
        super().__init__(dataset_path, multiview, num_views, num_samples)
        self.segmentation_ind = {}
    
    def LoadColor(self, file_path):
        left_img = Image.open(os.path.join(file_path, "image.jpg"))
        right_img = Image.open(os.path.join(file_path, "image.jpg"))
        left_img = np.array(left_img)
        right_img = np.array(right_img)
        return left_img, right_img
    
    def LoadDepth(self, file_path):
        depth_path = os.path.join(file_path, 'detph_GroundTruth.exr')
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
    
    def LoadSegmentation(self, file_path, dimension=(480, 640)):
        segmentation = np.array(Image.open(os.path.join(file_path, 'instance_segment.png')))[:,:, 0]
        for i in np.unique(segmentation):
            if i == 0 or i == 255:
                continue
            self.segmentation_ind[int(i/10 - 1)] = segmentation == i
        semantic_segmentation = segmentation.copy()
        semantic_segmentation[semantic_segmentation>=1] = 2.
        return segmentation, semantic_segmentation
    
    def LoadObjectPoses(self, file_path, img_size=(480,640)):

        # Open json & load object poses as seen in the camera coordinates
        print(file_path)
        f = open(os.path.join(file_path, 'pose_type.json'))
        data = json.load(f)
        f.close()

        # Get annotations for particular camera
        camera_annotations = data
        heatmaps = list()
        z_centroids = list()
        cov_matricies = list()
        not_in_scene = list()
        boxes = list()
        camera_intrinsic = np.array([[613.9624633789062, 0, 324.4471435546875], [0, 613.75634765625, 239.1712188720703], [0, 0, 1]])
        for object in camera_annotations.keys():
            object_annotations = camera_annotations[object]
            object = int(object)

            # Match to the segmentation of the object
            if object not in self.segmentation_ind.keys():
                print(f"{object} not in {self.segmentation_ind.keys()}")
                continue
            individual_segmentation = self.segmentation_ind[object]

            # Get the heatmap
            obj_heatmap = self.GetHeatMap(mask = individual_segmentation, dimensions = img_size)

            # Check if obj_heatmap is none, if none means object not in frame; remove object
            if isinstance(obj_heatmap, type(None)):
                print("heat map is none")
                not_in_scene.append(object)
                continue

            # Get Bounding Boxes
            object_bounding_boxes_c = np.array(object_annotations['3d']['3d']).T
            object_bounding_boxes = camera_intrinsic @ object_bounding_boxes_c
            object_bounding_boxes = object_bounding_boxes[:2] / object_bounding_boxes[2]
            object_bounding_boxes[0,:] = img_size[1] - object_bounding_boxes[0,:]
            object_bounding_boxes = object_bounding_boxes.T

            if object_bounding_boxes.shape[0] != 8:
                print("bbox out of frame")
                not_in_scene.append(object)
                continue
            boxes.append(object_bounding_boxes)
            heatmaps.append(obj_heatmap)

            # Need to get the covariance matrix
            cov_matrix = np.array(object_annotations['covariance'])
            cov_matricies.append(cov_matrix)

            # Get z_centroid
            z_centroid = np.average(np.array(object_annotations['3d']['3d']),axis = 0)[2]
            z_centroids.append(z_centroid)

        # Check if heatmaps are empty
        if len(heatmaps) == 0:
            print("no heat map")
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
            print(
                f'Heatmaps has problems! isnan = {np.isnan(heatmaps_combined).any()} and isinf = {np.isinf(heatmaps_combined).any()}, {heatmaps_combined}')
            return None
        if np.isnan(vertex_field).any() or np.isinf(vertex_field).any():
            print(
                f'vertex_field has problems! isnan = {np.isnan(vertex_field).any()} and isinf = {np.isinf(vertex_field).any()}, {vertex_field}')
            return None
        if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
            print(
                f'covariance_matrix has problems! isnan = {np.isnan(covariance_matrix).any()} and isinf = {np.isinf(covariance_matrix).any()}, {covariance_matrix}')
            return None
        if np.isnan(z_centroid_matrix).any() or np.isinf(z_centroid_matrix).any():
            print(
                f'z_centroid_matrix has problems! isnan = {np.isnan(z_centroid_matrix).any()} and isinf = {np.isinf(z_centroid_matrix).any()}, {z_centroid_matrix}')
            return None

        pose = OBB(heat_map = heatmaps_combined, vertex_target = vertex_field, z_centroid = z_centroid_matrix, cov_matrices = covariance_matrix)
        poses = [pose]
        return poses
    
    def LoadDetections(self, file_path, img_size=(512,640)):

        # Open json & load object poses as seen in the camera coordinates
        f = open(os.path.join(file_path, 'pose_type.json'))
        data = json.load(f)
        f.close()

        # Get annotations for particular camera
        camera_annotations = data

        not_in_scene = list()
        # boxes = list()
        detections = list()
        for object in camera_annotations.keys():
            object_annotations = camera_annotations[object]
            object=int(object)

            # Match to the segmentation of the object
            if object not in self.segmentation_ind.keys():
                print(f"{object} not in {self.segmentation_ind.keys()}")
                continue
            individual_segmentation = self.segmentation_ind[object]
            # Get the heatmap
            obj_heatmap = self.GetHeatMap(mask = individual_segmentation, dimensions = img_size)

            # Check if obj_heatmap is none, if none means object not in frame; remove object
            if isinstance(obj_heatmap, type(None)):
                print("detect heat map is none")
                not_in_scene.append(object)
                continue
            # Bbox
            try:
                bbox = np.array(object_annotations['3d']['image_frame'])
            except:
                print("detect missing bbox")
                continue
            detection = Detection(camera_T_object = np.array(object_annotations['3d']['3d']),
                        scale_matrix = np.zeros((4,4)),
                        bbox = bbox)
            detections.append(detection)
        return detections

    def LoadCameraParams(self, file_path):
        f = open(os.path.join(file_path, 'pose_type.json'))
        data = json.load(f)
        f.close()
        camera_annotations = data
        camera_extrinsic=None
        for object in camera_annotations.keys():
            object_pose = camera_annotations[object]["pose"]
            object_pose.append([0, 0, 0, 1])
            camera_extrinsic = np.linalg.inv(np.asarray(object_pose))
            break

        camera_params = {'focal_length': 613.9624633789062, 
                            'camera_intrinsic': [[613.9624633789062, 0, 324.4471435546875], [0, 613.75634765625, 239.1712188720703], [0, 0, 1]], 
                            'camera_location': None, 
                            'quaternion_WXYZ': None, 
                            'camera_extrinsic': camera_extrinsic, 
                            'sensor_width': None, 
                            'sensor_height': None, 
                            'sensor_fit': None, 
                            'resolution_y':480, 
                            'resolution_x':640, 
                            'baseline': None}
        return camera_params

    def GenerateData(self, save_pkl=False, start_at=None, end_at=None):
        handles = []
        scene_paths = [f.path for f in os.scandir(self.dataset_path) if f.is_dir()]
        for path_num, path in enumerate(scene_paths):
            uid = float(path.split('/')[-1])
            try:
                left_color, right_color = self.LoadColor(path)
                stereo = Stereo(left_color = left_color, right_color = right_color, compressed = False)
                depth = self.LoadDepth(path)
                segmentation, semantic_segmentation = self.LoadSegmentation(path)
                object_poses = self.LoadObjectPoses(path, img_size = (480,640))
                if isinstance(object_poses, type(None)):
                    print("object pose is none")
                    continue
                detections = self.LoadDetections(path, img_size = (480,640))
                camera_params = self.LoadCameraParams(path)

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
                                    uid = uid,
                                    compressed = False,
                                    camera_params = camera_params)
                if save_pkl:
                    self.write(datapoint)
                else:
                    print(str(uid))
                    handles.append(datapoint)
            except Exception as e:
                print(e)
                raise ValueError
        return sorted(handles, key=lambda x: x.uid)

    def LoadFromPkl(self, load_pkl=False):
        handles = []
        # Stereo version
        for path in self.dataset_path.glob('*.pickle.zstd'):
            uid = '.'.join(path.name.split('.')[:2])
            handles.append(LocalReadHandle(self.dataset_path, float(uid)))
        if not self.multiview:
            return sorted(handles, key=operator.attrgetter('uid'))
        else:  # multiview
            handles = sorted(handles, key=lambda x: x.uid)
            group_handle = []
            previous_time = 0
            diffs = []
            temp_mv_handles=[]
            view_count=0
            
            for pkl in handles:
                diff = pkl.uid - previous_time
                if np.abs(diff) < 100 and previous_time != 0:
                    if view_count < self.num_views:
                        temp_mv_handles.append(LocalReadHandle(self.dataset_path, float(pkl.uid)))
                        view_count += 1
                    else:
                        group_handle.append(temp_mv_handles)
                        temp_mv_handles = []
                        view_count = 0
                else:
                    temp_mv_handles = []
                    view_count = 0
                previous_time = pkl.uid
            return sorted(group_handle, key=lambda x: x[0].uid)


def plot(data,index, postfix='depth'):
  print('Plotting')
  print('UNIQUE', np.unique((((data-np.amin(data))/(np.amax(data) - np.amin(data)))*255).astype(np.uint8)))
  plt.imshow((((data-np.amin(data))/(np.amax(data) - np.amin(data)))*255).astype(np.uint8), interpolation='nearest')
  plt.show()


if __name__ == '__main__':
    # run data/TODD/object2bbox.py first before using TODD dataloader
    train_ds = ToddLocalDataset(Path('transparent-perception/data/TODD/val'))
    ds_items = train_ds.list(save_pkl=True, start_at=None, end_at=None)
    visualize_depth = True
    visualize_seg = True
    print(ds_items[0])
    if visualize_seg:
        for i in range(0, len(ds_items)):
            print(np.unique(ds_items[i].segmentation))
            plot(ds_items[i].segmentation, ds_items[i].uid, postfix='segmentation')
    if visualize_depth:
        for i in range(0, len(ds_items)):
            if np.isnan(ds_items[i].depth).any():
                print('Has NAN')
            if np.isinf(ds_items[i].depth).any():
                print('Has INF')
            if (ds_items[i].depth > 2).any():
                print('HAS LARGE VALS')
                ds_items[i].depth[ds_items[i].depth > 3] = 3.
