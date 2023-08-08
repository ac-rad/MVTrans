import pathlib
from collections import defaultdict
import random
random.seed(10)
import json
import os
import glob
def get_neighbours(center_scene, num_multiview, 
                scene_num, scene_paths_list):
    # def get_neighbours(center_scene, num_multiview):

    num_rows = int(7/2)+1
    num_cols = 14

    neighbours = {0: [i for i in range(1,15)],
                1: [14,2,28,15,16],}

    id_num = int(center_scene.split('/')[-1].split('.')[0])
    root_path, _ = os.path.split(center_scene)
    cam_num = id_num % 100
    assert (id_num - cam_num) / 100 == scene_num

    valid_cams = [i for i in range(num_rows * num_cols)]

    # Get neighbouring cams
    if cam_num == 0:
        neighbours = [i for i in range(1,num_cols+1)]
    elif (cam_num % 14) == 1:
        if cam_num <= 1*num_cols:
            neighbours = [0,
                        cam_num + 13, cam_num + 1,
                        cam_num + 27, cam_num + 14, cam_num + 15]
        elif cam_num <= 2*num_cols:
            neighbours = [cam_num - 1, cam_num - 14, cam_num - 13, 
                        cam_num + 13, cam_num + 1,
                        cam_num + 27, cam_num + 14, cam_num + 15]
        elif cam_num <= 3*num_cols:
            neighbours = [cam_num - 1, cam_num - 14, cam_num - 13, 
                        cam_num + 13, cam_num + 1]
    elif (cam_num % 14) == 0:
        if cam_num <= 1*num_cols:
            neighbours = [0,
                        cam_num - 1, cam_num - 13,
                        cam_num + 13, cam_num + 14, cam_num + 1]
        elif cam_num <= 2*num_cols:
            neighbours = [cam_num - 15, cam_num - 14, cam_num - 27, 
                        cam_num - 1, cam_num - 13,
                        cam_num + 13, cam_num + 14, cam_num + 1]
        elif cam_num <= 3*num_cols:
            neighbours = [cam_num - 15, cam_num - 14, cam_num - 27, 
                        cam_num - 1, cam_num - 13]
    else:
        if cam_num <= 1*num_cols:
            neighbours = [0, 
                        cam_num - 1, cam_num + 1,
                        cam_num + 13, cam_num + 14, cam_num + 15]
        elif cam_num <= 2*num_cols:
            neighbours = [cam_num - 15, cam_num - 14, cam_num - 13, 
                        cam_num - 1, cam_num + 1,
                        cam_num + 13, cam_num + 14, cam_num + 15]
        elif cam_num <= 3*num_cols:
            neighbours = [cam_num - 15, cam_num - 14, cam_num - 13, 
                        cam_num - 1, cam_num + 1]
        else:
            raise ValueError
    # print(scene_paths_list)
    # print(neighbours)
    neighbour_paths = []
    for i in neighbours:
        assigned_id = i + scene_num*100
        neighbour_path = os.path.join(root_path, f'{assigned_id}.pickle.zstd')
        if not os.path.exists(neighbour_path):
            print('Does not exist: ', neighbour_path)
            continue
        else:
            neighbour_paths.append(neighbour_path)
    
    try: 
        # Sample x paths based on number of views
        sampled_paths = random.sample(neighbour_paths, num_multiview-1)
    except:
        print('Cannot sample')
        return False


    return [center_scene] + sampled_paths

if __name__ == '__main__':
    path_root = '/home/xuhaopin/scratch/helen/datasets/synthetic_dataset_pkl'
    dataset_path = pathlib.Path(path_root)
    scenes = defaultdict(list)
    num_multiview = 2
    num_samples = 25
    to_save = defaultdict(list)
    skipped_count = 0
    total_processed = 0
    reduction_factor = 2

    mode = 'neighbour' # {random, neighbour}
    
    print('Generating Scenes Dictionary')
    for path_name in sorted(glob.glob(os.path.join(path_root, '*.pickle.zstd')), key = lambda x: int(x.split('/')[-1].split('.')[0])):
        # Group paths by scene
        anchor_value = int(path_name.split('/')[-1].split('.')[0])

        scene_num = int(anchor_value / 100)

        scenes[scene_num].append(path_name)
    
    # Group paths together based on scene number and number of multiviews  
    
    for scene in scenes.keys():
        print(f'Processing {scene}')
        scene_paths_list = scenes[scene]
        for i in range(int(len(scene_paths_list)/reduction_factor)):
            if mode == 'random':
                combo = random.sample(scene_paths_list, num_multiview)
            elif mode == 'neighbour':
                cam_num = 43
                # Skip cam views parallel to table
                while cam_num > 42:
                    anchor_scene = random.sample(scene_paths_list, 1)
                    id_num = int(anchor_scene[0].split('/')[-1].split('.')[0])
                    cam_num = id_num % 100
                

                combo = get_neighbours(anchor_scene[0], num_multiview = num_multiview, scene_num = scene, scene_paths_list = scene_paths_list)
                if not combo:
                    skipped_count += 1
                    continue
            to_save['data'].append(combo)
            total_processed += 1
    
    to_save['num_multiview'] = num_multiview
    to_save['num_samples'] = num_samples

    # print(f'Saving to: { os.path.join(path_root , f'{num_multiview}_{num_samples}.json')}')
    # Save to json
    with open(os.path.join(path_root,f'{num_multiview}_{num_samples}.json'), 'w') as f:
        json.dump(to_save, f)
    print(total_processed)
    print(skipped_count)
