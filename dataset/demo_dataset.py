import torch.utils.data as data
import numpy as np
import joblib
import torch
import time
import cv2

import time
class HabitatDemoSingleGoalDataset(data.Dataset):
    def __init__(self, cfg, data_list, include_stop = False):
        self.data_list = data_list
        self.img_size = (64, 256)
        self.action_dim = 4 if include_stop else 3
        self.max_demo_length = 100#cfg.dataset.max_demo_length

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        s = time.time()
        demo_data = joblib.load(self.data_list[index])
        #print('file loading time:', time.time() - s)
        scene = self.data_list[index].split('/')[-1].split('_')[0]
        start_pose = [demo_data['position'][0], demo_data['rotation'][0]]
        target_rgb, target_depth = demo_data['target_rgb'], demo_data['target_depth']
        target_goal = np.concatenate([target_rgb/255.,np.expand_dims(target_depth,2)],2)

        demo_rgb = np.array(demo_data['rgb'], dtype=np.float32)
        demo_dep = np.array(demo_data['depth'], dtype=np.float32)

        demo_length = len(demo_rgb) - 1
        #if demo_length > self.max_demo_length:
        #    print('longggg', self.data_list[index])
        demo_rgb_out = np.zeros([self.max_demo_length, demo_rgb.shape[1], demo_rgb.shape[2], 3])
        demo_rgb_out[:demo_length] = demo_rgb[:demo_length]
        demo_dep_out = np.zeros([self.max_demo_length, demo_rgb.shape[1], demo_rgb.shape[2], 1])
        demo_dep_out[:demo_length] = demo_dep[:demo_length]

        demo_act = np.array(demo_data['action'], dtype=np.int8)
        if self.action_dim > 3: demo_act[-1] = 0
        demo_act_out = np.ones([self.max_demo_length]) * (-100)
        # print(demo_act.shape, demo_length, 'rgbd', len(demo_data['rgb']), len(demo_data['depth']), len(demo_data['action']))
        demo_act_out[:demo_length] = demo_act

        targets = np.zeros([self.max_demo_length])
        targets[:demo_length] = 0
        target_img = np.zeros([1, demo_rgb.shape[1], demo_rgb.shape[2] , 4])
        target_num = 1
        target_img[:target_num] = np.array(target_goal)#[start_idx:start_idx+demo_length])
        positions = np.zeros([self.max_demo_length,3])
        positions[:demo_length] = demo_data['position'][:demo_length]
        return_tensor = [torch.from_numpy(demo_rgb_out).float(), torch.from_numpy(demo_dep_out).float(),
                         torch.from_numpy(demo_act_out).float(), torch.from_numpy(positions), targets,
                         torch.from_numpy(target_img).float(), scene, start_pose]
        return return_tensor
class HabitatDemoMultiGoalDataset(data.Dataset):
    def __init__(self, cfg, data_list, include_stop = False):
        self.data_list = data_list
        self.img_size = (64, 256)
        self.action_dim = 4 if include_stop else 3
        self.max_demo_length = 100#cfg.dataset.max_demo_length
        self.single_goal = False

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        s = time.time()
        demo_data = joblib.load(self.data_list[index])
        #print('file loading time:', time.time() - s)
        scene = self.data_list[index].split('/')[-1].split('_')[0]
        start_pose = [demo_data['position'][0], demo_data['rotation'][0]]
        target_indices = np.array(demo_data['target_idx'])

        # There are two random indices to sample
        # 1. when to start making graph
        # 2. when to start predict action

        # goals = np.unique(target_indices)
        #starts = [np.where(target_indices == g)[0].min() for g in goals]
        orig_data_len = len(demo_data['position'])
        if self.single_goal:
            try_num = 0
            while True:
                start_idx = np.random.randint(orig_data_len - 10) if orig_data_len > 10 else orig_data_len
                start_target_idx = target_indices[start_idx]
                end_idx = np.where(target_indices == start_target_idx)[0][-1]
                if end_idx - start_idx >= 10 : break
                try_num += 1
                if try_num > 1000:
                    end_idx = -1
                    break
        else:
            start_idx = np.random.randint(orig_data_len - 10) if orig_data_len > 10 else orig_data_len
            end_idx = - 1

        demo_rgb = np.array(demo_data['rgb'][start_idx:end_idx], dtype=np.float32)
        demo_length = np.minimum(len(demo_rgb), self.max_demo_length)

        demo_dep = np.array(demo_data['depth'][start_idx:end_idx], dtype=np.float32)

        demo_rgb_out = np.zeros([self.max_demo_length, demo_rgb.shape[1], demo_rgb.shape[2], 3])
        demo_rgb_out[:demo_length] = demo_rgb[:demo_length]
        demo_dep_out = np.zeros([self.max_demo_length, demo_rgb.shape[1], demo_rgb.shape[2], 1])
        demo_dep_out[:demo_length] = demo_dep[:demo_length]

        demo_act = np.array(demo_data['action'][start_idx:start_idx+demo_length], dtype=np.int8)
        if self.action_dim > 3: demo_act[-1] = 0
        demo_act_out = np.ones([self.max_demo_length]) * (-100)
        # print(demo_act.shape, demo_length, 'rgbd', len(demo_data['rgb']), len(demo_data['depth']), len(demo_data['action']))
        demo_act_out[:demo_length] = demo_act -1 if self.action_dim == 3 else demo_act

        targets = np.zeros([self.max_demo_length])
        targets[:demo_length] = demo_data['target_idx'][start_idx:start_idx+demo_length]
        target_img = np.zeros([5, demo_rgb.shape[1], demo_rgb.shape[2] , 4])
        target_num = len(demo_data['target_img'])
        target_img[:target_num] = np.array(demo_data['target_img'])#[start_idx:start_idx+demo_length])
        positions = np.zeros([self.max_demo_length,3])
        positions[:demo_length] = demo_data['position'][start_idx:start_idx+demo_length]
        return_tensor = [torch.from_numpy(demo_rgb_out).float(), torch.from_numpy(demo_dep_out).float(),
                         torch.from_numpy(demo_act_out).float(), torch.from_numpy(positions), targets,
                         torch.from_numpy(target_img).float(), scene, start_pose]
        return return_tensor

if __name__ == '__main__':
    import sys
    from IL_configs.default import get_config
    from dataset.demo_dataset import HabitatDemoMultiGoalDataset
    import os
    from tqdm import tqdm
    cfg = get_config('IL_configs/gmt.yaml')
    data_list = [os.path.join('/disk4/obin/vistarget_demo_gibson/train/random',x) for x in os.listdir('/disk4/obin/vistarget_demo_gibson/train/random')]
    data_list += [os.path.join('/disk4/obin/vistarget_demo_gibson/val/random',x) for x in os.listdir('/disk4/obin/vistarget_demo_gibson/val/random')]
    dataset = HabitatDemoMultiGoalDataset(cfg, data_list, True)
    print(len(dataset))
    for idx in tqdm(range(len(dataset))):
        if 'Angiola_019_env0.dat.gz' in dataset.data_list[idx]:
            dataset.pull_image(idx)

