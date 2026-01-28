import os
import numpy as np
import h5py
#import hdf5plugin

import torch
from numpy.distutils.system_info import gtkp_x11_2_info
from torch.utils.data import Dataset
import torch.nn.functional as F

from event_process import events_to_voxel_grid, filter_events_spatial, filter_events_temporal


class Dataloader_train(Dataset):
    def __init__(self, args):
        super(Dataloader_train, self).__init__()
        self.args = args
        self.dataset_path = args.dataset_path
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        h5_name = os.path.basename(file_name)  # 00000000.h5
        seq = os.path.basename(os.path.dirname(file_name))  # 000

        timestamps, events = load_data_single(file_name)
        true_0, true_1 = load_rgb_pair(
            self.dataset_path, 'train', seq, h5_name
        )
        gt = load_gt_frames(
            self.dataset_path, 'train', seq, h5_name
        )

        # short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        # short_1 = np.transpose(short_1, (2,0,1))
        # long_0 = np.transpose(long_0, (2,0,1))
        # long_1 = np.transpose(long_1, (2,0,1))
        # relong_0 = np.transpose(relong_0, (2,0,1))
        # relong_1 = np.transpose(relong_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        gt = np.transpose(gt, (2,0,1))
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = true_0.shape
        timestamp_target = timestamps[0]
        timestamp_start = timestamps[2]
        timestamp_end = timestamps[4]

        """ -------------------- random crop -------------------- """
        if true_0.shape[1] > self.args.crop_height and true_0.shape[2] > self.args.crop_width:
            y = np.random.randint(low=1, high=(true_0.shape[1] - self.args.crop_height + 1))
            x = np.random.randint(low=1, high=(true_0.shape[2] - self.args.crop_width + 1))

            # short_0 = short_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            # short_1 = short_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            # long_0 = long_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            # long_1 = long_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            # relong_0 = relong_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            # relong_1 = relong_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_0 = true_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_1 = true_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            gt = gt[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            
            events = np.stack((event_t, event_x, event_y, event_p), axis=1).astype(np.float64)
            events = filter_events_spatial(events, y, x, self.args.crop_height, self.args.crop_width)
            events_0 = filter_events_temporal(events, timestamp_target, timestamp_start)
            events_1 = filter_events_temporal(events, timestamp_target, timestamp_end)
            evg_0 = events_to_voxel_grid(events_0, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)
            evg_1 = events_to_voxel_grid(events_1, num_bins=6, width=self.args.crop_width, height=self.args.crop_height)

        """ -------------------- to tensor -------------------- """
        # short_0 = torch.from_numpy(short_0).float() / 255.
        # short_1 = torch.from_numpy(short_1).float() / 255.
        # long_0 = torch.from_numpy(long_0).float() / 255.
        # long_1 = torch.from_numpy(long_1).float() / 255.
        # relong_0 = torch.from_numpy(relong_0).float() / 255.
        # relong_1 = torch.from_numpy(relong_1).float() / 255.
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.
        gt = torch.from_numpy(gt).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()

        # c,h,w = evg_0.shape
        # evg_0 = torch.cat((torch.sum(evg_0.view(1,-1,h,w), dim=1), torch.sum(evg_0.view(3,-1,h,w), dim=1), evg_0), 0)
        # evg_1 = torch.cat((torch.sum(evg_1.view(1,-1,h,w), dim=1), torch.sum(evg_1.view(3,-1,h,w), dim=1), evg_1), 0)

        # return short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, evg_0, evg_1, prefix
        return torch.cat([true_0, true_1], dim=0), gt, torch.cat([evg_0, evg_1], dim=0)


    def readFilePaths(self, suffix='.h5'):
        file_names = []
        path = os.path.join(self.args.dataset_path, 'train', 'train_h5')
        for seq in sorted(os.listdir(path)):
            for bag in sorted(os.listdir(os.path.join(path, seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(path, seq, bag))

        return file_names


class Dataloader_val(Dataset):
    def __init__(self, args):
        super(Dataloader_val, self).__init__()
        self.args = args
        self.dataset_path = args.dataset_path
        self.file_names = self.readFilePaths(suffix='.h5')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """ -------------------- load all data -------------------- """
        file_name = self.file_names[idx]
        h5_name = os.path.basename(file_name)  # 00000000.h5
        seq = os.path.basename(os.path.dirname(file_name))  # 000

        timestamps, events = load_data_single(file_name)
        true_0, true_1 = load_rgb_pair(
            self.dataset_path, 'val', seq, h5_name
        )
        gt = load_gt_frames(
            self.dataset_path, 'val', seq, h5_name
        )

        # short_0 = np.transpose(short_0, (2,0,1))  # [h,w,3] -> [3,h,w]
        # short_1 = np.transpose(short_1, (2,0,1))
        # long_0 = np.transpose(long_0, (2,0,1))
        # long_1 = np.transpose(long_1, (2,0,1))
        # relong_0 = np.transpose(relong_0, (2,0,1))
        # relong_1 = np.transpose(relong_1, (2,0,1))
        true_0 = np.transpose(true_0, (2,0,1))
        true_1 = np.transpose(true_1, (2,0,1))
        gt = np.transpose(gt, (2, 0, 1))
        event_t = events['t'][:]
        event_x = events['x'][:]
        event_y = events['y'][:]
        event_p = events['p'][:]

        """ -------------------- get voxel grid -------------------- """
        _,h,w = true_0.shape
        timestamp_target = timestamps[0]
        timestamp_start = timestamps[2]
        timestamp_end = timestamps[4]
        
        eidx = np.logical_and(event_t>=timestamp_target, event_t<=timestamp_start)
        events_0 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        # evg_0 = events_to_polarity_integration(events_0, num_bins=16, width=w, height=h)
        evg_0 = events_to_voxel_grid(events_0, num_bins=6, width=w, height=h)

        eidx = np.logical_and(event_t>=timestamp_target, event_t<=timestamp_end)
        events_1 = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float64)
        # evg_1 = events_to_polarity_integration(events_1, num_bins=16, width=w, height=h)
        evg_1 = events_to_voxel_grid(events_1, num_bins=6, width=w, height=h)

        """ -------------------- random crop --------------------
        if long_0.shape[1] > self.args.crop_height and long_0.shape[2] > self.args.crop_width:
            y = np.random.randint(low=1, high=(long_0.shape[1] - self.args.crop_height + 1))
            x = np.random.randint(low=1, high=(long_0.shape[2] - self.args.crop_width + 1))

            short_0 = short_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            short_1 = short_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_0 = long_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            long_1 = long_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_0 = relong_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            relong_1 = relong_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_0 = true_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            true_1 = true_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            evg_0 = evg_0[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            evg_1 = evg_1[..., y:y + self.args.crop_height, x:x + self.args.crop_width] """

        """ -------------------- to tensor -------------------- """
        # short_0 = torch.from_numpy(short_0).float() / 255.
        # short_1 = torch.from_numpy(short_1).float() / 255.
        # long_0 = torch.from_numpy(long_0).float() / 255.
        # long_1 = torch.from_numpy(long_1).float() / 255.
        # relong_0 = torch.from_numpy(relong_0).float() / 255.
        # relong_1 = torch.from_numpy(relong_1).float() / 255.
        true_0 = torch.from_numpy(true_0).float() / 255.
        true_1 = torch.from_numpy(true_1).float() / 255.
        gt = torch.from_numpy(gt).float() / 255.

        evg_0 = torch.from_numpy(evg_0).float()
        evg_1 = torch.from_numpy(evg_1).float()

        c,h,w = evg_0.shape
        # evg_0 = torch.cat((torch.sum(evg_0.view(1,-1,h,w), dim=1), torch.sum(evg_0.view(3,-1,h,w), dim=1), evg_0), 0)
        # evg_1 = torch.cat((torch.sum(evg_1.view(1,-1,h,w), dim=1), torch.sum(evg_1.view(3,-1,h,w), dim=1), evg_1), 0)

        return torch.cat([true_0, true_1], dim=0), gt, torch.cat([evg_0, evg_1], dim=0)

    def readFilePaths(self, suffix='.h5'):
        file_names = []
        path = os.path.join(self.args.dataset_path, 'val', 'val_h5')
        for seq in sorted(os.listdir(path)):
            for bag in sorted(os.listdir(os.path.join(path, seq))):
                if os.path.splitext(bag)[-1] == suffix:
                    file_names.append(os.path.join(path, seq, bag))

        return file_names


# def load_data_single(file_name):
#     # load data
#     h5_file = os.path.join(file_name)
#     h5 = h5py.File(h5_file, "r")
#
#     timestamps = h5['timestamps'][:]
#     # short_0 = h5['short_0'][:]
#     # short_1 = h5['short_1'][:]
#     # long_0 = h5['long_0'][:]
#     # long_1 = h5['long_1'][:]
#     # relong_0 = h5['relong_0'][:]
#     # relong_1 = h5['relong_1'][:]
#     true_0 = h5['true_0'][:]
#     true_1 = h5['true_1'][:]
#     gt_0 = h5['gt_0'][:]
#     gt_1 = h5['gt_1'][:]
#     gt_2 = h5['gt_2'][:]
#     events = h5['events']
#
#     prefix = file_name
#     return timestamps,true_0, true_1, events,gt_0, gt_1, gt_2, events


# 读取事件、时间戳
def load_data_single(file_name):
    with h5py.File(file_name, 'r') as h5:
        timestamps = h5['timestamps'][:]
        events = {
            't': h5['events']['t'][:],
            'x': h5['events']['x'][:],
            'y': h5['events']['y'][:],
            'p': h5['events']['p'][:],
        }
    return timestamps, events

# 读取前后帧
import os
import re
import cv2
import random

def load_rgb_pair(dataset_path, split, seq, h5_name):
    """
    h5_name: '00000004.h5'
    """
    base_idx = int(re.search(r'\d+', h5_name).group())

    rgb_dir = os.path.join(
        dataset_path,
        split,
        f"{split}_orig",
        seq
    )

    img0 = cv2.imread(
        os.path.join(rgb_dir, f"{base_idx:08d}.png")
    )
    img1 = cv2.imread(
        os.path.join(rgb_dir, f"{base_idx + 4:08d}.png")
    )

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    return img0, img1

# 读取中间帧gt
def load_gt_frames(dataset_path, split, seq, h5_name):
    base_idx = int(re.search(r'\d+', h5_name).group())

    rgb_dir = os.path.join(
        dataset_path, split, f"{split}_orig", seq
    )

    gt_paths = [
        os.path.join(rgb_dir, f"{base_idx + 1:08d}.png"),
        os.path.join(rgb_dir, f"{base_idx + 2:08d}.png"),
        os.path.join(rgb_dir, f"{base_idx + 3:08d}.png"),
    ]

    gt_path = random.choice(gt_paths)
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    return gt




if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_height', type=int, default=256, help='cropped image size height')
    parser.add_argument('--crop_width', type=int, default=256, help='cropped image size width')
    parser.add_argument('--dataset_path', default='E:\\', type=str, help='root path of dataset')
    parser.add_argument('--is_load_gt', action='store_true')
    parser.add_argument('--is_rand_flip', action='store_true')
    parser.add_argument('--ev_pack', default='right', type=str)
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    dataset = Dataloader_val(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    x, gt0, gt1, gt2, ev = next(iter(dataloader))
    print("input:", x.shape)
    print("gt:", gt0.shape, gt1.shape, gt2.shape)
    print("event:", ev.shape)

    # data = StDataLoader(args, 'train')
    # dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=2, shuffle=True,
    #                                          num_workers=2, pin_memory=True, drop_last=True)
    #
    # tbar = tqdm(dataloader)
    # for idx, (blurry, sharp_gt_3,
    #           esis, mask,
    #           bag_name, image_iter) in enumerate(tbar):
    #
    #     print(esis, esis.shape)
    #     print(mask, mask.shape)
    #
    #     break
