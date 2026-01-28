import torch
from torch.utils.data import DataLoader

from train import train, Model, Dataloader_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    epoch = 1
    batch_size = 1
    local_rank = 0
    world_size = 1
    crop_height = 128
    crop_width = 128
    dataset_path = "E:\\"

args = Args()

model = Model(args.local_rank)

# 把 args.world_size 设置为 1，不用分布式
args.world_size = 1
args.epoch = 1
args.batch_size = 1

# 创建 dataloader
dataset = Dataloader_val(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# 随机取一批数据跑一下 model.update
rgb_pair, gt_0, gt_1, gt_2, voxel_pair = next(iter(dataloader))
rgb_pair = rgb_pair.to(device)
voxel_pair = voxel_pair.to(device)
gt = gt_1.to(device)

model.update(rgb_pair, voxel_pair, gt, learning_rate=1e-4, training=True)
print("train pipeline ran successfully!")
