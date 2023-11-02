from torch.nn import DataParallel
from torch.utils.data import DataLoader
from utils.dataset import TestDatasetFolder
from backbones.iresnet import iresnet100
import torch


def compute_correct_batch_size(length=1251416):
    max = 0
    for bs in range(1, 2048):
        if length % bs == 0:
            max = bs
    return max


def load_model(path, embedding_size, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    backbone = iresnet100(dropout=0.4, num_features=embedding_size, use_se=False)
    backbone.load_state_dict(torch.load(path, map_location=torch.device(device)))
    backbone = DataParallel(module=backbone, device_ids=[device_id])
    backbone.eval()
    return backbone, device


def get_dataloader(real_path, synth_path=None, local_rank=0):
    dataset = TestDatasetFolder(root_dir=synth_path, local_rank=local_rank, root2=real_path)
    batch_size = compute_correct_batch_size(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=8, drop_last=False)
    return dataset, dataloader, batch_size
