import math
from os import listdir, makedirs, path
from random import choices, randint
from typing import Any, Callable, Dict, List, Tuple, Optional

import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
import torchvision.transforms as transforms
from dataclasses import dataclass
from pathlib import Path
import os
from glob import glob

from prismatic.util import set_global_seed
from prismatic.util.data_utils import CollatorForLatentAction, CollatorForMultiViewVideo
from prismatic.vla.datasets import RLDSDataset, EpisodicRLDSDataset, RLDSBatchTransformVideo
import json
from torchvision.transforms.v2 import Resize, InterpolationMode
import numpy as np

def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 64,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = 0    # For RLDS parallelism
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # shuffle unspecified for iteratable datasets
        # self.train_shuffle = train_shuffle
        # self.val_shuffle = val_shuffle

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
            return DataLoader(
                self.train_dataset,
                sampler=self.train_sampler,
                batch_size=self.batch_size,
                # shuffle=shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
            shuffle = True  # enable shuffle for map-style datasets
            return DataLoader(
                self.train_dataset,
                sampler=self.train_sampler,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
            return DataLoader(
                self.val_dataset,
                sampler=self.val_sampler,
                batch_size=self.val_batch_size,
                # shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
            return DataLoader(
                self.val_dataset,
                sampler=self.val_sampler,
                batch_size=self.val_batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
            return DataLoader(
                self.test_dataset,
                sampler=self.test_sampler,
                batch_size=self.val_batch_size,
                # shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
            return DataLoader(
                self.test_dataset,
                sampler=self.test_sampler,
                batch_size=self.val_batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn
            )



from PIL import Image
import random

@dataclass
class random_crop_resize():
    def __init__(
        self,
        target_size=224
    ):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image):
        width, height = image.size

        if width < height:
            crop_size = width
        else:
            crop_size = height

        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

        image_cropped = image.crop((left, top, left + crop_size, top + crop_size))
        image_resized = image_cropped.resize((self.target_size, self.target_size), Image.BILINEAR)
        image_resized = self.to_tensor(image_resized)
        
        return image_resized



class LightningOpenX(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
            self,
            data_root: str,
            data_mix: str,
            batch_size:int = 16,
            resolution: int = 256,
            num_frames: int = 16,
            episodic: bool = False,
            shuffle_buffer_size: int = 100_000,
            image_aug:bool = False,
            # Custom frame-pair dataset options
            use_custom_frames: bool = False,
            custom_frames_root: Optional[str] = None,
            frame_interval: int = 10,
            num_workers_custom: int = 12,
            **kwargs
    ) -> None:
        super(LightningOpenX, self).__init__(**kwargs)

        self.data_root_dir = data_root
        self.data_mix = data_mix

        self.batch_size = batch_size
        self.resolution = (resolution, resolution)
        self.num_frames = num_frames

        self.episodic = episodic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_aug = image_aug

        # Custom frames configuration
        self.use_custom_frames = use_custom_frames
        self.custom_frames_root = custom_frames_root
        self.frame_interval = frame_interval
        self.num_workers_custom = num_workers_custom

        self.num_workers = 0    # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        self.worker_init_fn = set_global_seed(42, get_worker_init_fn=True)

        self.batch_transform = RLDSBatchTransformVideo(
            image_transform=transforms.ToTensor() 
        )
        self.collate_fn = CollatorForLatentAction()

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        # When using custom frames, build a map-style dataset and enable workers/shuffle
        if self.use_custom_frames:
            assert self.custom_frames_root is not None, "custom_frames_root must be set when use_custom_frames=True"
            # Override workers for map-style dataset
            self.num_workers = self.num_workers_custom
            
            if stage == "fit":
                self.train_dataset = NpzDataset_for_MotoGPT_Video_Multiview(
                    split='train',
                    skip_frame=self.frame_interval,
                    sequence_length=1,
                    npz_dir=self.custom_frames_root,
                    rgb_shape_static=self.resolution,
                    rgb_shape_gripper=self.resolution,
                    rgb_preprocessor=Resize(self.resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
                )

                self.val_dataset = NpzDataset_for_MotoGPT_Video_Multiview(
                    split='val',
                    skip_frame=self.frame_interval,
                    sequence_length=1,
                    npz_dir=self.custom_frames_root,
                    rgb_shape_static=self.resolution,
                    rgb_shape_gripper=self.resolution,
                    rgb_preprocessor=Resize(self.resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
                )
            elif stage == "test":
                self.test_dataset = NpzDataset_for_MotoGPT_Video_Multiview(
                    split='val',
                    skip_frame=self.frame_interval,
                    sequence_length=1,
                    npz_dir=self.custom_frames_root,
                    rgb_shape_static=self.resolution,
                    rgb_shape_gripper=self.resolution,
                    rgb_preprocessor=Resize((self.resolution, self.resolution), interpolation=InterpolationMode.BICUBIC, antialias=True),
                )
            else:
                raise ValueError(f"Invalid stage: {stage}")
            return

        cls = RLDSDataset if not self.episodic else EpisodicRLDSDataset
        if stage == "fit":
            self.train_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=self.image_aug,
                training_phase='lam',
            )
            self.val_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=False,
                image_aug=False,
                training_phase='lam',
            )
        elif stage == "test":
            self.test_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=False,
                training_phase='lam',
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")



class FramePairDataset(Dataset):
    """
    Minimal map-style dataset that yields (initial, target) frame pairs at a fixed interval.

    Directory layout assumptions (one of):
      - root/
          episode_*/
              frame_*.{jpg,png,...}
      - root/ (flat)
          frame_*.{jpg,png,...}

    It produces dictionaries compatible with CollatorForLatentAction:
      {initial_pixel_values, target_pixel_values, task_instruction, action, dataset_name}
    """

    IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(
        self,
        root_dir: str,
        interval: int,
        image_transform: Callable,
        default_instruction: str = "",
        dataset_name: str = "custom_frames",
    ) -> None:
        super().__init__()
        self.root = Path(root_dir)
        self.interval = int(interval)
        self.image_transform = image_transform
        self.default_instruction = default_instruction
        self.dataset_name = dataset_name

        assert self.root.exists(), f"Root directory not found: {self.root}"

        # Build list of frame pairs within each episode directory (or flat directory)
        episode_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        if len(episode_dirs) == 0:
            # flat directory
            frames = self._sorted_images(self.root)
            self.pairs = self._make_pairs(frames)
        else:
            pairs: List[Tuple[Path, Path]] = []
            for ep in episode_dirs:
                frames = self._sorted_images(ep)
                pairs.extend(self._make_pairs(frames))
            self.pairs = pairs

        if len(self.pairs) == 0:
            raise ValueError(f"No valid frame pairs found in {self.root} with interval {self.interval}")

    def _sorted_images(self, directory: Path) -> List[Path]:
        files: List[Path] = []
        for pat in self.IMG_EXTS:
            files.extend(sorted(directory.glob(pat)))
        # sort by name to preserve chronological order
        files = sorted(files)
        return files

    def _make_pairs(self, frames: List[Path]) -> List[Tuple[Path, Path]]:
        k = self.interval
        if len(frames) <= k:
            return []
        return [(frames[i], frames[i + k]) for i in range(0, len(frames) - k)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src, dst = self.pairs[idx]

        with Image.open(src) as im0:
            im0 = im0.convert("RGB")
            initial = self.image_transform(im0)
        with Image.open(dst) as im1:
            im1 = im1.convert("RGB")
            target = self.image_transform(im1)

        # Action is unused in Stage-2; provide a dummy 1D vector for collation
        action = torch.zeros(1, dtype=torch.float32).numpy()

        return dict(
            initial_pixel_values=initial,
            target_pixel_values=target,
            task_instruction=self.default_instruction,
            action=action,
            dataset_name=self.dataset_name,
        )


class NpzDataset_for_MotoGPT_Video(Dataset):
    def __init__(
        self, split, skip_frame, # split: train/val, skip_frame: 5
        sequence_length, # 1
        npz_dir=None, rgb_shape=(224, 224), # npz_dir: "/group/ycyang/yyang-infobai/task_ABC_D/", rgb_shape: [200, 200]
        rgb_preprocessor=None, max_skip_frame=None, npz_metadata_path=None, *args, **kwargs): # 'do_extract_future_frames': True, 'do_extract_action': False

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        if split == 'train':
            split = 'training'
        elif split == 'val':
            split = 'validation'
        else:
            raise NotImplementedError

        self.npz_dir = os.path.join(npz_dir, split)
        self.rgb_preprocessor = rgb_preprocessor

        if npz_metadata_path is None:
            npz_metadata_path = os.path.join(self.npz_dir, 'npz_metadata.json')
        else:
            print(f"specified npz_metadata_path: {npz_metadata_path}")
        
        with open(npz_metadata_path) as f:
            npz_metadata = json.load(f)

        self.npz_metadata = npz_metadata
        self.dataset_len = len(npz_metadata) - skip_frame

    def get_npz_path(self, npz_basename):
        return os.path.join(self.npz_dir, npz_basename)

    def extract_frames(self, npz_basename, delta_t, 
                       rgb_initial, rgb_future, latent_mask):
        
        def _extract_frame(npz_idx):
            npz_path = self.get_npz_path(f"episode_{str(npz_idx).zfill(7)}.npz")
            try:
                frame = Image.fromarray(np.load(npz_path)['rgb_static']).convert("RGB")
            except Exception as e:
                raise e

            frame = np.array(frame)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        start_npz_path = self.get_npz_path(npz_basename)
        start_npz_idx = int(npz_basename.split("_")[-1].split(".")[0])
        rgb_initial[0] = _extract_frame(start_npz_idx)

        for i in range(self.sequence_length):
            next_npz_idx = start_npz_idx+(i+1)*delta_t
            try:
                rgb_future[i] = _extract_frame(next_npz_idx)
                latent_mask[i] = 1
            except:
                break

    
    def obtain_item(self, idx, delta_t=None):
        npz_basename = self.npz_metadata[idx]
        npz_idx = int(npz_basename.split("_")[-1].split(".")[0])

        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                delta_t = random.randint(self.skip_frame, self.max_skip_frame)

        # dummy features
        rgb_initial = self.dummy_rgb_initial.clone()
        rgb_future = self.dummy_rgb_future.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        self.extract_frames(
            npz_basename=npz_basename,
            delta_t=delta_t,
            rgb_initial=rgb_initial, 
            rgb_future=rgb_future, 
            latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        # Adapt outputs to LightningOpenX CollatorForLatentAction expectations
        # Use the first initial frame and the first future frame (sequence_length typically 1)
        initial = rgb_initial[0]
        target = rgb_future[0]

        # Ensure float tensor in [0,1]
        if initial.dtype == torch.uint8:
            initial = initial.float() / 255.0
        if target.dtype == torch.uint8:
            target = target.float() / 255.0

        return {
            "initial_pixel_values": initial,
            "target_pixel_values": target,
            "task_instruction": "",  # no instruction for NPZ dataset
            "action": torch.zeros(1, dtype=torch.float32).numpy(),  # dummy action
            "dataset_name": "npz_singleview",

            # keep original auxiliary fields if needed downstream
            "latent_mask": latent_mask,
            "idx": idx,
            "delta_t": delta_t,
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                return self.obtain_item(idx)
            except Exception as e:
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len

class NpzDataset_for_MotoGPT_Video_Multiview(Dataset):
    def __init__(
        self, split, skip_frame, # split: train/val, skip_frame: 5
        sequence_length, # 1
        npz_dir=None, rgb_shape_static=(224, 224), rgb_shape_gripper=(224, 224), # npz_dir: "/group/ycyang/yyang-infobai/task_ABC_D/", rgb_shape: [200, 200]
        rgb_preprocessor=None, max_skip_frame=None, npz_metadata_path=None, *args, **kwargs): # 'do_extract_future_frames': True, 'do_extract_action': False

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.dummy_rgb_initial_static = torch.zeros(1, 3, rgb_shape_static[0], rgb_shape_static[1], dtype=torch.uint8)
        self.dummy_rgb_future_static = torch.zeros(sequence_length, 3, rgb_shape_static[0], rgb_shape_static[1], dtype=torch.uint8)
        self.dummy_rgb_initial_gripper = torch.zeros(1, 3, rgb_shape_gripper[0], rgb_shape_gripper[1], dtype=torch.uint8)
        self.dummy_rgb_future_gripper = torch.zeros(sequence_length, 3, rgb_shape_gripper[0], rgb_shape_gripper[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        if split == 'train':
            split = 'training'
        elif split == 'val':
            split = 'validation'
        else:
            raise NotImplementedError

        self.npz_dir = os.path.join(npz_dir, split)
        self.rgb_preprocessor = rgb_preprocessor

        if npz_metadata_path is None:
            npz_metadata_path = os.path.join(self.npz_dir, 'npz_metadata.json')
        else:
            print(f"specified npz_metadata_path: {npz_metadata_path}")
        
        with open(npz_metadata_path) as f:
            npz_metadata = json.load(f)

        self.npz_metadata = npz_metadata
        self.dataset_len = len(npz_metadata) - skip_frame

    def get_npz_path(self, npz_basename):
        return os.path.join(self.npz_dir, npz_basename)

    def extract_frames(self, npz_basename, delta_t, 
                       rgb_initial_static, rgb_initial_gripper, rgb_future_static, rgb_future_gripper, latent_mask):
        
        def _extract_frame(npz_idx):
            npz_path = self.get_npz_path(f"episode_{str(npz_idx).zfill(7)}.npz")
            try:
                frame_static = Image.fromarray(np.load(npz_path)['rgb_static']).convert("RGB")
                frame_gripper = Image.fromarray(np.load(npz_path)['rgb_gripper']).convert("RGB")
            except Exception as e:
                raise e

            frame_static = np.array(frame_static)
            frame_gripper = np.array(frame_gripper)
            frame_static = torch.from_numpy(rearrange(frame_static, 'h w c -> c h w'))
            frame_gripper = torch.from_numpy(rearrange(frame_gripper, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame_static  = self.rgb_preprocessor(frame_static)
                frame_gripper  = self.rgb_preprocessor(frame_gripper)
            return frame_static, frame_gripper

        start_npz_path = self.get_npz_path(npz_basename)
        start_npz_idx = int(npz_basename.split("_")[-1].split(".")[0])
        rgb_initial_static[0], rgb_initial_gripper[0] = _extract_frame(start_npz_idx)

        for i in range(self.sequence_length):
            next_npz_idx = start_npz_idx+(i+1)*delta_t
            try:
                rgb_future_static[i], rgb_future_gripper[i] = _extract_frame(next_npz_idx)
                latent_mask[i] = 1
            except:
                break

    
    def obtain_item(self, idx, delta_t=None):
        npz_basename = self.npz_metadata[idx]
        npz_idx = int(npz_basename.split("_")[-1].split(".")[0])

        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                delta_t = random.randint(self.skip_frame, self.max_skip_frame)

        # dummy features
        rgb_initial_static = self.dummy_rgb_initial_static.clone()
        rgb_future_static = self.dummy_rgb_future_static.clone()
        rgb_initial_gripper = self.dummy_rgb_initial_gripper.clone()
        rgb_future_gripper = self.dummy_rgb_future_gripper.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        self.extract_frames(
            npz_basename=npz_basename,
            delta_t=delta_t,
            rgb_initial_static=rgb_initial_static,
            rgb_initial_gripper=rgb_initial_gripper,
            rgb_future_static=rgb_future_static,
            rgb_future_gripper=rgb_future_gripper,
            latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        # Adapt outputs to LightningOpenX multiview expectations
        # Static view becomes initial/target; gripper view provided as separate keys
        initial_static = rgb_initial_static[0]
        target_static = rgb_future_static[0]
        initial_gripper = rgb_initial_gripper[0]
        target_gripper = rgb_future_gripper[0]

        # Ensure float tensors in [0,1]
        for t in (initial_static, target_static, initial_gripper, target_gripper):
            if t.dtype == torch.uint8:
                # convert in-place-safe via creating new tensor
                pass
        if initial_static.dtype == torch.uint8:
            initial_static = initial_static.float() / 255.0
        if target_static.dtype == torch.uint8:
            target_static = target_static.float() / 255.0
        if initial_gripper.dtype == torch.uint8:
            initial_gripper = initial_gripper.float() / 255.0
        if target_gripper.dtype == torch.uint8:
            target_gripper = target_gripper.float() / 255.0

        return {
            "initial_pixel_values": initial_static,
            "target_pixel_values": target_static,
            "initial_pixel_values_gripper": initial_gripper,
            "target_pixel_values_gripper": target_gripper,
            "task_instruction": "",
            "action": torch.zeros(1, dtype=torch.float32).numpy(),
            "dataset_name": "npz_multiview",

            # auxiliary
            "latent_mask": latent_mask,
            "idx": idx,
            "delta_t": delta_t,
        }

    
    def __getitem__(self, idx):
        # while True:
        #     try:
        #         return self.obtain_item(idx)
        #     except Exception as e:
        #         idx = random.randint(0, len(self)-1)
        try:
            return self.obtain_item(idx)
        except Exception as e:
            raise e
            

    def __len__(self):
        return self.dataset_len