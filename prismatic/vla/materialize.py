"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSBatchTransformLatentAction, RLDSDataset
from prismatic.vla.datasets.calvin_dataset import DiskCalvinDataset, DiskCalvinIterableDataset
from transformers import AutoProcessor


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator


def get_latent_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    image_transform_lam: ImageTransform,
    latent_action_tokenizer: PreTrainedTokenizerBase, 
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    use_calvin: bool = False,
    pretrained_vla_path: str = "",
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # action_tokenizer = ActionTokenizer(tokenizer)

    batch_transform = RLDSBatchTransformLatentAction(
        action_tokenizer=latent_action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        image_transform_lam=image_transform_lam,
        prompt_builder_fn=prompt_builder_fn
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )


    # Build RLDS Iterable Dataset
    if use_calvin:
        processor = AutoProcessor.from_pretrained(pretrained_vla_path, trust_remote_code=True)  # Replace with actual path
        disk_dataset = DiskCalvinDataset(
            datasets_dir=data_root_dir / "training",  # Replace with actual path
            image_fn=None,
            text_fn=None,
            window_size=10,
            traj_cons=False,
            text_aug=False,
            dif_ws=False,
            min_window_size=10,
            max_window_size=10,
            partial_data=False,
            sampling_step=1,
            action_tokenizer = None,
            base_tokenizer = None,
            image_transform = processor.image_processor.apply_transform,
            prompt_builder_fn = None,
        )
        dataset = DiskCalvinIterableDataset(
            base=disk_dataset,
            resize_resolution=default_image_resolution[1:],
            shuffle=True,
            batch_transform=batch_transform,
        )
        return dataset, tokenizer, collator
    else:
        cls = RLDSDataset if not episodic else EpisodicRLDSDataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
            training_phase='pre-training',
        )

        return dataset, tokenizer, collator