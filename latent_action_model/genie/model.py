from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
from genie.modules.lam import ControllableDINOLatentActionModelMultiView



class DINO_LAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
            stage_two_ckpt: str = None,
    ) -> None:
        super(DINO_LAM, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        lam = UncontrolledDINOLatentActionModel if stage == 'stage-1' else ControllableDINOLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            for key in lam_ckpt.keys():
                if 'vq' in key or 'action_latent' in key:
                    stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage1_ckpt, strict=False)
        
        if stage_two_ckpt and path.exists(stage_two_ckpt) and stage == 'stage-2':
            print("Loading stage-2 checkpoint:", stage_two_ckpt)
            lam_ckpt = torch.load(stage_two_ckpt)['state_dict']
            stage2_ckpt = {}
            for key in lam_ckpt.keys():
                stage2_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage2_ckpt, strict=False) # load all weights in stage-2 finetuning


        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss
        
        # Optimize uncontrollable queries in stage-2 (the codebook is frozen though)
        if "z_q_uncontrol" in outputs.keys():
            q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
            commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
            loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )

        if "indices_uncontrol" in outputs.keys():
            unique, counts = torch.unique(outputs["indices_uncontrol"], return_counts=True)
            index_counts = torch.zeros(32, dtype=torch.long).cuda()
            index_counts[unique] = counts
            uncontrol_code_usage = (index_counts != 0).float().mean()

            loss_logs = (
                ("mse_loss", mse_loss),
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("q_loss_uncontrol", q_loss_uncontrol),
                ("commit_loss_uncontrol", commit_loss_uncontrol),
                ("code_usage", code_usage),
                ("code_usage_uncontrol", uncontrol_code_usage),
            )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim



class DINO_LAM_MultiView(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
            stage_two_ckpt: str = None,
    ) -> None:
        super(DINO_LAM_MultiView, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        # Multi-view uses the dedicated multi-view LAM in stage-2, and single-view in stage-1
        if stage == 'stage-1':
            lam_cls = UncontrolledDINOLatentActionModel
        else:
            lam_cls = ControllableDINOLatentActionModelMultiView

        self.lam = lam_cls(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            for key in lam_ckpt.keys():
                if 'vq' in key or 'action_latent' in key:
                    stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage1_ckpt, strict=False)
        
        if stage_two_ckpt and path.exists(stage_two_ckpt) and stage == 'stage-2':
            print("Loading stage-2 checkpoint:", stage_two_ckpt)
            lam_ckpt = torch.load(stage_two_ckpt)['state_dict']
            stage2_ckpt = {}
            for key in lam_ckpt.keys():
                stage2_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage2_ckpt, strict=False) # load all weights in stage-2 finetuning


        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # Forward multi-view LAM
        outputs = self.lam(batch)

        # Reconstruction targets and predictions for two views
        gt_view1 = outputs["target_view1"]
        gt_view2 = outputs["target_view2"]
        pred_view1 = outputs["recon_view1"]
        pred_view2 = outputs["recon_view2"]

        # Reconstruction MSE per view, then mean
        mse_view1 = ((gt_view1 - pred_view1) ** 2).mean()
        mse_view2 = ((gt_view2 - pred_view2) ** 2).mean()
        mse_loss = 0.5 * (mse_view1 + mse_view2)

        # Codebook commit + quantization losses for fused controllable branch (shared across views)
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss

        # Uncontrollable branch losses per view (codebook frozen, but queries optimized)
        q_loss_uncontrol_v1 = ((outputs["emb_uncontrol_view1"].detach() - outputs["z_uncontrol_view1"]) ** 2).mean()
        commit_loss_uncontrol_v1 = ((outputs["emb_uncontrol_view1"] - outputs["z_uncontrol_view1"].detach()) ** 2).mean()

        q_loss_uncontrol_v2 = ((outputs["emb_uncontrol_view2"].detach() - outputs["z_uncontrol_view2"]) ** 2).mean()
        commit_loss_uncontrol_v2 = ((outputs["emb_uncontrol_view2"] - outputs["z_uncontrol_view2"].detach()) ** 2).mean()

        # Average uncontrollable losses across views
        q_loss_uncontrol = 0.5 * (q_loss_uncontrol_v1 + q_loss_uncontrol_v2)
        commit_loss_uncontrol = 0.5 * (commit_loss_uncontrol_v1 + commit_loss_uncontrol_v2)
        loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        # Code usage for controllable fused codebook (indices from fused z_3d quantization)
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        # Code usage for uncontrollable per-view codebook (aggregate both views)
        unique_v1, counts_v1 = torch.unique(outputs["indices_uncontrol_view1"], return_counts=True)
        unique_v2, counts_v2 = torch.unique(outputs["indices_uncontrol_view2"], return_counts=True)
        index_counts_unctrl = torch.zeros(32, dtype=torch.long, device=counts.device)
        index_counts_unctrl[unique_v1] += counts_v1
        index_counts_unctrl[unique_v2] += counts_v2
        uncontrol_code_usage = (index_counts_unctrl != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("mse_view1", mse_view1),
            ("mse_view2", mse_view2),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("q_loss_uncontrol", q_loss_uncontrol),
            ("commit_loss_uncontrol", commit_loss_uncontrol),
            ("code_usage", code_usage),
            ("code_usage_uncontrol", uncontrol_code_usage),
        )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_epoch_end(self):
        # Reset usage of uncontrollable codebook if present
        if hasattr(self.lam, 'vq'):
            self.lam.vq.random_restart()
            self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
