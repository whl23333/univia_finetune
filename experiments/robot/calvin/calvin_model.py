import math
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import time

import torch.nn as nn

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from experiments.robot.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from prismatic.models.policy.transformer_utils import MAPBlock



class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size=5, hidden_dim=512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim//64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim//64)


        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens , init_embed=visual_embed))
        
        return action



class ActionDecoderWrapper(nn.Module):
    def __init__(self, window_size=12):
        super().__init__()
        self.net = ActionDecoder(window_size)
        self.action_chunk_size = window_size

    def reset(self):
        pass

    def forward(self, latent_actions, visual_embed):
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), 
                               visual_embed.to(torch.float)).reshape(-1, self.action_chunk_size, 7)
        pred_action = np.array(pred_action.tolist())[0]

        return pred_action


class WrappedModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load action decoder
        self.action_decoder = ActionDecoderWrapper(cfg.window_size)
        self.action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))

        # Load VLA
        self.vla = get_model(cfg)



class WrappedCalvinEvaluation(CalvinBaseModel):
    def __init__(self, cfg, wrapped_model):
        super().__init__()
        self.cfg = cfg

        self.model = wrapped_model
        # [OpenVLA] Get Hugging Face processor
        self.processor = get_processor(cfg)
        self.prev_hist_action = ['']

        

    def reset(self,):
        """
        This is called
        """ 
        self.model.action_decoder.reset()
        self.prev_hist_action = ['']


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        img = obs["rgb_obs"]['rgb_static']

        observation = {
            "full_image": img,
            "state": [],
        }

        # Register forward hooks to capture internal features during the model's forward pass
        captures = {}

        def _save_patch_features(module, inputs, output):
            captures["patch_features"] = output

        def _save_projected(module, inputs, output):
            captures["projected_patch_embeddings"] = output

        def _save_input_embeddings(module, inputs, output):
            captures["input_embeddings"] = output

        def _save_action_embedings(module, inputs, output):
            captures['action_embeddings'] = output

        # Attach hooks
        # patch_features
        vh = self.model.vla.vision_backbone.register_forward_hook(_save_patch_features)
        # projected_patch_embeddings
        pj = self.model.vla.projector.register_forward_hook(_save_projected)
        emb_module = self.model.vla.get_input_embeddings()
        # input_embeddings
        eh = emb_module.register_forward_hook(_save_input_embeddings)
        # action_embeddings
        ah = self.model.action_decoder.net.latent_action_pool.register_forward_hook(_save_action_embedings)

        try:
            # Query model to get latent action (this triggers the hooks once per step)
            latent_action, visual_embed, generated_ids = get_latent_action(
                self.cfg,
                self.model.vla,
                observation,
                instruction,
                processor=self.processor,
            )
        finally:
            # Always remove hooks to avoid memory leaks or duplicate captures
            vh.remove()
            pj.remove()
            eh.remove()

        # Get decoded action
        try:
            action = self.model.action_decoder(latent_action, visual_embed)
        finally:
            ah.remove()
        embeddings = {}
        # embeddings['latent_action'] = latent_action
        # embeddings['visual_embed'] = visual_embed
        # embeddings['generated_ids'] = generated_ids
        # Add captured internal features if available
        # if "patch_features" in captures:
        #     embeddings["patch_features"] = captures["patch_features"]
        if "projected_patch_embeddings" in captures:
            embeddings["projected_patch_embeddings"] = captures["projected_patch_embeddings"]
        # if "input_embeddings" in captures:
        #     embeddings["input_embeddings"] = captures["input_embeddings"]
        if "action_embeddings" in captures:
            embeddings["action_embeddings"] = captures["action_embeddings"]

        return {"action": action, "embeddings": embeddings}
