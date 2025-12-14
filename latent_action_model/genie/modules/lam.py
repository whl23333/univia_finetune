from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from latent_action_model.genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(UncontrolledDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained('./t5-base')
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('./t5-base')

    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask) 
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat([torch.ones((B, self.num_codes + (H // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)

        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches, lang_embed.unsqueeze(1), attention_mask)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device




class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # local repo
        self.dino_encoder = torch.hub.load('/data/250010208/whl/local_repos/dinov2', 'dinov2_vitb14_reg', source='local')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
                                          video_patches],
                                          dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2]:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device



class ControllableDINOLatentActionModelMultiView(nn.Module):
    """
    Latent action VQ-VAE. Multi view
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModelMultiView, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # local repo
        self.dino_encoder = torch.hub.load('/data/250010208/whl/local_repos/dinov2', 'dinov2_vitb14_reg', source='local')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)

        self.cross_attn = SymmetricCrossAttention(
            dim=self.latent_dim,
            fuse='mean'
        )
    
    def single_view_encode(self, videos: Tensor) -> Dict:
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        with torch.no_grad():
            dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        return {
            "patches": dion_features,
            "z_action": z_action,
            # "z_q": z_q
            # "z": z,
            # "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            # "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }
    
    def encode_3d(self, z1, z2):
        B, L = z1.shape[:2]
        z1 = z1.reshape(B*L, self.num_codes, self.latent_dim)
        z2 = z2.reshape(B*L, self.num_codes, self.latent_dim)
        z_3d = self.cross_attn(z1, z2)
        z_q_3d, z_3d, emb_3d, indices_3d = self.vq_action(z_3d)
        z_q_3d = z_q_3d.reshape(B, L, self.num_codes, self.latent_dim)

        return {
            "z_q": z_q_3d,
            "z": z_3d,
            "emb": emb_3d,
            "indices": indices_3d
        }

        


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        # outputs = self.vq_encode(batch["videos"]) 
        outputs_view1 = self.single_view_encode(batch["videos"])
        outputs_view2 = self.single_view_encode(batch["videos_view2"])
        z_action_view1 = outputs_view1['z_action']
        z_action_view2 = outputs_view2['z_action']
        outputs_3d = self.encode_3d(z_action_view1, z_action_view2)


        # video_patches = self.patch_up(outputs["patches"][:, :-1])
        video_patches_view1 = self.patch_up(outputs_view1["patches"][:, :-1])
        video_patches_view2 = self.patch_up(outputs_view2["patches"][:, :-1])

        # Decode
        # video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
        #                                   self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
        #                                   video_patches],
        #                                   dim=2)
        video_action_patches_view1 = torch.cat([self.action_up(outputs_3d["z_q"]),
                                                self.action_up_uncontrol(outputs_view1["z_q_uncontrol"]),
                                                video_patches_view1],
                                                dim=2)
        video_action_patches_view2 = torch.cat([self.action_up(outputs_3d["z_q"]),
                                                self.action_up_uncontrol(outputs_view2["z_q_uncontrol"]),
                                                video_patches_view2],
                                                dim=2)
        video_recon_view1 = self.decoder(video_action_patches_view1)
        video_recon_view2 = self.decoder(video_action_patches_view2)
        # video_recon = self.decoder(video_action_patches)
        video_recon_view1 = video_recon_view1[:, :, -video_patches_view1.shape[2]:]
        video_recon_view2 = video_recon_view2[:, :, -video_patches_view2.shape[2]:]
        # video_recon = video_recon[:, :, -video_patches.shape[2]:]

        outputs = {
            "patches_view1": outputs_view1["patches"],
            "patches_view2": outputs_view2['patches'],
            'z_q': outputs_3d["z_q"],
            'z': outputs_3d['z'],
            'emb': outputs_3d['emb'],
            "z_q_uncontrol_view1": outputs_view1["z_q_uncontrol"],
            "z_q_uncontrol_view2": outputs_view2["z_q_uncontrol"],
            "z_uncontrol_view1": outputs_view1["z_uncontrol"],
            "z_uncontrol_view2": outputs_view2["z_uncontrol"],
            "emb_uncontrol_view1": outputs_view1["emb_uncontrol"],
            "emb_uncontrol_view2": outputs_view2["emb_uncontrol"],
            "indices": outputs_3d["indices"],
            "indices_uncontrol_view1": outputs_view1["indices_uncontrol"],
            "indices_uncontrol_view2": outputs_view2["indices_uncontrol"],
            "recon_view1": video_recon_view1,
            "recon_view2": video_recon_view2,
            "target_view1": outputs_view1["patches"][:, [-1]],
            "target_view2": outputs_view2["patches"][:, [-1]]
        } 

        # outputs.update(
        #     {
        #         "recon": video_recon,
        #         "target": outputs["patches"][:, [-1]]
        #     }
        # )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class SymmetricCrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, batch_first=True, fuse='mean'):
        """
        fuse: 'mean' | 'concat' | 'gate'
          - mean: 直接 (attn_ab + attn_ba)/2
          - concat: 拼接后线性还原
          - gate: 学习一个标量或向量做加权
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, dropout=dropout, batch_first=batch_first)
        self.fuse = fuse
        if fuse == 'concat':
            self.proj = nn.Linear(2*dim, dim)
        elif fuse == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, a, b, mask_a=None, mask_b=None):
        """
        a, b: (B, T, D)
        mask_*: (B, T) 可选，True 为需要忽略的位置（与 MultiheadAttention 的 key_padding_mask 语义一致）
        """
        # A 作为 query，B 作为 key,value
        attn_ab, _ = self.attn(query=a, key=b, value=b, key_padding_mask=mask_b)
        # B 作为 query，A 作为 key,value
        attn_ba, _ = self.attn(query=b, key=a, value=a, key_padding_mask=mask_a)

        if self.fuse == 'mean':
            y = 0.5 * (attn_ab + attn_ba)
        elif self.fuse == 'concat':
            y = torch.cat([attn_ab, attn_ba], dim=-1)
            y = self.proj(y)
        elif self.fuse == 'gate':
            g = self.gate(torch.cat([attn_ab, attn_ba], dim=-1))   # (B,T,D) 逐通道门控
            y = g * attn_ab + (1 - g) * attn_ba
        else:
            raise ValueError(f"Unknown fuse mode {self.fuse}")

        # 加一个残差稳定训练（可选）
        y = self.ln_out(y + 0.5 * (a + b))
        return y