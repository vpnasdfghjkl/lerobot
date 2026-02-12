from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed‑Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self,
                query: torch.Tensor, # (B, N, D)
                k: torch.Tensor, # (B, M, D)
                v: torch.Tensor, # (B, M, D)
                ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # residual + LN
        x = self.attn_norm(query + attn_out)

        # FFN + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        # Initialization trick: 
        # Initialize weights to be small and bias to be large positive 
        # so that sigmoid(bias) -> 1, meaning the gate initially prefers the original input (x1).
        # This preserves the pretrained behavior of the base model at the start of finetuning.
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj.bias, 5.0) # Sigmoid(5.0) approx 0.993

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: Original features
        # x2: Memory features
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2], dim=-1)
            )
        )

        fused = scale * x1 + (1 - scale) * x2
        return fused


class CogMemBank(nn.Module):
    def __init__(self,
                 dataloader_type: str = 'stream', # 'stream' (inference) or 'group' (training with sequences)
                 group_size: int = 1,
                 token_size: int = 2048,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'fifo', # 'fifo' or 'tome'
                 update_fused: bool = False,
                 ):
        super().__init__()
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(self.token_size)
            for _ in range(self.retrieval_layers)
        ])

        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = GateFusion(self.token_size)

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4)
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        # bank[episode_id] = [(timestep, feat[N,D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]

        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())

        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)

        bank[idx_max] = (timestep_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
            self,
            episode_id,
            feat: torch.Tensor,
            timestep: Optional[torch.Tensor]):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                # Default to FIFO if unknown
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]

    def forward(
        self,
        tokens: torch.Tensor, # [B, N, D]
        episode_ids: Optional[np.ndarray | list] = None,
        timesteps: Optional[np.ndarray | list | torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a batch of tokens, retrieving from memory and updating memory.
        If episode_ids/timesteps are None, assumes a single stream (inference mode) with episode_id=0.
        """
        B, N, D = tokens.shape
        outputs = []

        # Default handling for inference/simple cases
        if episode_ids is None:
            episode_ids = [0] * B
        if timesteps is None and self.use_timestep_pe:
            # If no timestep provided but needed, we might be in inference with unknown valid time
            # For robustness, just use 0 or current bank length. 
            # Ideally caller should provide this.
            timesteps = [0] * B 

        # Ensure types for iteration
        if isinstance(episode_ids, torch.Tensor):
            episode_ids = episode_ids.tolist()
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.tolist()

        # Training reset logic for group/stream dataloaders
        if self.training:
            if self.dataloader_type == 'group':
                 # In 'group' mode, clear all memory at start of each forward pass
                 self.bank.clear()
                 self.eid_stream = None
            elif self.dataloader_type == 'stream':
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            eid = episode_ids[i]

            # --- Per-sample episode transition (within batch) ---
            if self.training:
                if self.dataloader_type == 'stream':
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]
                elif self.dataloader_type == 'group':
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
            
            # --- Memory Retrieval ---
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)
            hist = self.bank.get(eid, [])
            
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                # Stack history: (T, N, D) -> (1, T*N, D)
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)

                if self.use_timestep_pe and timesteps is not None:
                    hist_ts_vals = [t for t, _ in hist]
                    hist_ts = torch.tensor(hist_ts_vals, dtype=torch.float32).to(working_mem.device)
                    pe = self.timestep_encoder(hist_ts).unsqueeze(0) # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1) # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

                query = working_mem
                for block in self.retrieval_blocks:
                    # Cross attention: Query=Current, Key/Value=History
                    # Add PE to Keys (History)
                    query = block(query, episode_mem + pe, episode_mem)

                retrieved_episode_mem = query
            else:
                # No history, use self as memory (skip retrieval essentially)
                retrieved_episode_mem = working_mem

            # --- Fusion ---
            if self.fusion_type == 'add':
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self.fusion_type == 'gate':
                fused_feats = self.gate_fusion_blocks(working_mem, retrieved_episode_mem)
            else:
                fused_feats = working_mem # Fallback

            outputs.append(fused_feats)

            # --- Memory Consolidate (Update) ---
            # We add the *original* tokens to memory (or fused, if update_fused=True)
            ts_val = timesteps[i] if (timesteps is not None and self.use_timestep_pe) else None
            
            feat_to_save = fused_feats.squeeze(0) if self.update_fused else tokens[i]
            self._memory_consolidate(eid, feat_to_save, ts_val)

        return torch.cat(outputs, dim=0)
