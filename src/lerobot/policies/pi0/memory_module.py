"""
MemoryVLA memory bank module for PI0.

Ported from temp/MemoryVLA/vla/memory_vla.py with minimal adaptation.
Provides episodic memory via cross-attention retrieval and gated fusion.

Dual-track memory:
  - PerMemBank: perception-level memory on SigLIP visual features (pre-LLM).
  - CogMemBank: cognition-level memory on PaliGemma output features (post-LLM).
    Stores and retrieves are decoupled because the store happens after the
    PaliGemma+Expert forward, while retrieval happens before the next forward.
    A learnable retrieve_norm + separate K/V projections in the cross-attention
    blocks bridge the semantic gap between post-LLM stored features and
    pre-LLM query features.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal PE + MLP."""

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
        return self.mlp(t_freq)


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = self.attn_norm(query + attn_out)
        return self.ffn_norm(x + self.ffn(x))


class GateFusion(nn.Module):
    """Gated fusion: output = gate * x_orig + (1 - gate) * x_retrieved.

    Args:
        dim: feature dimension.
        gate_init_bias: initial bias for the gate logit. Controls how much the
            model trusts its own observation vs. retrieved memory at init time.
            - 0.0  → sigmoid(0) = 0.5, equal mix (MemoryVLA default).
            - >0   → sigmoid(b) > 0.5, favors original observation (conservative).
            - <0   → sigmoid(b) < 0.5, favors retrieved memory.
    """

    def __init__(self, dim: int, gate_init_bias: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj.bias, gate_init_bias)

    def forward(self, x_orig: torch.Tensor, x_retrieved: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(torch.cat([x_orig, x_retrieved], dim=-1)))
        return gate * x_orig + (1 - gate) * x_retrieved


# ---------------------------------------------------------------------------
# Base memory bank — shared logic for both Per and Cog tracks
# ---------------------------------------------------------------------------

class MemBank(nn.Module):
    """
    Base episodic memory bank with cross-attention retrieval and gated fusion.

    Follows the MemoryVLA reference (process_batch interface). Subclasses
    (PerMemBank / CogMemBank) differ only in how they are wired into the
    PI0 forward pass, not in the memory logic itself.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = 'gate',
        consolidate_type: str = 'tome',
        update_fused: bool = False,
        gate_init_bias: float = 0.0,
    ):
        super().__init__()
        assert dataloader_type in ('stream', 'group')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')

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
            self.gate_fusion = GateFusion(self.token_size, gate_init_bias=gate_init_bias)

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4,
            )
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    # ------------------------------------------------------------------
    # Memory consolidation
    # ------------------------------------------------------------------

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
            f2 = feats[i + 1].flatten(1) if feats[i + 1].dim() > 1 else feats[i + 1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())
        timestep_i, feat_i = bank[idx_max]
        _, feat_j = bank[idx_max + 1]
        bank[idx_max] = (timestep_i, (0.5 * (feat_i + feat_j)).detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(self, episode_id, feat: torch.Tensor, timestep):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == 'tome':
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError(f"Unknown consolidate_type: {self.consolidate_type}")

    # ------------------------------------------------------------------
    # Episode management helpers (called at batch/sample boundaries)
    # ------------------------------------------------------------------

    def _manage_episodes_batch(self, episode_ids):
        """Batch-level episode housekeeping (called once per process_batch)."""
        if not self.training:
            return
        if self.dataloader_type == 'group':
            self.bank.clear()
            self.eid_stream = None
        elif self.dataloader_type == 'stream':
            first_eid = episode_ids[0]
            if self.eid_stream is not None and self.eid_stream != first_eid:
                self.clear_episode(self.eid_stream)
            self.eid_stream = first_eid

    def _manage_episodes_sample(self, i: int, episode_ids):
        """Per-sample episode transition (called inside the sample loop)."""
        if not self.training:
            return
        if self.dataloader_type == 'group':
            if i > 0 and i % self.group_size == 0:
                self.clear_episode(episode_ids[i - self.group_size])
        elif self.dataloader_type == 'stream':
            if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                self.clear_episode(episode_ids[i - 1])
                self.eid_stream = episode_ids[i]

    # ------------------------------------------------------------------
    # Retrieval + Fusion (single sample)
    # ------------------------------------------------------------------

    def _retrieve_and_fuse_single(
        self, working_mem: torch.Tensor, N: int, D: int, eid
    ) -> torch.Tensor:
        """Retrieve from memory bank and fuse with working_mem for one sample.

        Args:
            working_mem: (1, N, D) current observation features.
            N: token count.
            D: feature dimension.
            eid: episode id for this sample.

        Returns:
            Fused features (1, N, D).
        """
        hist = self.bank.get(eid, [])

        if len(hist) > 0:
            hist_feats = [feat for _, feat in hist]
            N_stored = hist_feats[0].shape[0]
            episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)

            if self.use_timestep_pe:
                hist_ts = torch.tensor(
                    [t for t, _ in hist], dtype=torch.float32
                ).to(working_mem.device)
                pe = self.timestep_encoder(hist_ts).unsqueeze(0)          # (1, T, D)
                pe = pe.repeat_interleave(N_stored, dim=1)                # (1, T*N_stored, D)
            else:
                pe = torch.zeros_like(episode_mem)

            query = working_mem
            for block in self.retrieval_blocks:
                query = block(query, episode_mem + pe, episode_mem)
            retrieved = query
        else:
            retrieved = working_mem

        if self.fusion_type == 'add':
            return (working_mem + retrieved) * 0.5
        else:  # gate
            return self.gate_fusion(working_mem, retrieved)

    # ------------------------------------------------------------------
    # Public interface — full batch processing (MemoryVLA-compatible)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        tokens: torch.Tensor,           # [B, N, D]
        episode_ids: np.ndarray | list,
        timesteps: np.ndarray | list,
    ) -> torch.Tensor:
        """Retrieve → fuse → store for every sample in the batch (sequentially).

        This is the standard MemoryVLA interface where retrieve and store
        happen in the same call.  Used by PerMemBank.
        """
        assert episode_ids is not None, "episode_ids must be provided"
        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided when use_timestep_pe=True"

        B, N, D = tokens.shape
        outputs = []

        self._manage_episodes_batch(episode_ids)

        for i in range(B):
            eid = episode_ids[i]
            self._manage_episodes_sample(i, episode_ids)

            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)
            fused = self._retrieve_and_fuse_single(working_mem, N, D, eid)
            outputs.append(fused)

            # Store into bank
            timestep_i = timesteps[i] if self.use_timestep_pe else None
            store_feat = fused.squeeze(0) if self.update_fused else tokens[i]
            self._memory_consolidate(eid, store_feat, timestep_i)

        return torch.cat(outputs, dim=0)  # [B, N, D]


# ---------------------------------------------------------------------------
# PerMemBank — perception-level memory (pre-LLM, on SigLIP features)
# ---------------------------------------------------------------------------

class PerMemBank(MemBank):
    """Perception Memory Bank. Applied on SigLIP visual embeddings in embed_prefix.

    Identical to base MemBank — uses process_batch where retrieve and store
    happen together in one call.
    """
    pass


# ---------------------------------------------------------------------------
# CogMemBank — cognition-level memory (post-LLM, on PaliGemma output)
# ---------------------------------------------------------------------------

class CogMemBank(MemBank):
    """Cognition Memory Bank. Stores post-LLM prefix_output features and
    retrieves them to augment pre-LLM prefix_embs before the next forward.

    Unlike PerMemBank, store and retrieve are **decoupled**:
      1. retrieve_and_fuse() — called in embed_prefix to augment prefix_embs
         using previously stored cognition features.
      2. store_batch()       — called after paligemma_with_expert.forward to
         store the current prefix_output into the memory bank.

    Space alignment:
      post-LLM features (prefix_output) live in a different distribution than
      pre-LLM embeddings (prefix_embs). Two mechanisms bridge this gap:

      1. ``retrieve_norm`` (LayerNorm) — normalizes stored post-LLM features
         at retrieval time to stabilize K/V distributions for cross-attention.
         This is learnable (affine=True) and receives gradients through the
         retrieval → fusion → loss path.

      2. The CrossTransformerBlock's K/V projections are learned to map from
         the post-LLM feature distribution to a compatible attention space.
         These projections naturally receive gradients during training.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Normalize stored post-LLM features at retrieval time.
        # Learnable affine parameters receive gradients through the retrieval path.
        self.retrieve_norm = nn.LayerNorm(self.token_size)

    def _retrieve_and_fuse_single(
        self, working_mem: torch.Tensor, N: int, D: int, eid
    ) -> torch.Tensor:
        """Override to apply retrieve_norm on stored features before cross-attention."""
        hist = self.bank.get(eid, [])

        if len(hist) > 0:
            hist_feats = [feat for _, feat in hist]
            N_stored = hist_feats[0].shape[0]
            episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)

            # Normalize post-LLM features to stabilize cross-attention
            episode_mem = self.retrieve_norm(episode_mem)

            if self.use_timestep_pe:
                hist_ts = torch.tensor(
                    [t for t, _ in hist], dtype=torch.float32
                ).to(working_mem.device)
                pe = self.timestep_encoder(hist_ts).unsqueeze(0)
                pe = pe.repeat_interleave(N_stored, dim=1)
            else:
                pe = torch.zeros_like(episode_mem)

            query = working_mem
            for block in self.retrieval_blocks:
                query = block(query, episode_mem + pe, episode_mem)
            retrieved = query
        else:
            retrieved = working_mem

        if self.fusion_type == 'add':
            return (working_mem + retrieved) * 0.5
        else:
            return self.gate_fusion(working_mem, retrieved)

    def retrieve_and_fuse(
        self,
        query_tokens: torch.Tensor,     # [B, N_q, D]  (prefix_embs)
        episode_ids: np.ndarray | list,
    ) -> torch.Tensor:
        """Retrieve cognition memory and fuse with current query tokens.

        Only performs retrieval+fusion — does NOT store anything.
        Episode management is handled here (batch + sample level).

        Args:
            query_tokens: current prefix embeddings to augment.
            episode_ids: episode ids for each sample in the batch.

        Returns:
            Memory-augmented tokens [B, N_q, D].
        """
        assert episode_ids is not None, "episode_ids must be provided"

        B, N, D = query_tokens.shape
        outputs = []

        self._manage_episodes_batch(episode_ids)

        for i in range(B):
            eid = episode_ids[i]
            self._manage_episodes_sample(i, episode_ids)

            working_mem = query_tokens[i].unsqueeze(0)
            fused = self._retrieve_and_fuse_single(working_mem, N, D, eid)
            outputs.append(fused)

        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def store_batch(
        self,
        tokens: torch.Tensor,           # [B, N, D]  (prefix_output)
        episode_ids: np.ndarray | list,
        timesteps: np.ndarray | list,
    ):
        """Store post-LLM features into the memory bank (no retrieval).

        Stores raw post-LLM features. Space alignment is handled at retrieval
        time by retrieve_norm, which is learnable and receives gradients.

        Args:
            tokens: prefix_output features to store.
            episode_ids: episode ids for each sample.
            timesteps: timestep indices for positional encoding.
        """
        assert episode_ids is not None, "episode_ids must be provided"
        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided when use_timestep_pe=True"

        B = tokens.shape[0]
        for i in range(B):
            eid = episode_ids[i]
            timestep_i = timesteps[i] if self.use_timestep_pe else None
            self._memory_consolidate(eid, tokens[i], timestep_i)
