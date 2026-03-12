"""Memory integration helper for VLA policies.

Provides a lightweight helper that encapsulates memory bank initialization,
prefix augmentation, post-forward storage, and reset logic. Policies integrate
memory by calling a few helper methods rather than embedding memory logic directly.

Usage in a VLA model's __init__:
    self.memory = MemoryIntegration(config.memory, token_size=hidden_dim, dtype=dtype)

Usage in embed_prefix (after image embedding, before language concat):
    img_embs = self.memory.augment_perception(img_embs, episode_ids, episode_timesteps)

Usage in embed_prefix (after all prefix tokens are concatenated):
    prefix_embs = self.memory.augment_cognition_retrieve(prefix_embs, episode_ids)

Usage after VLM forward:
    self.memory.store_cognition(prefix_out, episode_ids, episode_timesteps)

Usage in reset:
    self.memory.reset()
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.memory.memory_config import MemoryConfig
from lerobot.policies.memory.memory_module import CogMemBank, PerMemBank


class MemoryIntegration(nn.Module):
    """Encapsulates MemoryVLA bank initialization and augmentation hooks.

    This module owns the PerMemBank and CogMemBank instances and exposes
    a minimal API for integration into any VLA forward pass.

    Args:
        config: MemoryConfig instance (or None to disable).
        token_size: Hidden dimension of the VLM embeddings.
        dtype: Optional dtype string ("bfloat16" or "float32").
    """

    def __init__(self, config: MemoryConfig | None, token_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.enabled = config is not None
        self.cog_enabled = config is not None and config.use_cog_memory

        if not self.enabled:
            self.per_mem_bank = None
            self.cog_mem_bank = None
            return

        mem_bank_kwargs = dict(
            dataloader_type=config.memory_dataloader_type,
            group_size=config.memory_group_size,
            token_size=token_size,
            mem_length=config.memory_length,
            retrieval_layers=config.memory_retrieval_layers,
            fusion_type=config.memory_fusion_type,
            consolidate_type=config.memory_consolidate_type,
            use_timestep_pe=True,
            gate_init_bias=config.memory_gate_init_bias,
        )

        self.per_mem_bank = PerMemBank(**mem_bank_kwargs)
        if dtype == "bfloat16":
            self.per_mem_bank = self.per_mem_bank.to(dtype=torch.bfloat16)

        if self.cog_enabled:
            self.cog_mem_bank = CogMemBank(**mem_bank_kwargs)
            if dtype == "bfloat16":
                self.cog_mem_bank = self.cog_mem_bank.to(dtype=torch.bfloat16)
        else:
            self.cog_mem_bank = None

    def reset(self):
        """Clear all memory banks. Call on environment/episode reset."""
        if self.per_mem_bank is not None:
            self.per_mem_bank.reset()
        if self.cog_mem_bank is not None:
            self.cog_mem_bank.reset()

    def augment_perception(
        self,
        img_embs: torch.Tensor,
        episode_ids,
        episode_timesteps,
    ) -> torch.Tensor:
        """Apply perception-level memory (PerMemBank) to image embeddings.

        Args:
            img_embs: (B, N, D) merged image embeddings from vision encoder.
            episode_ids: Episode ID per sample in batch.
            episode_timesteps: Frame index per sample in batch.

        Returns:
            Memory-augmented image embeddings (B, N, D).
        """
        if self.per_mem_bank is None:
            return img_embs
        return self.per_mem_bank.process_batch(
            tokens=img_embs,
            episode_ids=episode_ids,
            timesteps=episode_timesteps,
        )

    def augment_cognition_retrieve(
        self,
        prefix_embs: torch.Tensor,
        episode_ids,
    ) -> torch.Tensor:
        """Retrieve cognition-level memory and fuse with prefix embeddings.

        Args:
            prefix_embs: (B, N, D) concatenated prefix embeddings.
            episode_ids: Episode ID per sample in batch.

        Returns:
            Memory-augmented prefix embeddings (B, N, D).
        """
        if self.cog_mem_bank is None:
            return prefix_embs
        return self.cog_mem_bank.retrieve_and_fuse(
            query_tokens=prefix_embs,
            episode_ids=episode_ids,
        )

    @torch.no_grad()
    def store_cognition(
        self,
        prefix_out: torch.Tensor,
        episode_ids,
        episode_timesteps,
    ):
        """Store post-VLM prefix features into cognition memory bank.

        Args:
            prefix_out: (B, N, D) VLM output features for the prefix.
            episode_ids: Episode ID per sample in batch.
            episode_timesteps: Frame index per sample in batch.
        """
        if self.cog_mem_bank is None:
            return
        self.cog_mem_bank.store_batch(
            tokens=prefix_out,
            episode_ids=episode_ids,
            timesteps=episode_timesteps,
        )
