"""Shared memory configuration for VLA policies."""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for MemoryVLA dual-track episodic memory.

    Can be embedded into any VLA policy config as a single field:
        memory: MemoryConfig | None = None

    When set to None (default), memory is disabled with zero overhead.
    """

    use_cog_memory: bool = False
    memory_length: int = 16
    memory_retrieval_layers: int = 2
    memory_fusion_type: str = "gate"
    memory_consolidate_type: str = "tome"
    memory_dataloader_type: str = "stream"
    memory_group_size: int = 16
    memory_gate_init_bias: float = 2.0
    drop_n_last_frames: int = 0

    def __post_init__(self):
        if self.memory_fusion_type not in ("gate", "add"):
            raise ValueError(f"Invalid memory_fusion_type: {self.memory_fusion_type}")
        if self.memory_consolidate_type not in ("fifo", "tome"):
            raise ValueError(f"Invalid memory_consolidate_type: {self.memory_consolidate_type}")
        if self.memory_dataloader_type not in ("stream", "group"):
            raise ValueError(f"Invalid memory_dataloader_type: {self.memory_dataloader_type}")
