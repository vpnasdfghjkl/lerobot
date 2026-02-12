# Memory-Augmented PI0 Policy

This document describes the integration of a MemoryVLA module (CogMemBank) into the PI0 policy. This enhancement allows the policy to leverage historical context from past frames within an episode, improving its ability to handle long-horizon tasks and temporal dependencies.

## Key Features

1.  **CogMemBank Integration**:
    -   A new `CogMemBank` module is added to manage memory storage and retrieval.
    -   It supports storing image embeddings from previous timesteps.
    -   Uses a cross-attention mechanism to retrieve relevant information from memory based on the current observation.
    -   Supports different fusion types (e.g., "gate") to combine current features with retrieved memory features.

2.  **Episode-Aware Training**:
    -   The training pipeline is updated to respect episode boundaries.
    -   `EpisodeAwareSampler` is configured to keep frames from the same episode contiguous when `use_memory` is enabled.
    -   Shuffle is disabled for the sampler to ensure sequential processing of frames required for memory state updates during training.

3.  **Configuration Options**:
    -   New configuration parameters have been added to `PI0Config` to control memory behavior.

## Configuration Parameters

The following parameters in `PI0Config` control the MemoryVLA feature:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `use_memory` | `bool` | `False` | Enables or disables the memory module. |
| `memory_length` | `int` | `16` | The maximum number of past frames to store in memory. |
| `memory_retrieval_layers` | `int` | `2` | Number of cross-attention layers used for memory retrieval. |
| `memory_fusion_type` | `str` | `"gate"` | Method to fuse memory features with current features (e.g., "gate", "add"). |
| `memory_consolidate_type` | `str` | `"fifo"` | Strategy for managing memory buffer (e.g., "fifo" for First-In-First-Out). |
| `drop_n_last_frames` | `int` | `0` | Number of frames to drop at the end of each episode. When `use_memory` is True, this defaults to `chunk_size` if not set, to ensure proper chunking. |

## Implementation Details

### `src/lerobot/policies/pi0/memory_module.py`
Contains the implementation of `CogMemBank`, `CrossTransformerBlock`, `GateFusion`, and `TimestepEmbedder`.

### `src/lerobot/policies/pi0/modeling_pi0.py`
-   **Initialization**: `CogMemBank` is initialized in `PI0Pytorch` if `use_memory` is set.
-   **Forward Pass**:
    -   In `embed_prefix`, image embeddings are passed through the memory bank before being used for the VLM.
    -   Episode IDs and timesteps are passed to `forward` and `embed_prefix` to manage memory state (resetting at episode boundaries).

### `src/lerobot/policies/pi0/configuration_pi0.py`
-   Added memory-related configuration fields.
-   Validation logic to ensure `drop_n_last_frames` is set correctly when using memory.

### `src/lerobot/scripts/lerobot_train.py`
-   Updates the data sampler to `EpisodeAwareSampler` with `shuffle=False` when `use_memory` is enabled, ensuring batches contain sequential frames from episodes.

### `src/lerobot/processor/converters.py`
-   Enhanced to extract `episode_index`, `frame_index`, and `timestamp` from the dataset, which are necessary for the memory module to track episode boundaries and timesteps.
