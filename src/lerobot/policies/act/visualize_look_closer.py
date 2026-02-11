"""
One-shot visualization for SpatialRefineGate + Look Closer in ACT.

Usage:
    from lerobot.policies.act.visualize_look_closer import visualize_act_vision

    # After loading policy and a batch:
    visualize_act_vision(policy, batch, save_dir="viz_output", step=0)

Generates a single figure with:
  - Row per camera: original image | Gate spatial mask overlay | feature activation before/after Gate
  - Cross-view attention heatmaps for each Look Closer pair
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from lerobot.utils.constants import OBS_IMAGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prep_img(tensor):
    """(C, H, W) tensor -> (H, W, C) numpy, normalized to [0, 1]."""
    img = tensor.cpu().float().permute(1, 2, 0).numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def _feat_to_heatmap(feat):
    """(C, Hf, Wf) feature map -> (Hf, Wf) numpy, channel-mean activation."""
    return feat.cpu().float().mean(dim=0).numpy()


def _upsample(heatmap, target_h, target_w):
    """Upsample a (Hf, Wf) numpy heatmap to (target_h, target_w)."""
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def _overlay(ax, img, heatmap, title, cmap="jet", alpha=0.5, vmin=None, vmax=None):
    """Show img with heatmap overlay."""
    ax.imshow(img)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize_act_vision(policy, batch, save_dir="viz_output", step=0, sample_idx=0):
    """
    Run one forward pass and generate a comprehensive visualization figure.

    Args:
        policy: ACTPolicy instance.
        batch: A data batch dict.
        save_dir: Output directory.
        step: Step number for file naming.
        sample_idx: Which sample in the batch to visualize.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = policy.model
    policy.eval()

    # --- Forward pass to populate intermediate states ---
    with torch.no_grad():
        # Collect raw backbone features before gate (hook)
        raw_features = []
        gated_features = []

        original_forward = model.spatial_refine_gate.forward if hasattr(model, "spatial_refine_gate") else None

        if original_forward is not None:
            def _hook_forward(x, _orig=original_forward):
                raw_features.append(x.detach())
                out = _orig(x)
                gated_features.append(out.detach())
                return out

            model.spatial_refine_gate.forward = _hook_forward

        # Run model
        policy.predict_action_chunk(batch)

        # Restore
        if original_forward is not None:
            model.spatial_refine_gate.forward = original_forward

    # --- Gather images ---
    if OBS_IMAGES in batch:
        imgs = batch[OBS_IMAGES]
    elif policy.config.image_features:
        imgs = [batch[key] for key in policy.config.image_features]
    else:
        print("No images found in batch.")
        return

    n_cams = len(imgs)
    cam_names = list(policy.config.image_features.keys()) if policy.config.image_features else [f"cam_{i}" for i in range(n_cams)]
    # Shorten names for display
    short_names = [n.split(".")[-1] for n in cam_names]

    img_h, img_w = imgs[0].shape[2], imgs[0].shape[3]

    # =========================================================================
    # Figure 1: SpatialRefineGate effect per camera
    # =========================================================================
    has_gate = hasattr(model, "spatial_refine_gate") and len(raw_features) > 0

    if has_gate:
        fig1, axes1 = plt.subplots(n_cams, 4, figsize=(20, 5 * n_cams))
        if n_cams == 1:
            axes1 = axes1[np.newaxis, :]

        for i in range(n_cams):
            img_np = _prep_img(imgs[i][sample_idx])

            # Raw feature activation
            raw_heat = _feat_to_heatmap(raw_features[i][sample_idx])
            raw_heat_up = _upsample(raw_heat, img_h, img_w)

            # Gated feature activation
            gated_heat = _feat_to_heatmap(gated_features[i][sample_idx])
            gated_heat_up = _upsample(gated_heat, img_h, img_w)

            # Spatial mask
            spatial_mask = model.spatial_refine_gate.last_spatial_mask[sample_idx, 0].cpu().numpy()
            mask_up = _upsample(spatial_mask, img_h, img_w)

            # Plot
            axes1[i, 0].imshow(img_np)
            axes1[i, 0].set_title(f"{short_names[i]}: Original", fontsize=9)
            axes1[i, 0].axis("off")

            _overlay(axes1[i, 1], img_np, mask_up,
                     f"Spatial Gate Mask [{spatial_mask.min():.2f}, {spatial_mask.max():.2f}]",
                     cmap="hot", vmin=0, vmax=1)

            _overlay(axes1[i, 2], img_np, raw_heat_up,
                     "Feature Activation (before Gate)")

            _overlay(axes1[i, 3], img_np, gated_heat_up,
                     "Feature Activation (after Gate)")

        fig1.suptitle(f"SpatialRefineGate Effect (step {step})", fontsize=13, y=1.01)
        fig1.tight_layout()
        p1 = save_path / f"gate_effect_{step:04d}.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved: {p1}")

    # =========================================================================
    # Figure 2: Look Closer cross-view attention
    # =========================================================================
    has_lc = getattr(model, "use_look_closer", False)

    if has_lc:
        # Collect attention pairs: (name, attn_module, query_cam_idx, key_cam_idx)
        lc_pairs = []
        if hasattr(model, "lc_attn_h2l"):
            lc_pairs.append(("head→wrist_l", model.lc_attn_h2l, 0, 1))
            lc_pairs.append(("wrist_l→head", model.lc_attn_l2h, 1, 0))
        if hasattr(model, "lc_attn_h2r") and n_cams >= 3:
            lc_pairs.append(("head→wrist_r", model.lc_attn_h2r, 0, 2))
            lc_pairs.append(("wrist_r→head", model.lc_attn_r2h, 2, 0))

        # Fallback for old 2-cam naming (lc_attn_1, lc_attn_2)
        if not lc_pairs:
            if hasattr(model, "lc_attn_1"):
                lc_pairs.append(("view0→view1", model.lc_attn_1, 0, 1))
                lc_pairs.append(("view1→view0", model.lc_attn_2, 1, 0))

        if lc_pairs:
            n_pairs = len(lc_pairs)
            fig2, axes2 = plt.subplots(n_pairs, 3, figsize=(15, 5 * n_pairs))
            if n_pairs == 1:
                axes2 = axes2[np.newaxis, :]

            for row, (pair_name, attn_mod, q_idx, k_idx) in enumerate(lc_pairs):
                attn_map = attn_mod.last_attention_map[sample_idx]  # (HW_k, HW_q)

                img_q = _prep_img(imgs[q_idx][sample_idx])
                img_k = _prep_img(imgs[k_idx][sample_idx])

                # Global attention: sum over query dim -> heatmap on key
                global_heat = attn_map.sum(dim=1).cpu().numpy()  # (HW_k,)
                hw_k = global_heat.shape[0]
                # Infer spatial dims (may not be square)
                # Use the feature map shape from raw_features if available
                if has_gate and k_idx < len(raw_features):
                    hf_k, wf_k = raw_features[k_idx].shape[2], raw_features[k_idx].shape[3]
                else:
                    hf_k = wf_k = int(np.sqrt(hw_k))
                global_heat = global_heat.reshape(hf_k, wf_k)
                global_heat_up = _upsample(global_heat, img_h, img_w)

                # Point attention: pick center of query feature map
                if has_gate and q_idx < len(raw_features):
                    hf_q, wf_q = raw_features[q_idx].shape[2], raw_features[q_idx].shape[3]
                else:
                    hw_q = attn_map.shape[1]
                    hf_q = wf_q = int(np.sqrt(hw_q))
                center_q = (hf_q // 2) * wf_q + (wf_q // 2)
                point_heat = attn_map[:, center_q].cpu().numpy().reshape(hf_k, wf_k)
                point_heat_up = _upsample(point_heat, img_h, img_w)

                # Plot: query img | global attn on key | point attn on key
                axes2[row, 0].imshow(img_q)
                axes2[row, 0].set_title(f"{pair_name}: Query ({short_names[q_idx]})", fontsize=9)
                # Mark center point on query image
                cy = int((hf_q // 2) / hf_q * img_h)
                cx = int((wf_q // 2) / wf_q * img_w)
                axes2[row, 0].plot(cx, cy, "r+", markersize=15, markeredgewidth=2)
                axes2[row, 0].axis("off")

                _overlay(axes2[row, 1], img_k, global_heat_up,
                         f"Global Attn on {short_names[k_idx]}")

                _overlay(axes2[row, 2], img_k, point_heat_up,
                         f"Point Attn on {short_names[k_idx]} (from center)")

            fig2.suptitle(f"Look Closer Cross-View Attention (step {step})", fontsize=13, y=1.01)
            fig2.tight_layout()
            p2 = save_path / f"cross_view_attn_{step:04d}.png"
            fig2.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved: {p2}")

    # =========================================================================
    # Figure 3: Channel attention weight distribution
    # =========================================================================
    if has_gate:
        ca_weight = model.spatial_refine_gate.last_channel_weight[sample_idx, :, 0, 0].cpu().numpy()

        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 3))
        ax3.bar(range(len(ca_weight)), sorted(ca_weight, reverse=True), width=1.0, color="steelblue")
        ax3.set_xlabel("Channel (sorted by weight)")
        ax3.set_ylabel("Gate weight")
        ax3.set_title(f"Channel Attention Distribution (step {step}) — "
                      f"min={ca_weight.min():.3f}, max={ca_weight.max():.3f}, "
                      f"mean={ca_weight.mean():.3f}")
        fig3.tight_layout()
        p3 = save_path / f"channel_attn_{step:04d}.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        print(f"Saved: {p3}")

    print(f"\nAll visualizations saved to {save_path}/")


if __name__ == "__main__":
    print("Usage:")
    print("  from lerobot.policies.act.visualize_look_closer import visualize_act_vision")
    print("  visualize_act_vision(policy, batch, save_dir='viz_output', step=0)")
