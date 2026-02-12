"""
Visualization utilities for ACT Policy features.
Generates single-model 4-stage feature maps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lerobot.utils.constants import OBS_IMAGES

def _prep_img(tensor):
    """(C, H, W) tensor -> (H, W, C) numpy, normalized to [0, 1]."""
    img = tensor.cpu().float().permute(1, 2, 0).numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

def _feat_to_heatmap(feat, method="norm"):
    """
    (C, H, W) feature -> (H, W) heatmap.
    method: "mean" (avg value) or "norm" (vector length/energy).
    Transformer features usually look better with "norm".
    """
    if feat.ndim == 2:
        return feat.cpu().float().numpy()

    if feat.ndim != 3:
        return np.zeros((10, 10))
    
    if method == "norm":
        # Calculate L2 norm across channel dimension
        # (C, H, W) -> (H, W)
        return torch.norm(feat.cpu().float(), p=2, dim=0).numpy()
    else:
        # Mean
        return feat.cpu().float().mean(dim=0).numpy()

def _min_max_norm(arr):
    """Normalize numpy array to [0,1] for cleaner plotting."""
    _min, _max = arr.min(), arr.max()
    if _max - _min < 1e-8:
        return arr - _min
    return (arr - _min) / (_max - _min)

def _upsample(heatmap, target_h, target_w):
    """Upsample a (Hf, Wf) numpy heatmap to (target_h, target_w)."""
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()

def visualize_single_model_features(
    policy, 
    batch, 
    save_dir, 
    step=0,
    fname_prefix="viz"
):
    """
    Visualizes available feature stages for a single model.
    Stages: Input -> Backbone -> Gate (opt) -> LookCloser (opt) -> Encoder -> DecoderAttn
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Inference with Hooks
    features, decoder_attn, gate_masks = _run_viz_forward_full(policy, batch)

    # 2. Get Input Images
    imgs = []
    if OBS_IMAGES in batch:
        imgs = batch[OBS_IMAGES]
    elif policy.config.image_features:
        imgs = [batch[key] for key in policy.config.image_features]
    
    if not imgs:
        print("No images found.")
        return

    # Sample 0
    sample_idx = 0
    n_cams = len(imgs)
    img_h, img_w = imgs[0].shape[2], imgs[0].shape[3]

    # 3. Prepare Data for Plotting
    plot_data = []
    
    # Check config 
    use_gate = getattr(policy.config, "use_spatial_refine_gate", False)
    use_lc = getattr(policy.config, "use_look_closer", False)

    # Feature Stages
    # 1. Backbone (Base Feature)
    if "backbone" in features:
        plot_data.append(("Backbone", features["backbone"], "norm"))
        
    # 2. Gate Mask (The attention map itself, 0-1) - NEW
    # This helps diagnose if the gate is learning a good mask.
    if use_gate and gate_masks:
        plot_data.append(("GateMask", gate_masks, "heatmap")) # direct heatmap

    # 3. Gate Output (Feature * Mask)
    if use_gate and "gate" in features and features["gate"]:
        plot_data.append(("GateFeat", features["gate"], "norm"))
        
    # 4. CrossAttn (LookCloser)
    if use_lc and "look_closer" in features and features["look_closer"]:
        plot_data.append(("CrossAttn", features["look_closer"], "norm"))

    # 5. Decoder Attention
    dec_attn_maps = _process_decoder_attn(decoder_attn, n_cams, img_h, img_w)
    if dec_attn_maps:
        plot_data.append(("DecAttn", dec_attn_maps, "norm")) # Already processed to 2D
        
    # Calculate grid size
    total_cols = 1 + len(plot_data)
    
    fig, axes = plt.subplots(n_cams, total_cols, figsize=(3 * total_cols, 3 * n_cams))
    if n_cams == 1:
        axes = axes[np.newaxis, :]

    for cam_idx in range(n_cams):
        # --- Column 0: Input Image ---
        img_np = _prep_img(imgs[cam_idx][sample_idx])
        axes[cam_idx, 0].imshow(img_np)
        if cam_idx == 0: axes[cam_idx, 0].set_title("Input Image", fontsize=10, fontweight='bold')
        axes[cam_idx, 0].axis("off")

        # --- Feature Columns ---
        for col_i, (label, data_list, viz_method) in enumerate(plot_data):
            ax = axes[cam_idx, col_i + 1]
            
            if data_list is None or cam_idx >= len(data_list):
                ax.axis("off")
                continue
            
            # data_list[cam_idx] is (B, C, H, W) OR (B, H, W) if pre-processed
            feat = data_list[cam_idx][sample_idx]
            
            # Prepare heatmap
            if viz_method == "heatmap":
                # It's already a single channel or 2D map (like Gate Mask or DecAttn)
                if feat.ndim == 3 and feat.shape[0] == 1: 
                    heatmap = feat[0].cpu().numpy()
                elif feat.ndim == 2:
                    heatmap = feat.cpu().numpy()
                else: 
                     # Unexpected
                    heatmap = feat.mean(dim=0).cpu().numpy()
            else: # "norm" or "mean"
                heatmap = _feat_to_heatmap(feat, method=viz_method)
            
            # Normalize heatmap for display (unless it is a probability/mask [0,1])
            # For GateMask (Sigmoid), it is already [0,1], but min_max helps visualization contrast 
            # if values are clustered (e.g. all 0.4-0.5).
            # But true 0-1 is better for "Mask".
            if label == "GateMask":
                # Don't normalize GateMask, show absolute strength
                # It comes from Sigmoid so it is in [0, 1]
                pass 
            else:
                heatmap = _min_max_norm(heatmap)
            
            heatmap_up = _upsample(heatmap, img_h, img_w)
            
            ax.imshow(img_np, alpha=1.0)
            ax.imshow(heatmap_up, cmap="jet", alpha=0.6, vmin=0, vmax=1) 
            # vmin/vmax=0/1 ensures GateMask is interpreted correctly as opacity
            
            if cam_idx == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            ax.axis("off")

    plt.tight_layout()
    out_file = save_path / f"{fname_prefix}_{step:06d}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _run_viz_forward_full(policy, batch):
    model = policy.model
    model.return_features_for_viz = True
    
    # --- Hook Decoder Attention ---
    last_decoder_layer = model.decoder.layers[-1]
    cross_attn_module = last_decoder_layer.multihead_attn
    captured_dec = {}
    
    def hook_dec_fn(module, input, output):
        # output is (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            captured_dec["weights"] = output[1]
    
    handle_dec = cross_attn_module.register_forward_hook(hook_dec_fn)

    # --- Hook Gate Mask (If exists) ---
    captured_masks = []
    handle_gate = None
    
    if hasattr(model, "spatial_refine_gate") and model.spatial_refine_gate is not None:
        # Hook the 'spatial_gate' submodule which outputs the mask
        # SpatialRefineGate structure: self.spatial_gate = nn.Sequential(...)
        gate_module = model.spatial_refine_gate.spatial_gate
        
        def hook_gate_fn(module, input, output):
            # output is the spatial mask (B, 1, H, W)
            captured_masks.append(output.detach())
            
        handle_gate = gate_module.register_forward_hook(hook_gate_fn)

    # --- Run Forward ---
    with torch.no_grad():
        policy.predict_action_chunk(batch)
    
    # Cleanup Hooks
    handle_dec.remove()
    if handle_gate:
        handle_gate.remove()
    
    # --- Collect Features ---
    feats = {}
    if hasattr(model, "viz_features"):
        for k, v in model.viz_features.items():
            if isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], torch.Tensor):
                    feats[k] = [x.detach() for x in v]
                else:
                    feats[k] = v 
            elif isinstance(v, torch.Tensor):
                feats[k] = v.detach()
    
    model.return_features_for_viz = False
    
    return feats, captured_dec.get("weights"), captured_masks
def _process_decoder_attn(attn_weights, n_cams, img_h, img_w):
    """
    attn_weights: (B, Chunk_Size, Seq_Enc)
    Seq_Enc includes Latent(1) + Robot(1) + Env(1) + Images(N*H*W).
    We need to extract the Image part and reshape.
    """
    if attn_weights is None:
        return None
        
    # Average attention across all Action Chunk queries (temporal average)
    # (B, Seq_Enc)
    attn_avg = attn_weights.mean(dim=1) 
    
    # Remove non-image tokens. 
    # Usually first 1 (latent), +1 (robot state), maybe +1 (env state).
    # This depends on config. 
    # Let's assume standard ACT: Latent + RobotState.
    # But strictly we should calculate.
    # Seq_Enc len = M. 
    # Image tokens = N_cams * H_feat * W_feat.
    # The suffix is image tokens.
    
    # H_feat, W_feat? We need to infer from total length or pass in backbone feat size.
    # Let's guess 15x20 (480x640 / 32).
    # Better: H_feat = ceil(img_h / 32), W_feat = ceil(img_w / 32) for ResNet18/34/50
    # Actually, let's use the known n_cams. 
    # M = offset + n_cams * feat_pixels.
    
    seq_len = attn_avg.shape[1]
    
    # Estimate standard feat size (ResNet stride 32)
    fh = int(np.ceil(img_h / 32))
    fw = int(np.ceil(img_w / 32))
    img_tokens_count = n_cams * fh * fw
    
    # If mismatch, try strict division
    if img_tokens_count > seq_len:
        # Maybe stride is different or something
        # Try finding offset
        # common offsets: 1, 2, 3.
        # Let's try to fit.
        valid = False
        for off in [1, 2, 3]:
            rem = seq_len - off
            if rem % n_cams == 0:
                pix = rem // n_cams
                # is pix square-ish?
                ratio = img_w / img_h
                # w_feat / h_feat ~ ratio
                # w * h = pix
                # w = h * ratio -> h^2 * ratio = pix -> h = sqrt(pix/ratio)
                h_est = int(np.sqrt(pix / ratio))
                w_est = pix // h_est
                if h_est * w_est == pix:
                    fh, fw = h_est, w_est
                    offset = off
                    valid = True
                    break
        if not valid:
            return None # Cannot reshape
    else:
        offset = seq_len - img_tokens_count
    
    # Slice image tokens
    attn_img = attn_avg[:, offset:] # (B, Img_Tokens)
    
    maps = []
    pix_per_cam = fh * fw
    for i in range(n_cams):
        start = i * pix_per_cam
        end = (i+1) * pix_per_cam
        cam_attn = attn_img[:, start:end] # (B, H*W)
        cam_map = cam_attn.view(-1, fh, fw) # (B, fh, fw)
        maps.append(cam_map) # List of (B, H, W)
        
    return maps



def _process_encoder_out(encoder_out, n_cams, backbone_feats):
    if encoder_out is None or not backbone_feats:
        return None
    
    feat_sample = backbone_feats[0] 
    feat_h, feat_w = feat_sample.shape[2], feat_sample.shape[3]
    
    # encoder_out: (B, Seq, D)
    enc_T = encoder_out.transpose(1, 2) # (B, D, Seq)
    
    pixels_per_cam = feat_h * feat_w
    
    if enc_T.shape[2] != n_cams * pixels_per_cam:
        # Fallback/Error check
        return None
        
    enc_cam_features = []
    
    for i in range(n_cams):
        start = i * pixels_per_cam
        end = (i+1) * pixels_per_cam
        cam_toks = enc_T[:, :, start:end] # (B, D, H*W)
        cam_feat = cam_toks.view(-1, cam_toks.shape[1], feat_h, feat_w)
        enc_cam_features.append(cam_feat)
        
    return enc_cam_features
