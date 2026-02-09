
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

def visualize_cross_view_attention(policy, batch, save_dir="attention_viz", step=0):
    """
    Visualizes the Cross-View Attention maps from the modified ACT policy.
    
    Args:
        policy: The ACTPolicy instance (must have use_look_closer=True).
        batch: A batch of data containing images.
        save_dir: Directory to save visualization images.
        step: Current step or index for naming files.
    """
    policy.eval()
    
    # Ensure directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Run forward pass to populate attention maps
        if policy.config.temporal_ensemble_coeff is not None:
             # This calls model() internally
             policy.predict_action_chunk(batch)
        else:
             policy.model(batch)

    if not getattr(policy.model, "use_look_closer", False):
        print("Policy does not use Look Closer module. Cannot visualize attention.")
        return

    # Get images
    # Replicate logic from ACTPolicy to retrieve images ordered correctly
    from lerobot.utils.constants import OBS_IMAGES
    
    if OBS_IMAGES in batch:
        imgs = batch[OBS_IMAGES]
    elif policy.config.image_features:
        imgs = [batch[key] for key in policy.config.image_features]
    else:
        print("No images found in batch config.")
        return

    # Denormalize images for plotting (assuming standard ImageNet normalization or similar)
    # If pixels are 0-1, just clone.
    
    img0_tensor = imgs[0][0].cpu() # First sample in batch, View 0
    img1_tensor = imgs[1][0].cpu() # First sample in batch, View 1
    
    # Retrieve stored attention maps
    # Shape: (Batch, HW_key, HW_query)
    # attn1: Query=View0, Key=View1. Shape (B, HW_1, HW_0) -> Attention on View 1 (Key) from View 0
    attn1 = policy.model.lc_attn_1.last_attention_map[0] # (HW_1, HW_0)
    
    # attn2: Query=View1, Key=View0. Shape (B, HW_0, HW_1) -> Attention on View 0 (Key) from View 1
    attn2 = policy.model.lc_attn_2.last_attention_map[0] # (HW_0, HW_1)

    # We want to visualize "Global Attention": Which parts of Key are most attended by Query?
    # Sum over Query dimension (dim=1)
    heat1 = attn1.sum(dim=1) # (HW_1,) -> Heatmap on View 1
    heat2 = attn2.sum(dim=1) # (HW_0,) -> Heatmap on View 0
    
    # Reshape to 2D
    # We need to know feature map size (h, w). sqrt(HW) is a good guess if square.
    hw1 = heat1.shape[0]
    h1 = w1 = int(np.sqrt(hw1))
    
    hw0 = heat2.shape[0]
    h0 = w0 = int(np.sqrt(hw0))
    
    heat1 = heat1.reshape(h1, w1)
    heat2 = heat2.reshape(h0, w0)
    
    # Upsample heatmaps to image size
    img_h, img_w = img0_tensor.shape[1], img0_tensor.shape[2]
    
    heat1_up = F.interpolate(heat1.unsqueeze(0).unsqueeze(0), size=(img_h, img_w), mode='bilinear')[0,0].numpy()
    heat2_up = F.interpolate(heat2.unsqueeze(0).unsqueeze(0), size=(img_h, img_w), mode='bilinear')[0,0].numpy()
    
    # Normalize images for display
    # Assuming images might be normalized, simplistic min-max to 0-1
    def prep_img(t):
        t = t.permute(1, 2, 0).numpy()
        t = t - t.min()
        t = t / (t.max() + 1e-6)
        return t

    obs0 = prep_img(img0_tensor)
    obs1 = prep_img(img1_tensor)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # View 0 (3rd Person)
    axes[0,0].imshow(obs0)
    axes[0,0].set_title("View 0 (Query for map below)")
    axes[0,0].axis('off')
    
    # Attention on View 0 (from View 1)
    axes[0,1].imshow(obs0)
    axes[0,1].imshow(heat2_up, cmap='jet', alpha=0.5)
    axes[0,1].set_title("Attn on View 0 (from View 1)")
    axes[0,1].axis('off')

    # View 1 (Ego)
    axes[1,0].imshow(obs1)
    axes[1,0].set_title("View 1 (Query for map above)")
    axes[1,0].axis('off')
    
    # Attention on View 1 (from View 0)
    axes[1,1].imshow(obs1)
    axes[1,1].imshow(heat1_up, cmap='jet', alpha=0.5)
    axes[1,1].set_title("Attn on View 1 (from View 0)")
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f"cross_view_attn_{step:04d}.png")
    plt.close()
    print(f"Saved attention visualization to {save_path / f'cross_view_attn_{step:04d}.png'}")

if __name__ == "__main__":
    print("This module provides 'visualize_cross_view_attention' function.")
    print("Import it in your training or eval loop.")
