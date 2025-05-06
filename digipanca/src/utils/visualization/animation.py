import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import Image, display
import matplotlib.patches as mpatches

from src.utils.tensors import prepare_tensors_for_visualization
from src.utils.visualization.colormaps import CMAP, NORM, CLASS_LABELS

def create_2d_animation(
        predictions,
        ground_truth,
        patient_id,
        output_dir,
        volume=None,
        alpha=0.6,
        filename=None
    ):
    """
    Creates an animation comparing predictions and ground truth masks,
    optionally overlaid on the CT volume, and saves it.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions (B, C, D, H, W)
    ground_truth : torch.Tensor
        Ground truth masks (B, D, H, W)
    patient_id : str
        Patient ID
    output_dir : str
        Output directory
    volume : torch.Tensor, optional
        CT volume (B, D, H, W). If provided, it will be used as grayscale 
        background.
    alpha : float, optional
        Transparency of the overlay (default: 0.6).
    filename : str, optional
        Filename for the saved animation. If not provided, a default name will 
        be used.
    """
    # Get prepared tensors for visualization
    pred_np, gt_np, volume_np = prepare_tensors_for_visualization(
        predictions,
        ground_truth,
        volume
    )

    D, H, W = pred_np.shape

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    
    # Use CT volume or black background
    background = volume_np[0] if volume_np is not None else np.zeros((H, W))
    
    # Initial slice
    gt_img = axes[0].imshow(background, cmap='gray')
    gt_overlay = axes[0].imshow(
        np.where(gt_np[0] > 0, gt_np[0], np.nan),
        cmap=CMAP,
        norm=NORM,
        alpha=alpha
    )
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    pred_img = axes[1].imshow(background, cmap='gray')
    pred_overlay = axes[1].imshow(
        np.where(pred_np[0] > 0, pred_np[0], np.nan),
        cmap=CMAP,
        norm=NORM,
        alpha=alpha
    )
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    title = fig.suptitle(
        f"Patient {patient_id} - Slice 0",
        fontsize=16,
        fontweight="bold",
        y=0.96
    )

    legend_elements = []
    for class_idx, class_name in enumerate(CLASS_LABELS, start=1):
        # Get the color for the class
        rgba = CMAP(NORM(class_idx))
        # Create a patch for the legend
        patch = mpatches.Patch(color=rgba, label=class_name)
        legend_elements.append(patch)

    # Add legend to the figure
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=min(len(CLASS_LABELS), 4),  # Divide if more than 4 classes
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # Ajustar la disposición para dejar espacio para la leyenda
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    def update(frame):
        bg = volume_np[frame] if volume_np is not None else np.zeros((H, W))
        gt_img.set_array(bg)
        pred_img.set_array(bg)

        gt_overlay.set_array(np.where(gt_np[frame] > 0, gt_np[frame], np.nan))
        pred_overlay.set_array(np.where(pred_np[frame] > 0, pred_np[frame], np.nan))

        title.set_text(f"Patient {patient_id} - Slice {frame}")
        return [gt_img, gt_overlay, pred_img, pred_overlay, title]

    anim = animation.FuncAnimation(fig, update, frames=D, interval=200, blit=False)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save animation
    if filename is None:
        filename = f"patient_{patient_id}_2d_animation.gif"
    output_path = os.path.join(output_dir, filename)
    
    # Use PillowWriter directly
    writer = animation.PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)  # Close the figure to free memory
    
    # print(f"2D animation saved to {output_path}")
    return output_path

def display_gif_in_notebook(gif_path):
    """
    Display a GIF in a Jupyter notebook.

    Parameters
    ----------
    gif_path : str
        Path to the GIF file.
    """
    display(Image(filename=gif_path))