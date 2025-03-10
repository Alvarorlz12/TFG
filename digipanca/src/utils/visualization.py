import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Colormap and normalization for the segmentation mask
CMAP = mcolors.ListedColormap(['green', 'purple', 'red', 'blue'])
BOUNDARIES = [0.5, 1.5, 2.5, 3.5, 4.5]
NORM = mcolors.BoundaryNorm(BOUNDARIES, CMAP.N)

def visualize_sample(dataset, idx, alpha=0.5, zoom=1.0):
    """
    Visualize a sample from the dataset with the segmentation mask overlay.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object.
    idx : int
        Index of the sample to visualize.
    alpha : float, optional
        Transparency of the overlay (default: 0.5).
    zoom : float, optional
        Zoom factor for the cropped image (default: 1.0, max).
    """
    # Get the image, mask, and patient ID
    image, mask, patient_id = dataset[idx]
    
    # Convert to NumPy arrays
    image = np.array(image.squeeze())  # Remove channel dimension
    mask = np.array(mask)

    # Create a masked array for the segmentation mask (to ignore background)
    mask_no_bg = np.ma.masked_where(mask == 0, mask)

    # Crop the image and mask
    h, w = image.shape
    crop_h, crop_w = round(200 * zoom), round(150 * zoom)
    
    x_start, x_end = max(0, crop_h), min(h, h - crop_h)
    y_start, y_end = max(0, crop_w), min(w, w - crop_w)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. CT image
    axes[0].imshow(image[x_start:x_end, y_start:y_end], cmap='gray')
    axes[0].set_title(f'CT Image - Patient {patient_id}')
    axes[0].axis('off')

    # 2. Overlay
    axes[1].imshow(image[x_start:x_end, y_start:y_end], 
                   cmap='gray', extent=[0, w, 0, h])
    axes[1].imshow(mask_no_bg[x_start:x_end, y_start:y_end],
                   cmap=CMAP, norm=NORM, alpha=alpha, extent=[0, w, 0, h])
    axes[1].set_title(f'Overlay - Patient {patient_id}')
    axes[1].axis('off')

    # 3. Segmentation mask
    axes[2].imshow(mask_no_bg[x_start:x_end, y_start:y_end],
                   cmap=CMAP, norm=NORM)
    axes[2].set_title(f'Segmentation Mask - Patient {patient_id}')
    axes[2].axis('off')

    # Add a colorbar with the segmentation classes
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    colorbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                            cax=cbar_ax)
    colorbar.set_ticks([1, 2, 3, 4])
    colorbar.set_ticklabels(['Pancreas', 'Tumor', 'Arteries', 'Veins'])

    # Adjust the layout
    plt.subplots_adjust(wspace=0.4, right=0.9)
    plt.show()

def visualize_label_mask(dataset, idx, label, title="Label visualization"):
    """
    Visualize a binary mask for a specific label. Ignores all labels except the
    specified one, which is shown in white.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object.
    idx : int
        Index of the sample to visualize.
    label : int
        Label to visualize.
    title : str, optional
        Title of the plot (default: "Label visualization").
    """
    # Get the image, mask, and patient ID
    _, mask, _ = dataset[idx]
    
    # Convert to NumPy arrays
    mask = np.array(mask)

    # Create a binary mask for the specified label
    binary_mask = (mask == label).astype(np.uint8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap='gray')  # White for label, black for BG
    plt.title(title)
    plt.axis('off')
    plt.show()