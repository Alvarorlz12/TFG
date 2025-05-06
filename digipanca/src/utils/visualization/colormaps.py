import matplotlib.colors as mcolors

# Color map for segmentation classes
CMAP = mcolors.ListedColormap(['green', 'purple', 'red', 'blue'])

# Boundaries between segmentation classes
BOUNDARIES = [0.5, 1.5, 2.5, 3.5, 4.5]

# Normalization based on boundaries
NORM = mcolors.BoundaryNorm(BOUNDARIES, CMAP.N)

# Class labels
CLASS_LABELS = ['Pancreas', 'Tumor', 'Arteries', 'Veins']