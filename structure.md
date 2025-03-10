pancreas-segmentation/
├── data/
│   ├── raw/                      # Original DICOM/NIFTI files
│   ├── processed/                # Preprocessed data ready for training
│   ├── splits/                   # Train/val/test splits
│   └── external/                 # Additional datasets for transfer learning
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch dataset classes
│   │   ├── preprocessing.py      # Data preprocessing pipelines
│   │   ├── augmentation.py       # Data augmentation strategies
│   │   └── transforms.py         # Custom transforms
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet/                 # U-Net variants
│   │   │   ├── __init__.py
│   │   │   ├── standard.py
│   │   │   ├── attention.py
│   │   │   └── nested.py
│   │   ├── transformer/          # Transformer-based models
│   │   │   ├── __init__.py
│   │   │   ├── swin_unet.py
│   │   │   └── transunet.py
│   │   ├── backbones/            # Feature extractors
│   │   │   ├── __init__.py
│   │   │   ├── resnet.py
│   │   │   └── efficientnet.py
│   │   └── ensemble.py           # Model ensemble strategies
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py               # Dice loss implementations
│   │   ├── boundary.py           # Boundary-aware losses
│   │   ├── focal.py              # Focal loss for class imbalance
│   │   └── compound.py           # Compound loss functions
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── segmentation.py       # Segmentation metrics (Dice, IoU, etc.)
│   │   ├── clinical.py           # Clinical relevance metrics
│   │   └── visualization.py      # Metric visualization
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py             # Logging utilities
│   │   ├── visualization.py      # Visualization tools for 2D/3D data
│   │   ├── checkpoints.py        # Model checkpoint management
│   │   └── profiling.py          # Performance profiling
│   │
│   └── training/
│       ├── __init__.py
│       ├── trainer.py            # Training loop
│       ├── scheduler.py          # Learning rate schedulers
│       └── callbacks.py          # Training callbacks
│
├── configs/
│   ├── data/
│   │   ├── default.yaml          # Default data configuration
│   │   └── augmentation.yaml     # Augmentation configurations
│   ├── models/
│   │   ├── unet.yaml             # U-Net configuration
│   │   ├── attention_unet.yaml   # Attention U-Net configuration
│   │   └── swin_unet.yaml        # Swin-UNet configuration
│   ├── training/
│   │   ├── default.yaml          # Default training parameters
│   │   └── optimizer.yaml        # Optimizer configurations
│   └── experiments/              # Full experiment configurations
│       ├── baseline.yaml
│       ├── experiment_001.yaml
│       └── experiment_002.yaml
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Inference script
│   ├── data_preprocessing.py     # Data preprocessing script
│   └── visualization.py          # Visualization script
│
├── notebooks/
│   ├── exploration/
│   │   ├── data_analysis.ipynb   # Dataset exploration and analysis
│   │   └── results_analysis.ipynb # Results analysis
│   ├── preprocessing/
│   │   └── preprocessing_demo.ipynb # Preprocessing demonstration
│   └── inference/
│       └── model_inference.ipynb # Model inference examples
│
├── experiments/
│   ├── experiment_001/           # Each experiment's outputs
│   │   ├── checkpoints/          # Model checkpoints
│   │   ├── logs/                 # Training logs
│   │   ├── predictions/          # Model predictions
│   │   └── metrics/              # Evaluation metrics
│   └── experiment_002/
│       ├── ...
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py              # Data pipeline tests
│   ├── test_models.py            # Model tests
│   └── test_training.py          # Training pipeline tests
│
├── docker/
│   ├── Dockerfile                # Docker configuration for reproducibility
│   └── docker-compose.yml        # Container orchestration
│
├── docs/
│   ├── data.md                   # Data documentation
│   ├── models.md                 # Model documentation
│   ├── training.md               # Training documentation
│   └── inference.md              # Inference documentation
│
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation
├── README.md                     # Project overview and setup instructions
└── .gitignore                    # Git ignore configuration