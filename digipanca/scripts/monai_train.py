import os
import csv
import numpy as np
from tqdm.auto import tqdm
import torch
from monai.data import DataLoader
import monai.transforms as mt
from monai.data import CacheDataset, load_decathlon_datalist, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from scipy import ndimage

def expand_labels(x):
        return np.concatenate(tuple([
                    ndimage.binary_dilation((x == c).astype(x.dtype), iterations=48).astype(float)
                    for c in range(5)
                ]), axis=0)

def clear_label4crop(x):
    return 0

def main():
    set_determinism(seed=42)
    data_dir = "data/prepared/"
    json_list = "data/splits/datalist.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5  # background, pancreas, tumor, artery, vein
    batch_size = 2
    roi_size = (96, 96, 96)
    lr = 1e-3
    max_epochs = 400
    val_interval = 1
    early_stop_patience = 20
    model_dir = "models/weights/monai_unet/"
    os.makedirs(model_dir, exist_ok=True)

    def make_transforms():
        deterministic = [
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), align_corners=True),
            mt.CastToTyped(keys=["image"], dtype=torch.float32),
            mt.ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0, b_max=1, clip=True),
            mt.CastToTyped(keys=["image", "label"], dtype=[np.float16, np.uint8]),
            mt.CopyItemsd(keys=["label"], times=1, names=["label4crop"]),
            mt.Lambdad(keys="label4crop", func=expand_labels, overwrite=True),
            mt.EnsureTyped(keys=["image", "label", "label4crop"]),
            mt.CastToTyped(keys=["image"], dtype=torch.float32),
            mt.SpatialPadd(keys=["image", "label", "label4crop"], spatial_size=roi_size, mode=["reflect", "constant", "constant"])
        ]

        random = [
            mt.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                spatial_size=roi_size,
                num_classes=num_classes,
                ratios=[1.0] * num_classes,
                num_samples=1
            ),
            mt.Lambdad(keys="label4crop", func=clear_label4crop),  # clean up the label4crop
            mt.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            mt.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            mt.RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            mt.RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.2),
            mt.RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
            mt.RandGaussianSmoothd(keys="image", prob=0.1),
            mt.RandGaussianNoised(keys="image", prob=0.1),
            mt.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            mt.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            mt.ToTensord(keys=["image", "label"])
        ]

        return mt.Compose(deterministic + random)

    def make_val_transforms():
        return mt.Compose([
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            mt.CastToTyped(keys=["image"], dtype=torch.float32),
            mt.ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0, b_max=1, clip=True),
            mt.CastToTyped(keys=["image", "label"], dtype=[np.float16, np.uint8]),
            mt.CastToTyped(keys=["image"], dtype=torch.float32),
            mt.EnsureTyped(keys=["image", "label"]),
            mt.ToTensord(keys=["image", "label"])
        ])

    # Load list
    tr_datalist = load_decathlon_datalist(json_list, True, "training", base_dir=data_dir)
    val_datalis = load_decathlon_datalist(json_list, True, "validation", base_dir=data_dir)
    print(f"Number of training images: {len(tr_datalist)}")
    print(f"Number of validation images: {len(val_datalis)}")

    train_transforms = make_transforms()
    val_transforms = make_val_transforms()

    train_ds = CacheDataset(
        data=tr_datalist,
        transform=train_transforms,
        cache_rate=0.1,
        num_workers=4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    val_ds = CacheDataset(
        data=val_datalis,
        transform=val_transforms,
        cache_rate=0.1,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch"
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Early stopping
    best_metric = -1
    best_metric_epoch = -1
    epochs_no_improve = 0

    # Save loss and metric values for plotting
    train_losses = []
    val_losses = []
    val_dice_scores = []

    print("🚀 Training...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_bar = tqdm(train_loader, desc=f"Training [{epoch+1}/{max_epochs}]", leave=False)
        
        for batch in train_bar:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        print(f"✅ Epoch {epoch+1} - Mean loss: {epoch_loss / len(train_loader):.4f}")
        epoch_train_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            dice_metric.reset()
            val_loss_epoch = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc="🧪 Validating", leave=False)
                for val_data in val_bar:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_images, roi_size, 1, model)
                    val_outputs = torch.softmax(val_outputs, 1)
                    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    val_labels = torch.unsqueeze(val_labels, 1)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_loss = loss_fn(val_outputs, val_labels)
                    val_loss_epoch += val_loss.item()

            val_loss_epoch /= len(val_loader)
            val_losses.append(val_loss_epoch)

            mean_dice, _ = dice_metric.aggregate()
            val_dice_scores.append(mean_dice.item())
            print(f"📊 Validation Mean Dice: {mean_dice:.4f}")
            dice_metric.reset()

            # Guardar modelo si mejora
            if mean_dice > best_metric:
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                print("💾 New best model saved.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"⏳ Model not improved in {epochs_no_improve} epoch(s).")

            # Early stopping
            if epochs_no_improve >= early_stop_patience:
                print("🛑 Early stopping.")
                break

        # Guardar último checkpoint cada epoch
        torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))

    print(f"🎯 Training finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    metrics_path = os.path.join(model_dir, "metrics.csv")
    with open(metrics_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                train_losses[i],
                val_losses[i],
                val_dice_scores[i]
            ])

    print(f"📁 Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Run the training script
    main()