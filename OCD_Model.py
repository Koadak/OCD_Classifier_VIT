import tensorflow as tf
import SimpleITK as sitk
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from einops import rearrange
import torch
from transformers import ViTModel, ViTFeatureExtractor

# Path to image directory
path = r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\images'

# Lists to store GM and FA images with labels
X_GM, y_GM = [], []
X_FA, y_FA = [], []

# Function to load NIfTI images with labels
def load_nifti_images(prefix, start_idx, num_images, storage_list, label_list, label):
    for i in range(num_images):
        file_path = f"{path}/{prefix}{start_idx + i}.nii"
        try:
            # Read and convert NIfTI to numpy
            img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(img)

            # Resize each slice (keeping original resolution)
            resized_slices = [scipy.ndimage.zoom(slice_, 1.0) for slice_ in img_array[:90]]

            # Append processed slices and label
            storage_list.append(resized_slices)
            label_list.append(label)

        except Exception as e:
            continue
            # print(f"Skipping {file_path}: {e}")  # Print the skipped file

# Load GM and FA images with matching indices
load_nifti_images("GM", 1000, 500, X_GM, y_GM, label=0)
load_nifti_images("FA", 1000, 500, X_FA, y_FA, label=0)

load_nifti_images("GM", 2000, 2000, X_GM, y_GM, label=1)
load_nifti_images("FA", 2000, 2000, X_FA, y_FA, label=1)

# Ensure matching pairs
X_combined = [np.stack((gm, fa), axis=1) for gm, fa in zip(X_GM, X_FA)]
y_combined = y_GM  # Labels remain the same

# Convert list to numpy array for model processing
X_combined = np.array(X_combined)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.1, random_state=34)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Select an image and slice for display
image_index = 25
slice_index = 45
#
# if image_index >= len(X_combined):
#     print(f"Error: image_index {image_index} is out of range! Max index: {len(X_combined)-1}")
#     exit()
#
# # Display GM and FA images stacked
# plt.subplot(1, 2, 1)
# plt.imshow(X_combined[image_index][0][slice_index], cmap='gray')
# plt.title(f"GM Slice {slice_index}")
#
# plt.subplot(1, 2, 2)
# plt.imshow(X_combined[image_index][1][slice_index], cmap='gray')
# plt.title(f"FA Slice {slice_index}")
#
# plt.show()

import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, Spacingd, Orientationd, ScaleIntensityd, ResizeWithPadOrCropd, ToTensord
)
from monai.networks.nets import SwinUNETR

# Define preprocessing
transforms = Compose([
    ScaleIntensityd(keys=["image"]),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=(96, 96, 96)),
    ToTensord(keys=["image"]),
])

# Convert dataset into dictionary format for MONAI
data_dicts = [{"image": np.stack([gm, fa], axis=0), "label": label} for (gm, fa, label) in zip(X_GM, X_FA, y_GM)]

dataset = monai.data.CacheDataset(data=data_dicts, transform=transforms)

# Data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)



############################

# Load 3D Swin UNETR Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=2,  # GM + FA as channels
    out_channels=768,  # Feature vector size
).to(device)

# Extract features from the model
def extract_3d_features(image):
    image = image.to(device)
    with torch.no_grad():
        features = model(image)  # Extract embeddings
    return features.mean(dim=(2, 3, 4))  # Global average pooling over spatial dimensions

# Extract features from a sample scan
sample = next(iter(dataloader))["image"].to(device)
features = extract_3d_features(sample)
print(f"Extracted 3D feature shape: {features.shape}")  # Expected (1, 768)

# Store extracted features and labels
all_features = []
all_labels = []

# Iterate through the DataLoader
for batch in dataloader:
    image = batch["image"].to(device)  # Move image to GPU
    with torch.no_grad():  # Disable gradient calculation for efficiency
        features = model(image)  # Extract embeddings

    # Apply global average pooling over spatial dimensions
    features = features.mean(dim=(2, 3, 4))  # Shape: (batch_size, 768)

    # Store features and corresponding labels
    all_features.append(features.cpu().numpy())  # Move to CPU for storage
    all_labels.append(batch["label"].cpu().numpy())  # Move labels to CPU

# Convert to NumPy arrays
all_features = np.vstack(all_features)  # Shape: (num_samples, 768)
all_labels = np.array(all_labels).reshape(-1)  # Flatten labels

# Print dataset stats
print(f"Extracted features shape: {all_features.shape}")
print(f"Labels shape: {all_labels.shape}")

# Save features and labels (optional)
np.save("features.npy", all_features)

# DEBUG: Check label values before saving
print(f"Label sample: {all_labels[:10]}")  # Print first 10 labels to verify

# Save labels properly
labels_path = r"C:\Users\jgdga\PycharmProjects\OCD_Tester2\labels.npy"
np.save(labels_path, all_labels)

print(f"Labels saved to: {labels_path}")
