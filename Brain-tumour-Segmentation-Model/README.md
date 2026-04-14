# BraTS 2023 Glioma Segmentation using SegResNet

This project trains a **3D brain tumor segmentation model** on the **BraTS 2023 GLI dataset** using **MONAI** and **PyTorch**.

The model is based on **SegResNet** and is trained on 3D MRI volumes to segment different glioma tumor regions. The training pipeline is designed to be practical and stable for **Windows + NVIDIA CUDA**, with settings kept safe for an **RTX 4050 8GB GPU**.

This script includes:
- 3D medical image loading and preprocessing
- patch-based training for memory efficiency
- data augmentation
- validation with BraTS region-wise Dice scores
- early stopping
- checkpoint saving and resume support
- optional external validation prediction export in NIfTI format

---

## Project Goal

The goal of this project is to segment glioma tumor regions from multi-modal 3D MRI scans.

Instead of only predicting whether a tumor is present, the model predicts a **voxel-wise segmentation map**, which means it tries to mark the tumor regions throughout the full 3D brain volume.

This project focuses on the **BraTS 2023 GLI task**, where the model learns from four MRI modalities and produces multi-class tumor segmentation.

---

## What This Project Does

This training pipeline:

1. loads 3D MRI scans from the BraTS dataset
2. reads all required MRI modalities for each case
3. preprocesses the scans into a standard format
4. trains a 3D SegResNet model
5. validates the model using BraTS tumor-region Dice scores
6. saves the best checkpoints during training
7. supports resuming training from the last saved checkpoint
8. can export predictions for external validation cases


**Why This Project Is Useful**

This project demonstrates a complete 3D medical image segmentation workflow.

It includes:

>multi-modal MRI handling

>3D preprocessing

>augmentation

>memory-aware patch training

>model checkpointing

>validation

>resume support

>export of native-space predictions

So this is not only a simple script.
It is a more complete end-to-end deep learning pipeline for brain tumor segmentation.



**Current Limitations**

training requires a CUDA-capable GPU for practical speed

full 3D segmentation is computationally expensive

performance depends on dataset quality and preprocessing consistency

results may vary on unseen clinical data

this project is intended for educational and research purposes only

it is not meant for direct clinical decision-making


**Future Improvements**

Possible future improvements include:

adding an inference script for single-case prediction

adding visualization of predicted segmentation masks

adding test set evaluation summaries

comparing SegResNet with UNet or SwinUNETR

tuning patch size and ROI size further

adding experiment tracking with plots

deploying a simple interface for prediction

## MRI Modalities Used

Each BraTS case contains four MRI modalities:

- **T1n** – native T1-weighted MRI
- **T1c** – contrast-enhanced T1-weighted MRI
- **T2f** – T2-FLAIR MRI
- **T2w** – T2-weighted MRI

These four modalities are combined and used as the input to the model.

So the model input has **4 channels**, one for each MRI type.

---

## Target Output

The model predicts segmentation labels for tumor regions in 3D.

BraTS evaluation commonly focuses on these three regions:

- **WT (Whole Tumor)**  
  all tumor-related regions together

- **TC (Tumor Core)**  
  the core tumor region excluding edema

- **ET (Enhancing Tumor)**  
  the actively enhancing tumor region

These regions are used during validation to measure performance.

---

## Model Used

This project uses **SegResNet**, a 3D convolutional neural network architecture from MONAI designed for medical image segmentation.

### Why SegResNet?
SegResNet is well suited for 3D medical imaging because it:
- works directly on volumetric MRI data
- captures spatial context in all three dimensions
- is effective for segmentation tasks
- is commonly used in medical imaging workflows

---

## Why Patch-Based Training Is Used

3D MRI volumes are very large and require a lot of GPU memory.

To make training possible on an 8GB GPU, this project uses **patch-based training**.

That means:
- the full MRI volume is not processed at once
- smaller 3D patches are cropped from the image
- the model trains on those patches

This makes training more memory-efficient and practical.

The patch size used in this project is:
(96, 96, 64)



**Main Features**

1> 3D glioma segmentation using SegResNet

2> built with PyTorch and MONAI

3> supports BraTS 2023 GLI dataset structure

4>uses four MRI modalities as input

5>patch-based training for safer GPU memory usage

6>safe defaults for Windows + NVIDIA CUDA

7>mixed precision training with AMP

8>gradient accumulation for stable training on limited VRAM

9>augmentation for better robustness

10>validation using BraTS region-wise Dice scores:

11>WT,TC,ET

12>early stopping with patience

13>multiple best checkpoints:

14>best mean score,guarded mean score, ET score

15>resume support from last checkpoint

>optional export of external validation predictions in NIfTI format


**Dataset Structure**

The script expects the dataset to be arranged inside a data/ folder.

data/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│   ├── BraTS-GLI-00001-000/
│   │   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   │   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   │   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   │   ├── BraTS-GLI-00001-000-t2w.nii.gz
│   │   └── BraTS-GLI-00001-000-seg.nii.gz
│   └── ...
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
│   ├── BraTS-GLI-xxxxx/
│   │   ├── *-t1n.nii.gz
│   │   ├── *-t1c.nii.gz
│   │   ├── *-t2f.nii.gz
│   │   └── *-t2w.nii.gz
│   └── ...


**Folder and Output Structure**

A typical project structure can look like this:

brats-glioma-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── train_brats_segnet.py
├── data/
│   ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│   └── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
└── outputs/
    ├── splits.json
    ├── metrics.jsonl
    ├── best_mean_any.pt
    ├── best_mean_guarded.pt
    ├── best_et.pt
    ├── last.pt
    └── extval_predictions/


**How the Training Pipeline Works**

1. Dependency Check

At the beginning, the script checks that the required packages are installed:

NumPy

PyTorch

MONAI

NiBabel

If something is missing, it gives an install instruction.

2. Path Resolution

The script tries to locate the project root and dataset folders safely.

This helps avoid common path bugs such as incorrect nested folder paths.

It builds paths for:

training data

validation data

outputs

checkpoints

split files

exported predictions

3. Data Discovery

The script scans the dataset folders and builds case dictionaries.

For each case, it searches for:

T1n

T1c

T2f

T2w

segmentation label (for training data)

Cases with missing required files are skipped.

4. Stable Train/Validation Split

The script creates a fixed internal train/validation split and saves it in:
>>outputs/splits.json


5. Preprocessing

The script applies several preprocessing steps:

load NIfTI volumes

ensure channel-first format

remap label 4 to 3 when needed

orient scans to RAS

resample spacing to 1.0 × 1.0 × 1.0

normalize intensity channel-wise

crop foreground to remove unnecessary empty space

This helps standardize the input data before training.

6. Data Augmentation

The training pipeline uses augmentation to improve robustness.

Possible augmentations include:

random flips

random 90-degree rotations

random intensity scaling

random intensity shifting

Gaussian noise

random affine transforms

optional realism-based transforms such as:

bias field

contrast changes

smoothing

Gibbs noise

These help the model generalize better to different scans.

7. Patch Sampling

Instead of training on the full MRI volume, the script samples patches using positive and negative label guidance.

This helps:

focus training on meaningful tumor regions

reduce memory usage

make training feasible on smaller GPUs

8. Model Training

The script trains a 3D SegResNet with:

4 input channels

4 output classes

The optimizer used is:

AdamW

The learning rate scheduler used is:

ReduceLROnPlateau

The loss function used is:

DiceFocalLoss

This combines region overlap learning with focal behavior for harder examples.

9. Mixed Precision and Memory Safety

To make training safer and faster on CUDA, the script uses:

automatic mixed precision (AMP)

gradient scaling

gradient accumulation

gradient clipping

These are useful when training large 3D models on limited GPU memory.

10. Validation

After each epoch, the script evaluates the model on the validation set.

It computes Dice scores for:

WT

TC

ET

It calculates both:

raw prediction scores

post-processed prediction scores

This gives a better understanding of model performance.

11. Post-Processing

The script includes post-processing to clean small enhancing tumor speckles.

This helps reduce small noisy ET predictions that may not be meaningful.

12. Checkpoint Saving

The script saves several checkpoints:

best_mean_any.pt
best mean validation score overall

best_mean_guarded.pt
best mean score with extra protection against ET collapse

best_et.pt
best enhancing tumor score

last.pt
most recent checkpoint for resume

This makes training safer and allows restarting without losing progress.

13. Resume Support

If last.pt exists, the script can resume training automatically.

This restores:

model weights

optimizer state

scheduler state

scaler state

epoch number

patience counter

best scores

This is helpful if training is interrupted.

14. External Validation Export

The script can also export predictions for external validation cases.

Predictions are:

inverted back to native image space

saved with original affine information

written in NIfTI format

This makes the outputs suitable for further use, including Synapse-style submission workflows.

Loss Function

This project uses DiceFocalLoss.

Why this loss is useful

In medical segmentation:

classes are often imbalanced

tumor regions can be small

overlap quality matters a lot

DiceFocalLoss helps because:

Dice part improves overlap

focal behavior helps focus on harder regions

The script also applies class weights, including a higher weight for the enhancing tumor class.

Validation Metrics

The main validation metric used in training is Dice.

The script computes BraTS region-wise Dice for:

WT – Whole Tumor

TC – Tumor Core

ET – Enhancing Tumor

The average of these scores is used as an important performance indicator.

Optional native-space metrics such as Hausdorff distance and volume error are also supported, though they are more expensive to compute.

Training Configuration

Important settings used in this script include:

patch size: 96 x 96 x 64

validation ROI size: 96 x 96 x 64

batch size: 1

gradient accumulation steps: 2

max epochs: 150

minimum epochs: 15

patience: 5

learning rate: 2e-4

weight decay: 1e-4

optimizer: AdamW

loss: DiceFocalLoss

model: SegResNet

mixed precision: enabled

CUDA-safe settings for RTX 4050 8GB

These values were chosen to balance segmentation performance and hardware safety.



**SUMMARY**
This project implements a 3D glioma segmentation pipeline on the BraTS 2023 GLI dataset using MONAI and PyTorch. It uses a SegResNet model with patch-based training, mixed precision, checkpointing, validation with BraTS Dice metrics, and optional export of native-space predictions for external validation.


**Author**

Ayush Patel

HAVE A WONDERFUL DAY AHEAD,AND BE THE BEST









