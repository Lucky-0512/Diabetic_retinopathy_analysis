
Claude finished the response

netravision.ipynb
ipynb

hey analysie this ipunb , and egenrate a readme markdown file explain the entire project and give me the file output

22:42

Viewed a file, ran a command
Viewed a file, ran a command
Now I have a thorough picture of the entire notebook. Let me generate the README.


Created a file, read a file
Created a file, read a file
Here's your README — fully generated from the notebook analysis. Here's a quick summary of what I covered:

What the notebook does (end-to-end):

Data pipeline — Downloads the APTOS 2019 dataset from Kaggle, organizes it in Google Drive, and builds a custom DRDataset PyTorch class with on-the-fly preprocessing
Preprocessing — Graham normalization + CLAHE on the L-channel (LAB space) — the same combo used in top DR research papers
Model — EfficientNet-B4 with a custom 5-class head, trained with AdamW + mixed precision (AMP) for 10 epochs, saving the best checkpoint by validation accuracy
Grad-CAM — Applied on the last EfficientNet block to generate attention heatmaps over the retina
Reporting — Two modes: batch inference over 25 val images (saves .txt reports + PNG heatmaps), and a single user-upload mode that spits out a full PDF report (severity, confidence, clinical action, embedded heatmap) downloaded directly from Colab
The README includes the full project structure, the 5-class severity table with clinical actions, architecture details, dependency list, step-by-step run guide, and the key design decision rationale table. Drop this straight into your GitHub repo root.


Readme
Document · MD 


Claude Fable 5 is currently unavailable.
Learn more(opens in new tab)





Claude is AI and can make mistakes. Please double-check responses.


Readme · MD
# 👁️ NetraVision — Diabetic Retinopathy Screening System
 
> An AI-powered retinal image analysis pipeline that screens for Diabetic Retinopathy (DR), classifies severity across 5 grades, generates Grad-CAM explainability heatmaps, and produces downloadable PDF clinical reports — all in a Google Colab notebook.
 
---
 
## 🧠 Overview
 
NetraVision is a deep learning project built for automated screening of **Diabetic Retinopathy (DR)** from fundus retinal images. It leverages a fine-tuned **EfficientNet-B4** model trained on the **APTOS 2019 Blindness Detection** dataset from Kaggle. The system classifies images into one of 5 DR severity levels and uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually highlight the retinal regions that influenced the model's decision — making predictions both accurate and explainable.
 
The end-to-end workflow covers:
 
- Retinal image preprocessing with Graham normalization and CLAHE
- Custom PyTorch dataset and dataloader pipeline
- EfficientNet-B4 fine-tuning with mixed precision training
- Grad-CAM heatmap overlay generation
- Automated PDF clinical report generation per image
---
 
## 📁 Project Structure
 
```
DR_Project/                        ← Root on Google Drive
├── dataset/
│   ├── train_1.csv                ← Training labels (id_code, diagnosis)
│   ├── valid.csv                  ← Validation labels
│   ├── train_images/train_images/ ← Training retinal PNGs
│   └── val_images/val_images/     ← Validation retinal PNGs
│
├── models/
│   └── best_dr_model.pth          ← Saved best checkpoint (by val accuracy)
│
└── outputs/
    ├── reports/                   ← Per-image text reports (batch mode)
    ├── gradcam/                   ← Grad-CAM visualizations (batch mode)
    └── user_uploads/
        ├── uploaded_original.png  ← User-uploaded fundus image (safe copy)
        ├── gradcam_visualization.png ← 3-panel comparison image
        └── DR_Report.pdf          ← Final downloadable clinical PDF report
```
 
---
 
## 🩺 DR Severity Classification
 
The model predicts one of 5 clinical classes:
 
| Class | Severity Level | Severity Score | Suggested Action |
|-------|---------------|---------------|-----------------|
| 0 | No Diabetic Retinopathy | 0 / 4 | Annual screening recommended |
| 1 | Mild DR | 1 / 4 | Follow-up in 6–12 months |
| 2 | Moderate DR | 2 / 4 | Ophthalmologist referral recommended |
| 3 | Severe DR | 3 / 4 | Urgent evaluation required |
| 4 | Proliferative DR | 4 / 4 | Immediate specialist treatment — high vision loss risk |
 
---
 
## 🔬 Technical Architecture
 
### Model — EfficientNet-B4
 
- **Backbone:** `EfficientNet-B4` (pretrained from `efficientnet_pytorch`)
- **Custom Head:**
```
  Linear(1792 → 512) → ReLU → Dropout(0.5) → Linear(512 → 5)
```
- **Output:** 5-class softmax probability distribution
- **Training Objective:** CrossEntropyLoss
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- **Mixed Precision:** PyTorch AMP (`GradScaler` + `autocast`)
- **Epochs:** 10 (with best model checkpointing by validation accuracy)
- **Batch Size:** 8
### Preprocessing Pipeline
 
Each fundus image goes through a multi-step preprocessing pipeline before being fed to the model:
 
1. **BGR → RGB conversion** (via OpenCV)
2. **Resize** to 512×512
3. **Graham Normalization** — weighted blend of the original and a Gaussian-blurred version to enhance local contrast and remove uneven illumination:
```python
   cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), size/30), -4, 128)
```
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — applied on the L-channel in LAB color space to further enhance fine retinal features
5. **Normalization** — pixel values scaled to `[0.0, 1.0]`
### Custom Dataset — `DRDataset`
 
A custom `torch.utils.data.Dataset` class that:
- Filters the CSV dataframe to only include rows where the corresponding image file physically exists on disk
- Applies the full preprocessing pipeline per image at load time
- Returns `(image_tensor [C×H×W], label_tensor)` pairs
- Gracefully falls back to the next valid image on load failure
---
 
## 🔍 Explainability — Grad-CAM
 
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is applied after prediction to generate a heatmap that highlights which regions of the retina were most important for the model's classification decision.
 
- **Target Layer:** Last block of EfficientNet-B4 (`model._blocks[-1]`)
- **Library:** `pytorch_grad_cam`
- **Output:** A color heatmap overlaid on the original retinal image, showing the model's "attention" regions
Each result is displayed as a **3-panel comparison**:
 
```
[ Original Image ] | [ Graham + CLAHE Preprocessed ] | [ Grad-CAM Heatmap ]
```
 
---
 
## 📋 Report Generation
 
### Batch Mode (Validation Set)
Runs inference on 25 randomly sampled validation images, saving for each:
- A 3-panel Grad-CAM visualization PNG
- A plain text `.txt` clinical report containing image ID, predicted severity, confidence score, and suggested action
### Single Image Mode (User Upload)
The notebook supports uploading a custom fundus image via `google.colab.files.upload()` and then:
1. Copies the uploaded file to a permanent location on Drive
2. Runs full preprocessing → inference → Grad-CAM pipeline
3. Generates the 3-panel visualization
4. Creates a **PDF report** using `reportlab` with:
   - Predicted severity level and score
   - Model confidence
   - Suggested clinical action
   - The Grad-CAM visualization image embedded in the PDF
5. Downloads the PDF directly to the user's machine via `files.download()`
---
 
## 🛠️ Dependencies
 
```bash
# Deep Learning
torch
efficientnet_pytorch
 
# Image Processing
opencv-python
albumentations
 
# Explainability
grad-cam
 
# Data
kaggle
kagglehub
pandas
numpy
 
# Visualization
matplotlib
 
# Report Generation
reportlab
 
# Utilities
tqdm
```
 
---
 
## 🚀 How to Run
 
### Prerequisites
- Google Colab with GPU runtime enabled (T4 or better recommended)
- Google Drive mounted at `/content/drive`
- Kaggle API credentials for dataset download
### Step-by-Step
 
**1. Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
 
**2. Verify GPU**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```
 
**3. Install Dependencies**
```bash
pip install efficientnet_pytorch albumentations grad-cam opencv-python kaggle reportlab
```
 
**4. Download Dataset** *(run once)*
```python
import kagglehub
path = kagglehub.dataset_download("mariaherrerot/aptos2019")
```
Copy the dataset to `DR_Project/dataset/` in your Drive.
 
**5. Run Training**
- Set up paths → create `DRDataset` instances → initialize model, loss, and optimizer → run the training loop for 10 epochs.
- The best model checkpoint is saved automatically to `DR_Project/models/best_dr_model.pth`.
**6. Load Trained Model & Run Inference**
- Reload the best checkpoint
- Run Grad-CAM on a batch of validation images or a custom upload
**7. Generate Report**
- Upload your own fundus image when prompted
- The notebook will produce and auto-download a `DR_Report.pdf`
---
 
## 📊 Dataset
 
**APTOS 2019 Blindness Detection** (Kaggle)
- Fundus retinal photographs from rural India
- Labels: 0–4 DR severity grades
- Images in PNG format, variable resolution (resized to 512×512 during preprocessing)
---
 
## 📌 Key Design Decisions
 
| Decision | Rationale |
|----------|-----------|
| EfficientNet-B4 | Strong accuracy-to-parameter ratio; well-suited for medical imaging |
| Graham + CLAHE preprocessing | Standard in DR literature; improves visibility of fine vascular features |
| Mixed precision (AMP) | Faster training on Colab GPU, reduced VRAM usage |
| Grad-CAM on last block | Provides the highest-level semantic feature maps before the classifier head |
| AdamW optimizer | Better weight decay handling than Adam; reduces overfitting |
| PDF via ReportLab | Lightweight, no browser dependency; embeds images cleanly |
 
---
 
## ⚠️ Disclaimer
 
NetraVision is a **research and educational project**. The predictions produced by this system are **not a substitute for professional medical diagnosis**. Always consult a qualified ophthalmologist for clinical decisions.
 
---
 
## 👤 Author
 
Built by **Sai** as part of an AI/ML portfolio project exploring computer vision for medical imaging.
 

