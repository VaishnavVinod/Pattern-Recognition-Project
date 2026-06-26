# Pattern-Recognition-Project

# Sports Action Recognition Framework
### Pattern Recognition Course – Phase 2
**CNN-Based Multi-Class Sports Image Classification (100 Categories)**

---

## Overview

This project implements a complete sports image classification pipeline using three models:
- **Custom CNN** — a 5-block convolutional network built from scratch
- **EfficientNetB0** — transfer learning with ImageNet pre-training + fine-tuning
- **MobileNetV2** — lightweight transfer learning baseline

Dataset: [Sports Classification – Kaggle (gpiosenka)](https://www.kaggle.com/datasets/gpiosenka/sports-classification)  
Framework: TensorFlow / Keras  
Platform: Kaggle Notebook / Google Colab

---

## Repository Structure

```
sports-action-recognition/
├── sports_classifier.py        # Main implementation (all models, evaluation, Grad-CAM)
├── proposal/
│   └── Sports_Action_Recognition_Proposal.docx
├── outputs/                    # Generated figures (after running)
│   ├── sample_images.png
│   ├── cnn_training_curves.png
│   ├── efficientnet_training_curves.png
│   ├── mobilenet_training_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── efficientnet_confusion_matrix.png
│   ├── mobilenet_confusion_matrix.png
│   ├── gradcam_sample_1.png  (through _4.png)
│   ├── roc_cnn.png
│   ├── roc_efficientnet.png
│   ├── roc_mobilenet.png
│   ├── error_analysis_cnn.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   └── final_summary.txt
└── README.md
```

---

## How to Run

### Option A — Kaggle Notebook (Recommended)

1. Go to [https://www.kaggle.com/datasets/gpiosenka/sports-classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and click **New Notebook**.
2. The dataset will be automatically available at `/kaggle/input/sports-classification/`.
3. Upload `sports_classifier.py` or paste its content into a notebook cell.
4. Enable GPU: *Settings → Accelerator → GPU T4 x2*.
5. Run all cells. Outputs are saved to `/kaggle/working/outputs/`.

### Option B — Google Colab

```python
# Mount Drive and install dependencies
from google.colab import drive
drive.mount('/content/drive')

# Download dataset via Kaggle API
!pip install kaggle
# (upload your kaggle.json token first)
!kaggle datasets download -d gpiosenka/sports-classification
!unzip sports-classification.zip -d /content/data

# Then adjust DATA_ROOT = "/content/data" in sports_classifier.py
```

### Option C — Local

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas opencv-python
# Download dataset from Kaggle, adjust DATA_ROOT, then:
python sports_classifier.py
```

---

## Models & Architecture

| Model | Blocks | Parameters | Strategy |
|---|---|---|---|
| Custom CNN | 5 Conv blocks | ~15M | Trained from scratch |
| EfficientNetB0 | EfficientNet base | ~5.3M | Frozen base → Fine-tune top 30 layers |
| MobileNetV2 | MobileNet base | ~3.5M | Frozen base → Classification head |

---

## Evaluation Outputs

- Accuracy / Loss training curves
- Confusion matrix (top-20 classes)
- Classification report (per-class precision, recall, F1)
- Macro-averaged ROC curve + AUC
- Grad-CAM visualisations (4 test images)
- Error analysis (top-10 misclassified pairs)
- Model comparison table & bar chart

---

## Dataset Information

- **Source**: [Kaggle – gpiosenka/sports-classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- **Classes**: 100 sport categories
- **Images**: ~13,493 total
- **Split**: Pre-partitioned train / valid / test directories
- **Image type**: RGB JPEG, resized to 224 × 224
