# Cat vs Dog Image Classifier with Transfer Learning

A deep learning image classifier that distinguishes between cats and dogs using **transfer learning** with MobileNetV2 and fine-tuning techniques.

## Results

- **Final Validation Accuracy:** 98.20%
- **Architecture:** MobileNetV2 (pre-trained on ImageNet)
- **Training Strategy:** Two-phase approach with fine-tuning

### Training Performance

![Training Results](C:\Users\anshm\OneDrive\Desktop\Git clone\Trainning Visuals.png)

**Phase 1 - Classifier Training (10 epochs):**
- Starting accuracy: 75.86% → Final: 96.68%
- Validation accuracy reached: 98.20%

**Phase 2 - Fine-Tuning (10 additional epochs):**
- Unfroze top layers of MobileNetV2
- Low learning rate (1e-5) for careful weight adjustment
- Maintained high validation accuracy: 98.20%

## Features

- **Transfer Learning:** Leverages MobileNetV2 trained on ImageNet
- **Data Augmentation:** Random flips, rotations, and zoom for better generalization
- **Two-Phase Training:**
  1. Train only the classifier head
  2. Fine-tune top layers of the base model
- **High Accuracy:** Achieves 98%+ validation accuracy
- **Optimized Performance:** Uses TensorFlow's data pipeline optimization

## Architecture

```
Input (180x180x3)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
MobileNetV2 Preprocessing
    ↓
MobileNetV2 Base (frozen initially, then partially unfrozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.2)
    ↓
Dense (1, sigmoid) → Binary classification
```

## Requirements

```bash
tensorflow>=2.12.0
matplotlib>=3.5.0
```

## Quick Start

### Installation

```bash
pip install tensorflow matplotlib
```

### Run Training

```python
python train.py
```

The script will automatically:
1. Download the cats and dogs dataset
2. Train the classifier for 10 epochs
3. Fine-tune for an additional 10 epochs
4. Display training curves
5. Show final validation accuracy

### Predict on Custom Images

```python
# Use the predict_image function
predict_image("path/to/your/image.jpg")
```

## Training Details

### Phase 1: Classifier Training
- **Epochs:** 10
- **Learning Rate:** Default Adam (1e-3)
- **Frozen Layers:** Entire MobileNetV2 base
- **Trainable Parameters:** Only the top Dense layer

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 75.86%    | 97.30%  | 0.4875     | 0.0926   |
| 5     | 96.40%    | 98.00%  | 0.0998     | 0.0525   |
| 10    | 96.68%    | 98.20%  | 0.0818     | 0.0451   |

### Phase 2: Fine-Tuning
- **Epochs:** 10 (11-20)
- **Learning Rate:** 1e-5 (10x smaller)
- **Unfrozen Layers:** Layers 100+ of MobileNetV2
- **Strategy:** Careful adjustment of pre-trained weights

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 10    | 81.32%    | 97.50%  | 0.4155     | 0.0720   |
| 20    | 96.98%    | 98.20%  | 0.0780     | 0.0471   |

## Key Implementation Details

### Data Augmentation
```python
RandomFlip("horizontal")
RandomRotation(0.1)
RandomZoom(0.1)
```

### Fine-Tuning Strategy
- Freeze bottom 100 layers of MobileNetV2
- Unfreeze top layers for domain-specific learning
- Use very low learning rate to prevent catastrophic forgetting

### Dataset
- **Source:** Cats and Dogs Filtered Dataset
- **Training Images:** ~2000
- **Validation Images:** ~1000
- **Image Size:** 180x180 pixels
- **Batch Size:** 32

## Project Structure

```
.
├── train.py                    # Main training script
├── README.md                   # This file
├── training_plots.png          # Training visualization
└── cats_and_dogs_filtered/     # Dataset (auto-downloaded)
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── validation/
        ├── cats/
        └── dogs/
```

## Learning Outcomes

This project demonstrates:
- Transfer learning with pre-trained models
- Fine-tuning strategies
- Data augmentation techniques
- Binary classification with CNNs
- TensorFlow/Keras best practices
- Model evaluation and visualization

## License

This project is open source and available for educational purposes.

## Acknowledgments

- MobileNetV2 pre-trained weights from ImageNet
- Dataset from TensorFlow/Google ML Education
- Built with TensorFlow and Keras

---

**Note:** The model achieves 98.20% validation accuracy, demonstrating the power of transfer learning for image classification tasks!
