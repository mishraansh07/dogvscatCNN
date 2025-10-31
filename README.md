# 🐾 Cat vs. Dog Image Classification with Transfer Learning

A **TensorFlow/Keras** project that demonstrates high-accuracy binary image classification (**Cats vs. Dogs**) using **Transfer Learning** and **Fine-Tuning** with the **MobileNetV2** model.

This implementation loads the "Cats and Dogs" dataset, builds a robust classifier leveraging pre-trained ImageNet weights, and performs a **two-phase training process** to achieve top-tier accuracy.

---

## 🚀 Key Features

- **Data Loading:** Efficiently loads and preprocesses the dataset using `tf.keras.utils.image_dataset_from_directory`.  
- **Data Augmentation:** Real-time augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) to prevent overfitting and improve model generalization.  
- **Transfer Learning:** Utilizes **MobileNetV2**, pre-trained on ImageNet, as a convolutional base for powerful feature extraction.  
- **Two-Phase Training:**
  1. **Feature Extraction:** Train only the new classifier head while keeping the base model frozen.  
  2. **Fine-Tuning:** Unfreeze the top layers of the base model and train with a very low learning rate to adapt pre-trained weights.  
- **Evaluation:** Automatically plots accuracy and loss graphs across both phases.  
- **Prediction:** Includes functions to predict on single images or visualize predictions from validation batches.  

---

## 🧠 The Model — Two-Phase Training Strategy

### **Model Architecture**
Built using the **Keras Functional API:**

| Layer | Description |
|-------|--------------|
| **Input** | `(180, 180, 3)` image input |
| **Data Augmentation** | Random transformations for robustness |
| **Preprocessing** | MobileNetV2-specific preprocessing |
| **Base Model** | `MobileNetV2(include_top=False)` |
| **GlobalAveragePooling2D** | Flattens feature maps |
| **Dropout(0.2)** | Prevents overfitting |
| **Dense(1, 'sigmoid')** | Binary classification output |

---
