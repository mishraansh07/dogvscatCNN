Cat vs. Dog Image Classifier with Transfer Learning

This project demonstrates how to build a high-accuracy image classifier using transfer learning with TensorFlow and Keras. It fine-tunes a pre-trained MobileNetV2 model to distinguish between images of cats and dogs.

The script implements a two-phase training process:

Feature Extraction: Training a new classifier head on top of the frozen, pre-trained base model.

Fine-Tuning: Unfreezing the top layers of the base model and continuing training with a very low learning rate to further improve accuracy.

Key Features

Framework: TensorFlow 2.x / Keras

Model: MobileNetV2 (pre-trained on ImageNet)

Technique: Transfer Learning & Fine-Tuning

Data Augmentation: Uses Keras preprocessing layers for random flips, rotations, and zooms to improve model robustness.

Visualization:

Plots the training and validation accuracy/loss curves.

Displays a 5x5 grid of validation images with their predicted and true labels.

Dataset

The script automatically downloads and extracts the "Cats and Dogs filtered" dataset provided by Google.

Source: https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

Content:

Training Set: 1000 cat images, 1000 dog images

Validation Set: 500 cat images, 500 dog images

Methodology

1. Data Pipeline

Images are loaded from their directories using tf.keras.utils.image_dataset_from_directory.

The prefetch method is used to optimize the data loading pipeline for performance.

Important: Input images are not normalized at this stage, as the MobileNetV2 model has its own specific preprocess_input layer that will be included in the model.

2. Model Architecture

The model is built using the Keras Functional API:

Input Layer: Expects images of size (180, 180, 3).

Data Augmentation: Sequential model with RandomFlip, RandomRotation, and RandomZoom.

Preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input layer to normalize images according to MobileNetV2's requirements.

Base Model: MobileNetV2 (with include_top=False) with its weights frozen. This acts as a feature extractor.

Classifier Head:

GlobalAveragePooling2D: Flattens the feature maps from the base model.

Dropout(0.2): Adds regularization to prevent overfitting.

Dense(1, activation='sigmoid'): The final output layer for binary (Cat/Dog) classification.

3. Training Process

Phase 1: Train Classifier Head

The base model (base_model.trainable = False) is completely frozen.

The model is compiled with the adam optimizer and binary_crossentropy loss.

The model is trained for 10 epochs. This trains only the weights of the new classifier head, adapting it to the new dataset without altering the pre-trained features.

Phase 2: Fine-Tuning

The entire base model is unfrozen (base_model.trainable = True).

To prevent destroying the learned features, all layers except the top 54 (from layer 100 onwards) are frozen again. This allows the model to adjust the high-level feature detectors.

The model is re-compiled with a very low learning rate (Adam(learning_rate=1e-5)) to make small, careful adjustments.

Training continues for an additional 10 epochs, starting from the end of Phase 1.

How to Run

Requirements

You will need Python with the following libraries installed:

tensorflow

matplotlib

You can install them using pip:

pip install tensorflow matplotlib


Execution

Save the code from the script as a Python file (e.g., train_cats_vs_dogs.py).

Run the script from your terminal:

python train_cats_vs_dogs.py


The script will:

Download and extract the dataset (this only happens the first time).

Print the model summary.

Begin Phase 1 training, printing progress for 10 epochs.

Begin Phase 2 fine-Tuning, printing progress for another 10 epochs.

Print the final validation accuracy.

Expected Results

After training, the script will generate two matplotlib plots:

Accuracy and Loss Plots: A window will appear showing two subplots:

Training vs. Validation Accuracy

Training vs. Validation Loss

A vertical dashed line indicates the switch from initial training to fine-tuning. This helps visualize the "boost" in performance from the fine-tuning step.

Prediction Grid: A 5x5 grid displaying the first 25 images from a validation batch.

Each image title shows the Predicted Label (with confidence) and the True Label.

The title color is green for a correct prediction and red for an incorrect one.
