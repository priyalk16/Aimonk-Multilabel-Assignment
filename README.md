Aimonk Multilabel Image Classification Assignment
Overview

This project implements a deep learning based multi-label image classification system using PyTorch.

Each image in the dataset contains four independent attributes. An image can have multiple attributes present at the same time. The labels are provided in a labels.txt file where:

1 indicates the attribute is present

0 indicates the attribute is absent

NA indicates that the attribute information is not available

The goal was to train a model that can predict all applicable attributes for a given image.

Approach
Model Selection

I used ResNet18 pretrained on ImageNet as the base model.
The final fully connected layer was modified to output 4 values corresponding to the 4 attributes.

The backbone layers were frozen and only the final classification layer was fine-tuned. This ensures that the model benefits from pretrained ImageNet features instead of training from scratch.

Why Multi-Label?

This is a multi-label problem because:

Each image can have more than one attribute.

The attributes are independent.

Therefore, Softmax cannot be used.

Instead:

Sigmoid activation is applied to each output.

BCEWithLogitsLoss is used.

Handling Missing Labels (NA)

Some attribute values in the dataset are marked as NA.

These were handled by:

Creating a mask tensor.

Computing loss only for valid attribute entries.

Ensuring that images are not discarded due to missing attribute information.

This allows the model to use all available data while ignoring only the missing attribute positions.

Dataset Imbalance

The dataset appears to be imbalanced across attributes.

The following techniques were considered:

Class-weighted BCE using pos_weight

Focal Loss

Weighted sampling

Attribute-wise threshold tuning

During experimentation, aggressive weighting caused instability due to the dataset size. Therefore, the final implementation uses masked BCEWithLogitsLoss for stable training. Imbalance-aware methods are suggested as future improvements.

Data Preprocessing

The following transformations were applied:

Resize to 224×224

Random horizontal flip

Conversion to tensor

Normalization using ImageNet mean and standard deviation

These preprocessing steps ensure compatibility with pretrained ResNet18 weights.

Training Details

Optimizer: Adam

Learning rate: 1e-5

Batch size: 16

Epochs: 5

Loss function: Masked BCEWithLogitsLoss

Training loss is recorded per iteration and plotted.

The loss curve includes:

X-axis: iteration_number

Y-axis: training_loss

Title: Aimonk_multilabel_problem

Files Generated

After training, the following files are saved:

models/aimonk_model.pth – trained model weights

models/loss_curve.png – training loss curve

How to Run
Train the Model
python train.py

This will:

Train the model

Save weights

Generate loss curve

Run Inference
python inference.py

This will:

Load trained weights

Predict attributes for a sample image

Print the attributes that are present

Project Structure
AI-Monk Assignment-Priyal/
│
├── images/
├── models/
│   ├── aimonk_model.pth
│   └── loss_curve.png
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── labels.txt
└── README.md
Observations

The training loss decreases consistently over epochs.

The model successfully predicts multiple attributes per image.

Proper handling of NA values was critical for stable training.

A label parsing issue (automatic conversion of “NA” to NaN) was identified and corrected during development.

Possible Improvements

Unfreeze deeper ResNet layers for stronger fine-tuning

Add validation split and evaluation metrics (Precision, Recall, F1-score)

Apply focal loss or weighted BCE more robustly

Tune decision threshold per attribute

Conclusion

The final solution satisfies the assignment requirements:

Deep learning based multi-label classification

Fine-tuning on ImageNet pretrained weights

Proper handling of missing labels

Loss curve generation

Model weight saving

Working inference pipeline

Clean and modular code structure