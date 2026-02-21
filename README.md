#  Aimonk Multilabel Image Classification Assignment

##  Problem Statement

This project implements a **deep learning-based multi-label image classification system** using PyTorch.

Each image contains **4 independent attributes**, and multiple attributes can be present simultaneously.

Label meanings:

- `1` → Attribute is present  
- `0` → Attribute is absent  
- `NA` → Attribute information not available  

---

##  Model Architecture

- **Backbone:** ResNet18
- **Pretrained Weights:** ImageNet
- **Framework:** PyTorch
- **Final Layer:** Modified to output 4 neurons
- **Activation:** Sigmoid (for independent attribute prediction)
- **Loss Function:** Masked BCEWithLogitsLoss

The backbone layers are frozen and only the final classification layer is fine-tuned.

This ensures the model leverages pretrained ImageNet features instead of training from scratch.

---

##  Why Multi-Label?

This is a multi-label problem because:

- Each image can have more than one attribute.
- Attributes are independent.
- Softmax cannot be used.

Instead:
- Sigmoid activation is applied per attribute.
- BCEWithLogitsLoss is used for training.

---

##  Handling Missing Labels (NA)

Some attributes are marked as `NA` in the dataset.

Approach used:

- Detect `NA` explicitly during parsing.
- Create a mask tensor.
- Compute loss only for valid attributes.
- Do not discard any image.

This prevents missing labels from corrupting training.

---

##  Dataset Imbalance

The dataset shows imbalance across attributes.

The following strategies were considered:

- Class-weighted BCE (`pos_weight`)
- Focal Loss
- Weighted sampling
- Attribute-wise threshold tuning

Due to dataset size and stability concerns, the final implementation uses masked BCEWithLogitsLoss for stable training. Imbalance-aware strategies are identified as future improvements.

---

##  Data Preprocessing

The following transformations are applied:

- Resize to 224×224
- Random horizontal flip
- Convert to tensor
- Normalize using ImageNet mean and standard deviation

This ensures compatibility with pretrained ResNet18 weights.

---

##  Training Details

- Optimizer: Adam
- Learning Rate: 1e-5
- Batch Size: 16
- Epochs: 5
- Loss: Masked BCEWithLogitsLoss

Training loss is recorded per iteration.

The loss curve contains:

- X-axis: `iteration_number`
- Y-axis: `training_loss`
- Title: `Aimonk_multilabel_problem`

## Due to GitHub file size limits, the trained model weights (.pth) are not uploaded. 
The model can be reproduced by running `python train.py`.

