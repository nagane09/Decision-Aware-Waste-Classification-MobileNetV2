## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo – https://nagane09-decision-aware-waste-classification-mobilen-app-przxfn.streamlit.app/

* Upload waste images to see real-time predictions, confidence, and recycling guidance.

---

# Decision-Aware Waste Classification using Transfer Learning

## Abstract
Efficient waste segregation is critical for sustainable recycling systems. This project presents a **decision-aware image classification framework** for automatic waste material identification using **convolutional neural networks (CNNs)** and **transfer learning**. A lightweight **MobileNetV2** architecture is fine-tuned to classify waste images into material categories such as *plastic, metal, glass, and cardboard*. The system incorporates **confidence-based decision logic** to flag uncertain predictions and potential incorrect disposal actions. Comprehensive evaluation using accuracy, precision, recall, F1-score, and confusion matrices demonstrates the effectiveness of the proposed approach.

---

## 1. Problem Definition
Manual waste segregation is error-prone and inefficient. The objective of this work is to design a **computer vision–based classification system** that:
- Automatically identifies waste material types from images
- Minimizes misclassification through decision-aware confidence thresholds
- Balances **accuracy, inference speed, and model complexity** for real-world deployment

Formally, given an input image \( x \in \mathbb{R}^{224 \times 224 \times 3} \), the task is to learn a function  
\[
f(x; \theta) \rightarrow y, \quad y \in \{1, 2, \dots, C\}
\]
where \( C = 4 \) denotes the number of waste categories.

---

## 2. Dataset Description
- **Classes:** Cardboard, Glass, Metal, Plastic  
- **Source:** Public waste image dataset (Kaggle)  
- **Total images:** ~12,000  
- **Data split:**
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

Images are resized to \(224 \times 224\) and normalized to \([0,1]\).

---

## 3. Data Preprocessing & Augmentation
To improve generalization and reduce overfitting, the following augmentations were applied during training:

- Random rotation (\(\pm20^\circ\))
- Horizontal flipping
- Width and height shifts
- Zoom augmentation

Let \( x_i \) denote an input image. Augmentation applies a stochastic transformation:
\[
\tilde{x}_i = T(x_i), \quad T \sim \mathcal{A}
\]

---
## Dataset Handling and Preprocessing

### Dataset Organization
Raw images are organized class-wise and programmatically restructured into
training, validation, and testing subsets to avoid data leakage.

\[
D = D_{train} \cup D_{val} \cup D_{test}, \quad
D_{train} \cap D_{val} \cap D_{test} = \varnothing
\]

The dataset is split using a **70/15/15** ratio to ensure robust evaluation.

### Automated Dataset Splitting
A reproducible Python pipeline is implemented to:
- Randomly shuffle images within each class
- Allocate samples to train, validation, and test sets
- Preserve class balance across splits

This guarantees unbiased generalization assessment.

---

### Image Preprocessing
Each image \( x \) undergoes the following transformations:

- Resizing to \(224 \times 224\)
- Pixel normalization:
\[
x' = \frac{x}{255}
\]

These steps align the input distribution with ImageNet pre-training statistics.

---

### Data Augmentation Strategy
To reduce overfitting and improve robustness, stochastic augmentations are applied during training:

- Rotation (\(\pm20^\circ\))
- Horizontal flipping
- Width and height shifting
- Zoom transformations

Let \( T \sim \mathcal{A} \) represent a random augmentation operator.  
The augmented sample is defined as:
\[
\tilde{x} = T(x)
\]

Validation and test sets are **not augmented** to preserve evaluation integrity.

---

### Class Encoding
Class labels are automatically encoded using categorical one-hot vectors:
\[
y \in \{0,1\}^C, \quad C = 4
\]

This encoding enables optimization using categorical cross-entropy loss.

---

### Data Integrity Measures
- Fixed random seed for reproducibility
- No overlap between training, validation, and test samples
- Class-wise sample distribution verified after splitting

---

## Technology Stack

### Programming & Core Libraries
- **Python 3.x** – Primary programming language
- **TensorFlow / Keras** – Model development and training
- **NumPy** – Numerical computations
- **Matplotlib** – Training and evaluation visualizations
- **Scikit-learn** – Evaluation metrics (confusion matrix, classification report)

### Deep Learning & Computer Vision
- **MobileNetV2** – Pre-trained CNN backbone (ImageNet weights)
- **Transfer Learning** – Feature reuse and fine-tuning strategy
- **ImageDataGenerator** – Real-time data augmentation and preprocessing

### Data Management
- **SQLite** – Lightweight storage for prediction logs
- **OS / Shutil / Random** – Dataset organization and splitting

### Deployment & Tooling
- **Streamlit** – Interactive inference interface
- **Git / GitHub** – Version control and experiment tracking

---

## 4. Model Architecture

### 4.1 Base Network
The backbone network is **MobileNetV2**, pre-trained on ImageNet. The final classification layers are replaced with task-specific layers.

**Architecture:**
- MobileNetV2 (frozen feature extractor)
- Global Average Pooling
- Dense (128 units, ReLU)
- Dropout (0.3)
- Dense (Softmax, 4 classes)

The final prediction is computed as:
\[
\hat{y} = \text{softmax}(Wx + b)
\]

---

## 5. Training Configuration
- **Optimizer:** Adam  
- **Learning rate:** 0.001  
- **Loss function:** Categorical Cross-Entropy  
\[
\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
\]
- **Batch size:** 16  
- **Epochs:** 20  

---

## 6. Decision-Aware Classification
To reduce incorrect disposal recommendations, a **confidence thresholding mechanism** is introduced.

Let:
\[
p_{\max} = \max(\hat{y})
\]

If:
\[
p_{\max} < \tau \quad (\tau = 0.6)
\]
the prediction is flagged as **low-confidence**, prompting user verification or rejection.

This enables **wrong-bin detection** and improves reliability in real-world usage.

---

## 7. Evaluation Metrics
Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

For each class \( c \):
\[
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}
\]
\[
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
\]
\[
\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
\]

---

## 8. Experimental Results

- **Overall accuracy:** 90.74%
- Strong performance across all material categories
- Higher confusion observed between *plastic* and *metal* due to reflective surface similarity

The confusion matrix provides detailed class-wise error analysis.

---

## 9. Inference & Deployment
A lightweight inference pipeline is implemented for real-time prediction. The trained model is deployed using **Streamlit**, allowing users to upload an image and receive:
- Predicted material class
- Confidence-aware decision output

Deployment serves as a **demonstration interface** rather than the core contribution.

---

## 10. Conclusion
This project demonstrates the effectiveness of **transfer learning–based CNNs** for waste material classification. The introduction of **decision-aware confidence logic** enhances reliability and makes the system suitable for practical waste management scenarios. Future work includes extending the framework to multi-modal inputs and edge-device optimization.

