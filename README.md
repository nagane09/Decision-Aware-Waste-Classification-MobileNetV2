# Decision-Aware Waste Classification using Transfer Learning

## Live Demo
You can try the deployed Streamlit application here:  
🔗 https://nagane09-decision-aware-waste-classification-mobilen-app-przxfn.streamlit.app/

Users can upload waste images to obtain:
- Predicted material category
- Model confidence
- Decision-aware feedback for recycling guidance

---

## Abstract
Efficient waste segregation is essential for sustainable recycling systems. This project presents a **decision-aware image classification framework** for automatic waste material identification using **convolutional neural networks (CNNs)** and **transfer learning**. A lightweight **MobileNetV2** model is employed to classify waste images into *plastic, metal, glass, and cardboard*. The system integrates **confidence-based decision logic** to identify low-confidence predictions and potential incorrect disposal actions. Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

## 1. Problem Definition
Manual waste segregation is inefficient and error-prone. The goal of this project is to design an AI-based system that:

- Automatically classifies waste material from images  
- Reduces misclassification through confidence-aware decisions  
- Balances accuracy and computational efficiency for real-world use  

Formally, given an input image `x` of size 224 × 224 × 3, the model learns a mapping:

f(x; θ) → y  

where `y ∈ {cardboard, glass, metal, plastic}`.

---

## 2. Dataset Description
- **Classes:** Cardboard, Glass, Metal, Plastic  
- **Source:** Public waste image dataset (Kaggle)  
- **Total images:** ~12,000  

### Data Split
- Training: 70%  
- Validation: 15%  
- Testing: 15%  

Images are resized to 224 × 224 and normalized to the range [0, 1].

---

## 3. Dataset Handling & Preprocessing

### 3.1 Dataset Organization
Raw images are organized class-wise and programmatically split into training, validation, and test sets to avoid data leakage.

The dataset satisfies:
- No overlap between train, validation, and test sets  
- Balanced class distribution across splits  

A reproducible Python pipeline is used for shuffling and splitting.

---

### 3.2 Image Preprocessing
Each image undergoes:
- Resizing to 224 × 224  
- Pixel normalization using:

x_normalized = x / 255  

This ensures compatibility with ImageNet-pretrained weights.

---

### 3.3 Data Augmentation
To improve generalization and reduce overfitting, the following augmentations are applied during training:

- Random rotation (±20°)
- Horizontal flipping
- Width and height shifting
- Zoom augmentation

Validation and test datasets are **not augmented** to preserve evaluation integrity.

---

### 3.4 Class Encoding
Class labels are automatically encoded into categorical one-hot vectors, enabling optimization using categorical cross-entropy loss.

---

## 4. Technology Stack

### Programming & Libraries
- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

### Deep Learning & Computer Vision
- **MobileNetV2 (ImageNet pre-trained)**
- **Transfer Learning**
- **ImageDataGenerator**

### Data Management
- **SQLite** – Logging image predictions
- **OS / Shutil / Random** – Dataset handling

### Deployment & Tooling
- **Streamlit** – Interactive inference interface
- **Git / GitHub** – Version control

---

## 5. Model Architecture

### 5.1 Base Network
The model uses **MobileNetV2** as a frozen feature extractor.

**Architecture:**
- MobileNetV2 (without top layers)
- Global Average Pooling
- Dense (128 units, ReLU)
- Dropout (0.3)
- Dense (Softmax, 4 classes)

Final prediction is computed as:

ŷ = softmax(Wx + b)

---

## 6. Training Configuration
- **Optimizer:** Adam  
- **Learning Rate:** 0.001  
- **Loss Function:** Categorical Cross-Entropy  

Loss is defined as:

L = − Σ (y_c · log(ŷ_c)), for c = 1 to C  

- **Batch Size:** 16  
- **Epochs:** 20  

---

## 7. Decision-Aware Classification
To reduce incorrect disposal recommendations, a confidence-based decision mechanism is applied during inference.

If:

max(predicted probability) < 0.6  

the prediction is flagged as **low-confidence**, indicating uncertainty or potential misclassification.

This enables:
- Wrong-bin detection
- Safer real-world predictions
- Improved system reliability

---

## 8. Evaluation Metrics
Model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Metric definitions:

Precision_c = TP_c / (TP_c + FP_c)  
Recall_c = TP_c / (TP_c + FN_c)  

F1_c = 2 · (Precision_c · Recall_c) / (Precision_c + Recall_c)

---

## 9. Experimental Results
- **Overall Accuracy:** 90.74%  
- Consistent performance across all waste categories  
- Higher confusion observed between *plastic* and *metal* due to visual similarity  

Confusion matrix analysis provides detailed class-wise error insights.

---

## 10. Inference & Deployment
The trained model is deployed using **Streamlit** for real-time inference. Users can upload an image and receive:

- Predicted waste category  
- Model confidence  
- Decision-aware warning (if applicable)  

Deployment is intended as a **demonstration interface**, not the primary contribution.

---

## 11. Conclusion
This project demonstrates the effectiveness of **transfer learning–based CNNs** for automated waste classification. The introduction of **decision-aware confidence logic** improves robustness and reliability, making the system suitable for practical waste management scenarios. Future work includes model comparison, fine-tuning, and edge-device optimization.

