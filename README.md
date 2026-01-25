# Decision-Aware Waste Classification using Transfer Learning

## Live Demo
You can try the deployed Streamlit application here:  
🔗 https://nagane09-decision-aware-waste-classification-mobilen-app-przxfn.streamlit.app/

Users can upload waste images to obtain:
- Predicted material category
- Model confidence
- Decision-aware feedback for recycling guidance

---

## Introduction

Urban waste management and recycling are critical for environmental sustainability. However, manual sorting is time-consuming and error-prone, leading to inefficiencies and contamination in recycling streams. This project proposes a **Smart Waste Decision System**, an AI-driven framework that combines deep learning for automated waste classification with rule-based decision reasoning to improve recycling practices.  

The system leverages convolutional neural networks (CNNs) for image classification, integrates explainable AI techniques for interpretability, and incorporates a user-friendly interface for real-world deployment. This approach aims to enhance recycling efficiency, reduce human error, and provide actionable guidance for end users.

---

## Objectives

The main objectives of the project are:  

- To develop an image classification model capable of accurately distinguishing between key recyclable materials: cardboard, glass, metal, and plastic.  
- To implement a rule-based decision layer that recommends appropriate recycling actions and bin selection.  
- To enhance model transparency through explainability techniques, allowing users to understand model predictions.  
- To create a system architecture that supports logging, auditing, and continuous improvement of the predictive model.

---

## Dataset

The dataset consists of images of recyclable materials categorized into four classes: cardboard, glass, metal, and plastic. The images were collected from multiple sources to capture real-world variability, including different lighting conditions, orientations, and object sizes.  

Key characteristics of the dataset:  

- **Class Distribution:** Balanced across all four classes to avoid bias.  
- **Preprocessing:** Images were resized to a uniform resolution and normalized for consistent input to the model.  
- **Data Augmentation:** Techniques such as rotation, translation, zooming, and horizontal flipping were applied to increase data diversity and improve generalization.  
- **Dataset Splits:** Divided into training, validation, and test sets to ensure unbiased evaluation and hyperparameter tuning.

---

## Methodology

### Model Architecture

A transfer learning approach was employed using a pretrained convolutional neural network as a feature extractor. The pretrained network captures high-level visual representations, which are then fine-tuned with additional dense layers to specialize the model for the waste classification task. Dropout layers were incorporated to reduce overfitting and improve generalization.  

This design enables leveraging large-scale visual knowledge while maintaining computational efficiency and adaptability to a small, domain-specific dataset.

### Training Strategy

The training process incorporated:  

- **Augmented Data Input:** To simulate real-world variability in waste images.  
- **Supervised Learning:** Using categorical cross-entropy loss for multi-class classification.  
- **Validation Monitoring:** To prevent overfitting and evaluate generalization performance on unseen data.  
- **Optimization:** Adaptive gradient-based optimizers were applied for faster convergence.

### Evaluation Metrics

Model performance was assessed using standard classification metrics:  

- **Accuracy:** Overall correctness of model predictions.  
- **Precision, Recall, and F1-Score:** To measure the balance between false positives and false negatives for each class.  
- **Confusion Matrix:** To analyze misclassifications and guide model improvements.  

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Cardboard  | 1.00      | 0.97   | 0.98     | 61      |
| Glass      | 0.84      | 0.84   | 0.84     | 76      |
| Metal      | 0.85      | 0.89   | 0.87     | 62      |
| Plastic    | 0.85      | 0.84   | 0.84     | 73      |
| **Accuracy** |           |        | 0.88     | 272     |
| **Macro Avg** | 0.88      | 0.88   | 0.88     | 272     |
| **Weighted Avg** | 0.88      | 0.88   | 0.88     | 272     |


### Error Analysis

Misclassifications were studied to identify patterns where the model struggles, such as visually similar classes or images with low quality. This analysis informed decisions on dataset augmentation, class balancing, and potential model enhancements.

---

## Explainability

To ensure transparency, the system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual explanations. Grad-CAM highlights image regions that contributed most to the model’s decision, enabling:  

- Verification of model focus on relevant features.  
- Identification of potential weaknesses in the model.  
- Increased trust from end users and stakeholders in automated recommendations.

---

## Rule-Based Decision Layer

Beyond classification, a rule-based layer interprets predictions and provides actionable recommendations:  

- Suggests correct bin type for disposal.  
- Provides material-specific recycling instructions (e.g., cleaning requirements for cardboard and plastics).  
- Issues warnings when model confidence is low, prompting manual verification.  

This hybrid design ensures that AI-driven predictions are both interpretable and actionable in real-world scenarios.

---

## System Architecture and Deployment

The system integrates multiple components:  

- **Deep Learning Model:** Performs image classification.  
- **Web Interface:** Allows users to upload images and view predictions.  
- **Decision Engine:** Provides bin guidance and recycling instructions.  
- **Database Logging:** Captures predictions for auditing, performance monitoring, and continuous learning.  

This architecture facilitates scalability, real-time interaction, and long-term analysis of model performance in operational environments.

---

## Research Contributions

This project demonstrates several contributions at a Master’s level in Data Science:  

1. **Integration of Transfer Learning and Rule-Based Reasoning:** Combines data-driven AI with domain knowledge for actionable outcomes.  
2. **Explainable AI Application:** Incorporates visual explanations to improve trust and interpretability.  
3. **Data Augmentation and Error Analysis:** Applies rigorous preprocessing and dataset enhancement strategies to improve model generalization.  
4. **End-to-End Deployment:** Bridges the gap between research and practical applications through a deployable web interface and prediction logging system.

---

## Future Work

Potential directions for extending this research include:  

- Expanding the dataset to include additional waste types (organic, hazardous, mixed materials).  
- Improving robustness under challenging real-world conditions, such as poor lighting or occlusion.  
- Incorporating active learning to continuously improve the model from user feedback.  
- Integrating multi-modal inputs, such as sensor data, for more accurate classification.

---

## Conclusion

The Smart Waste Decision System demonstrates a **research-oriented, applied approach** to intelligent recycling. By combining deep learning, explainable AI, and rule-based reasoning, it provides a practical and interpretable solution for sustainable waste management. This project is aligned with Master’s-level research standards in Data Science, emphasizing methodological rigor, explainability, and real-world applicability.


