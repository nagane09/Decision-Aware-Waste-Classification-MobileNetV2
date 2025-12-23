# Decision-Aware-Waste-Classification-MobileNetV2

This project is a **decision-aware AI system** for classifying waste and providing actionable recycling recommendations. It integrates **deep learning (CNN)** with **rule-based reasoning** and logs predictions in a SQLite database.

---

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo – https://nagane09-decision-aware-waste-classification-mobilen-app-przxfn.streamlit.app/

* Upload waste images to see real-time predictions, confidence, and recycling guidance.

---

# ♻️ Smart Waste Decision System

A **Decision-Aware AI system** that predicts the type of waste and suggests proper recycling actions using **deep learning and rule-based decision logic**.

---

## 🛠️ Tech Stack

| Component             | Libraries / Tools                                  |
|----------------------|---------------------------------------------------|
| Data Handling         | pandas, numpy                                     |
| Image Preprocessing   | tensorflow.keras.preprocessing, PIL              |
| Deep Learning         | TensorFlow, Keras, MobileNetV2                   |
| Model Persistence     | h5 model files                                    |
| Visualization         | matplotlib, seaborn                               |
| Web Dashboard         | Streamlit                                        |
| Database              | SQLite                                           |

---

## 📱 MobileNetV2 – Technical Overview

MobileNetV2 is a lightweight Convolutional Neural Network (CNN) optimized for speed and efficiency, ideal for mobile and edge devices. It achieves competitive accuracy while being faster and smaller than traditional CNNs like VGG or ResNet.

### Key Features

1. **Inverted Residuals with Linear Bottlenecks**  
   Preserves essential features while reducing computation.  
   Output = LinearProjection(DepthwiseConv(Expansion(X)))

2. **Depthwise Separable Convolutions**  
   Splits convolution into depthwise and pointwise layers to reduce parameters.  
   Conv_DW+PW(X) = Pointwise(Depthwise(X))

3. **ReLU6 Activation**  
   Clipped ReLU used for improved performance on low-precision hardware.

### Advantages over Traditional CNNs

| Feature                     | MobileNetV2 | Traditional CNNs |
|-------------------------------|-------------|----------------|
| Parameters & FLOPS            | Very low    | High           |
| Inference Speed               | Fast        | Slower         |
| Accuracy                      | Competitive | High but heavy |
| Edge Deployment               | Excellent   | Limited        |

### Why It Was Used in This Project

- Efficiency: Fast real-time predictions in the Streamlit app.  
- Transfer Learning: Pre-trained on ImageNet, reducing training time.  
- Lightweight: Suitable for 4-class waste image classification on limited hardware.

### Project Implementation

- Base Model: MobileNetV2(weights='imagenet', include_top=False)  
- Custom Layers: GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(4, softmax)  
- Loss Function: categorical_crossentropy  
- Optimizer: Adam with learning rate 0.001  

**Final Prediction (Softmax formula):**  
predicted_class_probability_i = exp(z_i) / (exp(z_1) + exp(z_2) + exp(z_3) + exp(z_4))  
where z_i is the output of the last dense layer for class i.


---

# ♻️ Smart Waste Decision System

A **Decision-Aware AI system** that predicts the type of waste and suggests proper recycling actions using **deep learning and rule-based decision logic**.

---

## 📂 Dataset Preparation

1. **Raw Dataset Structure**:   dataset/raw/, cardboard/, glass/, metal/, plastic/
2. **Split Dataset**: Training (70%), Validation (15%), Test (15%)  
3. **Data Augmentation**:  
- Rescale (1./255)  
- Rotation, width/height shift, horizontal flip, zoom  

---

## 🧠 Model Architecture

**Base Model**: MobileNetV2 (pre-trained on ImageNet, top layers removed)  

**Custom Head**:
- GlobalAveragePooling2D
- Dense(128, activation='relu')
- Dropout(0.3)
- Dense(4, activation='softmax') → 4 classes: `cardboard`, `glass`, `metal`, `plastic`

**Training Details**:
- Loss: Categorical Crossentropy
- Optimizer: Adam, LR = 0.001
- Batch Size: 16
- Epochs: 20
- Input Shape: (224, 224, 3)
- Metrics: Accuracy

**Python Snippet**:

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
````

## 🏋️ Training Pipeline

1. **Load datasets**: Training and validation datasets are loaded using `ImageDataGenerator`.  
2. **Data Augmentation**: Applied to training data (rotation, shift, flip, zoom).  
3. **Train Model**: Train the model for 20 epochs.  
4. **Save Model**:  
```python
model.save("waste_classifier_model.h5")
```

----

## 📈 Plot Training History

- Accuracy  
- Loss  

---

## 🧪 Model Evaluation

1. **Evaluate on Test Set**:
```python
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
````

------

# 🖥️ Streamlit Deployment

**app.py Features:**

- Upload waste image.
- Select the bin type manually.
- Predict waste type using trained model.
- Show confidence and recommendation:
  - Low confidence (<70%) → manual check
  - High confidence → show recyclable suggestion
- Validate if user selected the correct bin.
- Display results interactively.

**Run the App:**
```bash
streamlit run app.py
```
