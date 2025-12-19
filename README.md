# Decision-Aware-Waste-Classification-MobileNetV2

This project is a **decision-aware AI system** for classifying waste and providing actionable recycling recommendations. It integrates **deep learning (CNN)** with **rule-based reasoning** and logs predictions in a SQLite database.

---

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo – https://nagane09-decision-aware-waste-classification-mobilen-app-przxfn.streamlit.app/

* Upload waste images to see real-time predictions, confidence, and recycling guidance.

---

## **Project Purpose**

* Classifies waste images into categories: `cardboard`, `glass`, `metal`, `plastic`.
* Assesses **prediction confidence**.
* Provides **recycling recommendations** based on predicted waste type.
* Validates whether the waste was **thrown in the correct bin**.
* Stores predictions in a **SQLite database** for tracking.

---

## **Workflow / Pipeline**

1. **Dataset Preparation**
   * Raw waste images in `dataset/raw/` organized by class.
   * Split into train, validation, test sets (70/15/15).
   * Output: `dataset/dataset/train/`, `validation/`, `test/`.

2. **Model Training**
   * Pretrained **MobileNetV2** backbone.
   * Custom layers: GlobalAveragePooling → Dense(128) → Dropout → Dense(4 softmax)
   * Image augmentation via `ImageDataGenerator`.
   * Output: `waste_classifier_model.h5`.

3. **Model Evaluation**
   * Test dataset used to compute **accuracy**, **classification report**, **confusion matrix**.

4. **Streamlit App**
   * Upload image → model predicts waste type + confidence.
   * Provides **recycling recommendations**.
   * Checks if user-selected bin matches predicted class.
   * Stores prediction in **SQLite database** (`waste_predictions.db`).

---

## **Files and Their Roles**

| File / Folder                                    | Purpose                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| `app.py`                                         | Streamlit web app for real-time waste classification and decision-making.       |
| `waste_classifier_model.h5`                      | Trained deep learning model for waste classification.                           |
| `dataset/raw/`                                   | Original waste images separated by class.                                       |
| `dataset/dataset/train/`, `validation/`, `test/` | Split dataset ready for model training.                                         |
| `database_setup.py`                              | Creates SQLite database and table to store predictions.                         |
| `dataset_split.py`                               | Splits raw images into train/validation/test folders automatically.             |
| `train_model.py`                                 | Trains MobileNetV2-based CNN on the dataset, saves `waste_classifier_model.h5`. |
| `requirements.txt`                               | Python dependencies for running the app.                                        |

---

## **Key Features**

* **Deep Learning + Decision Rules**: CNN classification combined with rule-based recycling suggestions.  
* **Confidence Threshold**: Warns if prediction confidence is below 70%.  
* **Interactive Feedback**: Checks if the uploaded waste is thrown in the correct bin.  
* **Database Logging**: Stores predictions in `waste_predictions.db` for tracking and analysis.  
* **Augmented Dataset**: Uses image augmentation to improve model generalization.  


