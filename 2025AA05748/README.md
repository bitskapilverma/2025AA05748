# 🐱🐶 Cat vs Dog Image Classification using Classical Machine Learning  
**Dataset:** CIFAR-10 (Cat vs Dog Subset)

---

## 📌 Project Overview

This project implements a **binary image classification system** to distinguish between **cats and dogs** using **classical machine learning algorithms**.  
Instead of deep learning end-to-end training, the project focuses on **feature extraction, dimensionality reduction (PCA), and traditional ML classifiers**, followed by deployment using **Streamlit**.

The CIFAR-10 dataset is filtered to retain only **cat (label 3)** and **dog (label 5)** images.

---

## 🎯 Objectives

- Perform image classification using classical ML models  
- Apply feature extraction and PCA  
- Compare multiple ML classifiers using standard metrics  
- Deploy the trained models using Streamlit  

---

## 📂 Dataset Description

- **Source:** tf.keras.datasets.cifar10  
- **Classes Used:** Cat (0), Dog (1)  
- **Total Samples:** ~12,000 images  
- **Image Size:** 64 × 64 (grayscale)  
- **Problem Type:** Binary Classification  

---

## ⚙️ Methodology

### 1️⃣ Data Loading
- CIFAR-10 dataset loaded using TensorFlow  
- Cat and Dog classes filtered  
- Labels converted to binary format  

### 2️⃣ Feature Extraction
- Resize images to 64 × 64  
- Convert to grayscale  
- Normalize pixel values  
- Flatten images into feature vectors  

### 3️⃣ Feature Scaling & Dimensionality Reduction
- StandardScaler for normalization  
- PCA applied with 25 components  

### 4️⃣ Machine Learning Models
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors  
- Naive Bayes  
- Random Forest  
- XGBoost  

### 5️⃣ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Matthews Correlation Coefficient  
- Confusion Matrix  

### 6️⃣ Model Saving
- Models, scaler, and PCA saved using joblib  

---

## 🖥️ Streamlit Application

A Streamlit-based web application is used to:
- Load trained models  
- Select models dynamically  
- Evaluate test data  
- Display metrics and confusion matrix  

⚠️ The Streamlit app is intended for deployment, not for execution inside notebooks.

---

## 📁 Project Structure

```
cifar10-cat-dog-classical-ml/
│── app.py
│── requirements.txt
│── README.md
│
└── model/
    ├── scaler.pkl
    ├── pca.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📜 Conclusion

This project shows that classical machine learning models combined with effective feature extraction and PCA can achieve reliable performance on image classification tasks.

---

## 👩‍💻 Author
Kapil Verma
Roll Number: 2025AA05748
Email: 2025aa05748@wilp.bits-pilani.ac.in
BITS Pilani - M.Tech (AIML)

📄 License
This project is created for educational purposes as part of BITS Pilani coursework.

🙏 Acknowledgments
BITS Pilani Work Integrated Learning Programmes
Kaggle for providing the dataset
Streamlit Community Cloud for free hosting
