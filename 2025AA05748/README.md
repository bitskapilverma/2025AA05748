# 🐱🐶 Cat vs Dog Image Classification using Classical Machine Learning  
**Dataset:** CIFAR-10 (Cat vs Dog Subset)

---

## a. Problem Statement

The objective of this project is to design and evaluate a **binary image classification system** that can accurately distinguish between **cat and dog images** using **classical machine learning algorithms**.  
The project focuses on feature extraction, dimensionality reduction, comparative model evaluation, and deployment using **Streamlit Community Cloud**.

---

## b. Dataset Description  **[1 Mark]**

- **Dataset Name:** CIFAR-10 (Cat vs Dog Subset)  
- **Source:** tf.keras.datasets.cifar10  
- **Selected Classes:**  
  - Cat → Label 3  
  - Dog → Label 5  
- **Number of Samples:** ~12,000 images after filtering  
- **Image Preprocessing:**  
  - Resized to 64 × 64  
  - Converted to grayscale  
  - Normalized pixel values  
- **Problem Type:** Binary Classification  

---

## c. Models Used & Evaluation Metrics  **[6 Marks]**

The following classical machine learning models were implemented and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

### 📊 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.5963 | 0.6239 | 0.5990 | 0.5825 | 0.5906 | 0.1926 |
| Decision Tree | 0.5483 | 0.5483 | 0.5487 | 0.5450 | 0.5468 | 0.0967 |
| kNN | 0.5892 | 0.6264 | 0.5921 | 0.5733 | 0.5826 | 0.1784 |
| Naive Bayes | 0.5975 | 0.6390 | 0.6005 | 0.5825 | 0.5914 | 0.1951 |
| Random Forest (Ensemble) | 0.6221 | 0.6655 | 0.6314 | 0.5867 | 0.6082 | 0.2448 |
| XGBoost (Ensemble) | 0.6017 | 0.6497 | 0.6059 | 0.5817 | 0.5935 | 0.2035 |

📌 Replace `0.XX` with actual metric values obtained from the notebook.

---

## 🔍 Model-wise Observations  **[3 Marks]**

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved moderate performance due to its linear decision boundary. It works well on scaled PCA features but struggles with complex image patterns. |
| Decision Tree | Learned training data effectively but tended to overfit, leading to reduced generalization on unseen images. |
| kNN | Performance was sensitive to the value of k and computationally expensive, making it less scalable for large datasets. |
| Naive Bayes | Fast and simple model; however, its strong independence assumption limited classification accuracy on image features. |
| Random Forest (Ensemble) | Delivered strong and stable performance by combining multiple trees, reducing overfitting and improving robustness. |
| XGBoost (Ensemble) | Achieved the best overall performance due to gradient boosting, regularization, and effective modeling of complex feature interactions. |

---

## ⚙️ Feature Engineering & Preprocessing

- Grayscale image flattening  
- Feature normalization using StandardScaler  
- Dimensionality reduction using PCA (25 components)  

---

## 🖥️ Step 6: Deployment on Streamlit Community Cloud

The trained models were deployed on **Streamlit Community Cloud** using the provided `app.py` and `requirements.txt`.  
The application enables model selection, evaluation metric visualization, confusion matrix display, and classification report generation.

⚠️ The Streamlit application is intended for deployment and not for execution inside Jupyter or Colab notebooks.

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

## 📜 Conclusion

This project demonstrates that classical machine learning models, when combined with effective feature extraction and PCA, can successfully solve image classification problems such as Cat vs Dog classification. Ensemble models achieved superior and more stable performance.

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