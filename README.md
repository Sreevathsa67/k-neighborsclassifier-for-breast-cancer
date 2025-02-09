# Breast Cancer Prediction using K-Nearest Neighbors (KNN)

## 📌 Overview
This project is a **breast cancer prediction system** using the **K-Nearest Neighbors (KNN) classifier**. It takes input features related to breast cancer diagnosis and predicts whether the tumor is **malignant (cancerous)** or **benign (non-cancerous)**.

The dataset used in this project is sourced from **Kaggle** and preprocessed for better model performance.

## 🚀 Features
- **Machine Learning Model**: Uses **K-Nearest Neighbors (KNN)** for classification.
- **Data Preprocessing**: Standardization using **StandardScaler**.
- **Model Training & Evaluation**: Splitting dataset into training and testing sets.
- **User Input Prediction**: Accepts real-world data and predicts cancer type.

## 🗂 Dataset
- The dataset contains **features related to breast cancer diagnosis**.
- It is sourced from Kaggle’s **Breast Cancer Wisconsin Dataset**.
- The `target` column represents:
  - `0`: Malignant (Cancerous)
  - `1`: Benign (Non-cancerous)

## 🔧 Technologies Used
- **Python**
- **Pandas** (Data handling)
- **NumPy** (Numerical computations)
- **Scikit-Learn** (Machine learning library)

## 📜 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Sreevathsa67/k-neighborsclassifier-for-breast-cancer.git
   ```
2. Navigate to the project folder:
   ```bash
   cd k-neighborsclassifier-for-breast-cancer
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗 Usage
1. **Run the script**
   ```bash
   python breast_cancer_knn.py
   ```
2. **Check the prediction output**:
   - `Malignant`: Cancer detected.
   - `Benign`: No cancer detected.

## 📊 Model Performance
- The model is trained on **80% of the dataset** and tested on the remaining **20%**.
- The accuracy score is displayed after training.

## 🔥 Future Improvements
- Improve accuracy with hyperparameter tuning.
- Implement a **GUI using Streamlit**.
- Deploy as a web application.

## 🏆 Acknowledgments
- Dataset from **Kaggle: Breast Cancer Wisconsin Dataset**.
- Libraries: **Scikit-learn, Pandas, NumPy**.

## 📜 License
This project is open-source and available under the **MIT License**.

