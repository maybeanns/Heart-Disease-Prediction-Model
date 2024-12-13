# Heart Disease Prediction Using Random Forest Classifier

## Overview
This project predicts the likelihood of heart disease using machine learning techniques, with a specific focus on **linear algebra** concepts. It uses the **UCI Heart Disease Dataset** and demonstrates the role of preprocessing, feature scaling, and classification algorithms. Additionally, the project includes an interactive **Streamlit web application** for real-time user interaction.
![Screenshot (21)](https://github.com/user-attachments/assets/fbefe998-6e69-4075-b3ac-5d0cc47bdc87)

## Features
- Predicts heart disease based on 13 clinical parameters.
- Implements feature scaling, matrix operations, and PCA for dimensionality reduction.
- Trained a **Random Forest Classifier** with hyperparameter tuning.
- Interactive **Streamlit web app** for predictions.
- Visualizations include confusion matrices and feature importance.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy, Scikit-learn
  - Matplotlib, Seaborn
  - Streamlit for web app development
- **Linear Algebra Concepts**:
  - Feature scaling: Standardization of features to have zero mean and unit variance.
  - PCA: Dimensionality reduction by finding eigenvectors and eigenvalues of covariance matrices.

## Dataset
- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Number of Instances**: 303
- **Features**: Age, Cholesterol, Max Heart Rate, and 10 others.


## Usage
1. Open the Streamlit web app in your browser.
2. Input clinical parameters such as age, cholesterol, and resting blood pressure.
3. Click "Predict" to see the likelihood of heart disease.
4. Visualize the confusion matrix and feature importance.

## Results
- **Model Accuracy**: 92%

- **Feature Importance**:
  Visualized the significance of clinical parameters in the Random Forest model.

## Linear Algebra Applications
- **Feature Scaling**:
  - StandardScaler:
    ```
    x' = (x - μ) / σ
    ```
    where μ is the mean and σ is the standard deviation.
- **PCA**:
  - Covariance matrix and eigenvector analysis for dimensionality reduction.

## Future Work
- Improve accuracy by testing advanced algorithms like XGBoost or LightGBM.
- Deploy the web app on a cloud platform like AWS or Azure.
- Expand the dataset for better generalization.

## Contribution
Contributions are welcome! Please create a pull request with detailed information about the changes.

## License
This project is licensed under the MIT License.
