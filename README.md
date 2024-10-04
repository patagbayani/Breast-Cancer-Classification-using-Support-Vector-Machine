# Breast Cancer Classification

This simple project is focused on building a machine learning model to classify breast cancer tumors as malignant or benign based on a set of features derived from digitized images of fine-needle aspirate (FNA) of breast mass tissue. The dataset used is the famous Breast Cancer Wisconsin dataset. Special thanks to Dr. Ryan Ahmed for guiding me throughout the project. 

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Technologies Used](#technologies-used)
7. [Contributing](#contributing)

## Project Overview
The goal of this project is to apply machine learning algorithms to classify breast cancer as malignant or benign. Various techniques are explored, such as data preprocessing, feature selection, model training, and evaluation using key performance metrics.

The project steps include:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Model training using different classification algorithms
- Model evaluation using accuracy, precision, recall, and other relevant metrics
- Hyperparameter tuning to optimize the model performance

## Dataset
The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains features computed from digitized images of fine-needle aspirate (FNA) of breast mass tissue.

### Features:
- **Radius** (mean of distances from the center to points on the perimeter)
- **Texture** (standard deviation of gray-scale values)
- **Perimeter**
- **Area**
- **Smoothness** (local variation in radius lengths)
- **Compactness** (perimeter^2 / area - 1.0)
- **Concavity** (severity of concave portions of the contour)
- **Concave points** (number of concave portions of the contour)
- **Symmetry**
- **Fractal dimension** ("coastline approximation" - 1)

### Target:
- **Diagnosis**: Malignant or Benign (1 for malignant, 0 for benign)

## Usage
To use this notebook, follow the steps below:

1. Clone the repository or download the `.ipynb` file.
2. Open the Jupyter Notebook.
3. Run the cells to preprocess the data, train models, and evaluate their performance.

You can also modify the code to test additional models or perform hyperparameter tuning.

```bash
jupyter notebook BreastCancerClassification.ipynb
```
## Model Training
In this project the machine learning model that was used is:
- **Support Vector Machines (SVM)**

The model is trained on a training set and evaluated on a separate test set using various performance metrics. The model's performance is measured based on accuracy, precision, recall, and F1-score, allowing for comprehensive evaluation and comparison.

## Results
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00      | 0.94   | 0.97     | 48      |
| 1.0   | 0.96      | 1.00   | 0.98     | 66      |
| **avg / total** | **0.97** | **0.97** | **0.97** | **114** |

The model achieves high accuracy in classifying whether a tumor is malignant or benign. The performance is evaluated using the following metrics:
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1-score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: A table used to describe the performance of the model by displaying true positives, false positives, true negatives, and false negatives.

Visualizations such as ROC curves and confusion matrices are included to illustrate model performance effectively.

## Technologies Used
- **Python**: The primary programming language used for data analysis and modeling.
- **Jupyter Notebook**: For running the project and documenting the process.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and handling arrays.
- **Matplotlib** and **Seaborn**: For data visualization and plotting graphs.
- **Scikit-learn**: For implementing machine learning algorithms and model evaluation.

## Contributing
Contributions to the project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to fork the repository, create a branch, and submit a pull request. Please ensure that your code adheres to the project's coding standards and includes appropriate documentation.

Thank you for your interest in contributing to this project!





