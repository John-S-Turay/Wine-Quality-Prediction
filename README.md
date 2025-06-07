# Wine Quality Prediction

This project aims to predict the quality of wine based on various physicochemical features. It involves data loading, exploration, preparation, and the application of machine learning models for classification.

## Project Overview

The project analyzes two datasets containing information about red and white wine quality. The goal is to build models that can classify the quality of a wine as 'Poor', 'Average', or 'Excellent' based on its characteristics.

The notebook covers the following steps:

1.  **Data Loading and Initial Exploration:** Loading the red and white wine datasets, checking for null values, and examining basic statistics and data types.
2.  **Data Merging and Exploration:** Combining the red and white wine datasets and exploring the combined data through visualizations and correlation analysis.
3.  **Data Preparation:** Creating a target variable ('target') by categorizing wine quality into 'Poor', 'Average', and 'Excellent'. Identifying and potentially dropping highly correlated features.
4.  **Data Splitting:** Splitting the data into training and testing sets.
5.  **Feature Engineering:** Creating new features based on existing ones to potentially improve model performance.
6.  **Feature Selection:** Selecting the most important features for model training.
7.  **Model Training and Evaluation:**
    *   Training and evaluating a RandomForestClassifier and a LogisticRegression model with hyperparameter tuning.
    *   Establishing a baseline using a DummyClassifier.
    *   Implementing a more exhaustive feature engineering pipeline including polynomial features and variance threshold.
    *   Training and evaluating a GradientBoostingClassifier and an SVC model.
    *   Combining the GradientBoostingClassifier and SVC models using a VotingClassifier (both soft and hard voting).
    *   Comparing the performance of all models with the baseline using metrics like accuracy, precision, recall, F1-score, and confusion matrices.

## Data

The project uses two datasets:

*   **Red Wine Quality Dataset:** This dataset contains information about red variants of the Portuguese "Vinho Verde" wine. It includes 11 physicochemical features and a quality score (output variable).
*   **White Wine Quality Dataset:** This dataset contains information about white variants of the Portuguese "Vinho Verde" wine. It also includes 11 physicochemical features and a quality score.

Both datasets are publicly available and commonly used for demonstrating classification tasks. In this project, they are combined to create a larger dataset for training and evaluating the models.

## Dependencies

The following Python libraries are required to run the notebook:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn`

You can install these dependencies using pip:


## How to Run the Code

1.  Open the notebook in Google Colab or a Jupyter Notebook environment.
2.  Upload the `winequality-red.csv` and `winequality-white.csv` files to your environment.
3.  Run the cells sequentially.

## Results

The notebook provides an analysis of different machine learning models for wine quality prediction, including their performance metrics and confusion matrices. The performance of the implemented models is compared against a DummyClassifier baseline.

## Conclusion

The notebook demonstrates a typical workflow for a classification problem, including data exploration, preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation. The comparison with the baseline model helps in understanding the effectiveness of the chosen machine learning algorithms.
