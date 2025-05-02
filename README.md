# Algerian Forest Fires Prediction Analysis

## Project Overview
1. This project analyzes the Algerian Forest Fires dataset and aims to predict the fire weather index (FWI) using various regression models. 

2. The models used in this project include Linear Regression, Lasso, Ridge, ElasticNet, and their cross-validated versions. 

3. The dataset contains information about fire incidents across different regions of Algeria, with various features influencing fire behavior.

### Key Features:
- **Data Preprocessing**: Cleaning and feature engineering, including handling missing values, feature scaling, and encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizing the dataset to find patterns and correlations.
- **Model Building**: Applying multiple regression techniques to predict the FWI.
- **Hyperparameter Tuning**: Using cross-validation to fine-tune model parameters for better performance.
- **Performance Evaluation**: Evaluating the models using Mean Absolute Error (MAE) and R² score.

## File Structure
The repository contains the following files:
- **Algerian_forest_fires_dataset.ipynb**: Data cleaning and exploratory data analysis (EDA) phase.
- **Model_Training.ipynb**: Building and evaluating multiple regression models.
- **images/**: Folder containing visualizations generated during the analysis (e.g., scatter plots, box plots, and heatmaps).

## 📂 File Structure

```bash
Algerian-Forest-Fires-Analysis/
├── data/
│   ├── Algerian_forest_fires_dataset.csv
│   └── Algerian_forest_fires_cleaned_dataset.csv
├── notebooks/
│   └── Algerian_forest_fires_dataset.ipynb
│   └── Model_Training.ipynb
├── images/
│   └── bejaia_monthly_fire_analysis.png
│   └── elastic_net_actual_vs_predicted.png
│   └── elastic_net_cv_actual_vs_predicted.png
│   └── fwi_boxplot.png
│   └── lasso_regression_actual_vs_predicted.png
│   └── lassocv_regression_actual_vs_predicted.png
│   └── linear_regression_actual_vs_predicted.png
│   └── multicollinearity_heatmap.png
│   └── ridge_cv_actual_vs_predicted.png
│   └── ridge_regression_actual_vs_predicted.png
│   └── numerical_features_histograms.png
│   └── sidi_bel_abbes_monthly_fire_analysis.png
│   └── standard_scaler_effect_boxplots.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Requirements
The following Python libraries are required to run the notebooks:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Project Workflow

### 1. Data Preprocessing & EDA
The first notebook focuses on:
- Loading and cleaning the dataset.
- Generating descriptive statistics.
- Visualizing data distributions and correlations through heatmaps, histograms, and box plots.

#### Key Visualizations:
- **Correlation Matrix**: To check multicollinearity between features.
- **Box Plots**: To analyze the distribution of features and detect outliers.
- **Histograms**: For feature distribution analysis.

#### Example:
- **Image Name**: `multicollinearity_heatmap.png`
- **Purpose**: Visual representation of feature correlations to detect multicollinearity.

### 2. Model Building and Evaluation
The second notebook focuses on training various regression models:
- **Linear Regression**
- **Lasso Regression (with Cross-Validation)**
- **Ridge Regression (with Cross-Validation)**
- **ElasticNet Regression (with Cross-Validation)**

Each model is trained and evaluated using Mean Absolute Error (MAE) and R² score. The models are also tuned using cross-validation where applicable (LassoCV, RidgeCV, and ElasticNetCV).

#### Example:
- **Image Name**: `elastic_net_cv_actual_vs_predicted.png`
- **Purpose**: Scatter plot comparing actual vs predicted values for ElasticNetCV regression.

#### Model Evaluation Metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

#### Example of Results:
For **ElasticNetCV** regression, the output is as follows:
- **Mean Absolute Error**: 0.66
- **R² Score**: 0.98

### 3. Hyperparameter Tuning
Hyperparameter tuning is done using cross-validation to improve the model's performance, particularly for Lasso, Ridge, and ElasticNet models.

## Visualizations Folder
The **images/** folder contains the following visualizations:
- **elastic_net_cv_actual_vs_predicted.png**: Actual vs Predicted values for the ElasticNetCV model.
- **ridge_cv_actual_vs_predicted.png**: Actual vs Predicted values for the RidgeCV model.
- **lasso_cv_actual_vs_predicted.png**: Actual vs Predicted values for the LassoCV model.
- **linear_regression_actual_vs_predicted.png**: Actual vs Predicted values for the Linear Regression model.
- **feature_importance.png**: Feature importance plot after training the models.

## How to Run the Notebooks

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mehul-kh2005/Algerian-Forest-Fires-Analysis.git
   ```

2. **Navigate to the project folder:**
    ```bash
    cd Algerian-Forest-Fires-Analysis
    ```

3. **Open the notebooks in Jupyter or VSCode:**
    ```bash
    jupyter notebook
    ```

4. **Execute the cells in the notebooks to perform the analysis and model training.**


## Acknowledgments
- **Dataset:** The Algerian Forest Fires dataset was used for this analysis, available from [Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset).

- **Libraries:** This project was developed using popular Python libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

## Author

### Mehul Khandelwal – [GitHub Profile](https://github.com/Mehul-kh2005/) 
📝 *This project was developed as part of a data science and machine learning exercise focused on regression modeling.*