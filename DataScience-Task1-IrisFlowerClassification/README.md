# Data Science Task 1 — Iris Flower Classification

**Intern:** Sheetal Patel  
**Organization:** Oasis Infobyte  
**Track:** Data Science  
**Task:** Iris Flower Classification

## Objective
Train machine-learning classifiers to identify Iris Setosa, Iris Versicolor, and Iris Virginica from sepal and petal measurements.

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Task Requirements Covered
- Built-in `sklearn.datasets.load_iris()` dataset
- Dataset shape, data types, null-value check, and descriptive statistics
- Pairplot showing feature distributions by species
- Box plots for all four features
- Feature-selection/discriminative-feature discussion
- 80/20 stratified train-test split
- Two classifiers: Logistic Regression and Random Forest
- Accuracy, precision, recall, F1-score, classification reports
- Confusion matrices for both models
- Transparent best-model selection using test accuracy with macro F1 as tie-breaker
- Random Forest feature-importance analysis
- Clean, commented notebook

## Project Files
```text
DataScience-Task1-IrisFlowerClassification/
├── Iris_Flower_Classification.ipynb
├── README.md
├── requirements.txt
└── outputs/
    ├── 01_pairplot.png
    ├── 02_boxplots.png
    ├── 03_correlation_heatmap.png
    ├── 04_model_comparison.png
    ├── 05_feature_importance.png
    ├── classification_report_logistic_regression.csv
    ├── classification_report_random_forest.csv
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── model_comparison.csv
    └── random_forest_feature_importance.csv
```

## Executed Results (random_state=42)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Random Forest | 90.00% | 90.24% | 90.00% | 89.97% |

For this execution, **Logistic Regression** was selected using higher test accuracy, with macro F1 as the tie-breaker. The Random Forest feature-importance analysis ranked `petal_length_cm` and `petal_width_cm` as the two most important features. These numbers are tied to the executed notebook run and should not be treated as universal model performance.

## Dataset
The Iris dataset is loaded directly from scikit-learn, so no external dataset download is required.

## How to Run
1. Create/activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `Iris_Flower_Classification.ipynb`.
5. Run all cells from top to bottom.

## Reproducibility
The notebook uses `random_state=42` for the train-test split and model configuration so that the evaluation can be reproduced on the same software/data setup.

## Academic Integrity
This submission is intended to be original work. Tutorials and documentation may be used for learning concepts, but the implementation, comments, analysis, and explanations should be understood and reviewed by the intern before submission.
