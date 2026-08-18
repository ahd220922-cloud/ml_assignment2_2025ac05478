# ML Assignment 2 - Breast Cancer Classification with Streamlit

## a. Problem Statement

Breast cancer diagnosis is a critical binary classification problem: given a set
of measurements computed from a digitized image of a fine needle aspirate (FNA)
of a breast mass, predict whether the mass is **malignant** or **benign**. Early
and accurate classification directly supports faster, more reliable diagnosis.
This project implements and compares 5 classical machine learning classifiers on
this task and exposes them through an interactive Streamlit web application.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (accessed via `sklearn.datasets.load_breast_cancer`,
  which bundles the original UCI dataset)
- **Instances:** 569 (>= 500 required ✅)
- **Features:** 30 numeric features (>= 12 required ✅), e.g. `mean radius`,
  `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean concavity`, etc.
  Features are computed from cell nuclei present in the FNA image (mean, standard
  error, and "worst"/largest value for 10 base measurements).
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant, 357 benign
- **Split used:** 80% train / 20% test, stratified by class, `random_state=42`

The held-out **test split** (features + true `target` label) is saved as
`test_data.csv` in this repository and is the file used to demonstrate the
Streamlit app (upload option).

## c. GitHub Repository Link

> Replace with your actual repository link after you push this project,
> (https://github.com/ahd220922-cloud/ml_assignment2_2025ac05478)

## d. Models Used

All 5 models were trained on the same dataset/split. Evaluation metrics were
computed on the held-out 20% test set (114 samples).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(Values are also saved in `model/metrics_results.csv`, generated directly by `model/train_models.py`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — the 30 features are largely linearly separable after standard scaling, so a linear decision boundary fits very well. Highest accuracy, AUC, and MCC of all 5 models. |
| Decision Tree | Weakest performer. A single unconstrained tree overfits the training data and does not generalize as well, giving the lowest accuracy, AUC, and MCC. Pruning or limiting `max_depth` would likely close some of this gap. |
| kNN | Strong performer with perfect recall (1.0) on the test set — it never misses a benign case — but this comes with a small drop in precision compared to Logistic Regression, meaning slightly more malignant cases get labeled benign-leaning. Performs well because scaled feature space places same-class points close together. |
| Naive Bayes | Decent but the weakest of the "well-behaved" models. The Gaussian Naive Bayes assumption of feature independence doesn't fully hold (several of the 30 features are correlated, e.g. mean radius vs mean area), which caps its ceiling. |
| Random Forest (Ensemble) | Very strong and stable — averaging many decision trees fixes the overfitting problem seen in the single Decision Tree, giving a large jump in every metric. Slightly behind Logistic Regression here, but typically more robust on noisier or non-linear datasets. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy (0.9825), AUC (0.9954), F1 (0.9861), and MCC (0.9623) of all 5 models on this test split. Random Forest is a close second and would likely generalize slightly better on unseen, noisier data due to its ensemble nature. |

> Note: results depend on the random seed / train-test split. Re-running
> `model/train_models.py` with a different `random_state` may shift rankings
> slightly, though Logistic Regression and Random Forest consistently perform
> best on this dataset given how linearly separable its classes are.

## Repository Structure

```
project-folder/
│-- app.py                # Streamlit app (main entry point)
│-- requirements.txt
│-- README.md
│-- test_data.csv         # held-out test data used in the Streamlit app
│-- model/
│   │-- train_models.py                       # trains all 5 models + saves artifacts
│   │-- model_logistic_regression.joblib
│   │-- model_decision_tree.joblib
│   │-- model_knn.joblib
│   │-- model_naive_bayes.joblib
│   │-- model_random_forest_ensemble.joblib
│   │-- scaler.joblib                         # StandardScaler fit on training data
│   │-- model_meta.json                       # feature names / which models need scaling
│   └-- metrics_results.csv                   # comparison table (source of truth)
```


## How the App Was Deployed

1. Pushed this repository to GitHub.
2. Signed in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Clicked **New App** → selected this repository → branch `main` → main file `app.py`.
4. Clicked **Deploy**. 

**Live Streamlit App Link:** > https://mlassignment22025ac05478.streamlit.app/
https://mlassignment22025ac05478.streamlit.app/

## Screenshot (BITS Virtual Lab Execution)

> <img width="1916" height="1078" alt="Screenshot 2026-08-18 191327" src="https://github.com/user-attachments/assets/e40fb1bb-c103-4819-bd9a-420062f366d0" />
<img width="1917" height="1078" alt="Screenshot 2026-08-18 191255" src="https://github.com/user-attachments/assets/77f36265-fa0f-4082-8112-11a6a2497a79" />
<img width="1915" height="1078" alt="Screenshot 2026-08-18 191233" src="https://github.com/user-attachments/assets/bdae0e60-b7e0-4077-82c8-4109961f3f48" />
<img width="1917" height="1032" alt="Screenshot 2026-08-18 191218" src="https://github.com/user-attachments/assets/2679fbc7-8242-48e1-a881-2bcbaa23b48c" />
<img width="1913" height="983" alt="Screenshot 2026-08-18 185525" src="https://github.com/user-attachments/assets/930f5def-58c9-45af-a2ca-9e607db72ca7" />





