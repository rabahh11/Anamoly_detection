# Anamoly_detection
# Credit Card Fraud Detection

A data science project for CS 685/785 that applies anomaly detection and supervised learning to the Kaggle credit card fraud dataset. We leverage Singular Value Decomposition (SVD) on PCA‑transformed transaction features to flag potentially fraudulent transactions, then refine detection with a logistic regression classifier and explore clustering structure in the data.

## Dataset

- **Source**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Contents**: 284,807 credit card transactions (rows) and 31 features, including  
  - `Time`, `Amount`  
  - 28 anonymized PCA features (`V1`…`V28`)  
  - `Class` (0: legitimate, 1: fraud)

Place the raw CSV in:

## Exploratory Data Analysis & Hypothesis

1. **Data summary**: no missing values; heavy class imbalance (~0.17% fraud).  
2. **Feature testing**: univariate t‑tests and boxplots for each PCA component reveal significant differences between fraud and legitimate transactions.  
3. **Hypothesis**: fraudulent transactions are structurally different and produce higher reconstruction error when the data are projected onto a low‑rank subspace.

## Anomaly Detection via SVD

1. **Build matrix** \(X\) of PCA features (`V1`…`V28`).  
2. **Compute SVD**: \(X = U\Sigma V^T\) and reconstruct using the top **k = 10** singular values.  
3. **Reconstruction error**: L₂ norm between original and reconstructed transaction.  
4. **Threshold selection**: set at the 95th percentile of reconstruction errors on the legitimate class.  
5. **Evaluate**: precision‑recall, classification report, and confusion matrix for the SVD‑only detector.

## Supervised Classification

- **Model**: Logistic Regression with balanced class weights  
- **Features**: original PCA features plus reconstruction error  
- **Evaluation**: classification report and confusion matrix show improved recall and precision over SVD alone; precision‑recall curves illustrate trade‑offs.

## Clustering Analysis

- **Algorithm**: K‑Means (k=2) on the PCA feature space  
- **Visualization**: scatter plot of the first two components colored by cluster label  
- **Insight**: clusters partially separate fraudulent from legitimate transactions, suggesting structure beyond reconstruction error.




