## Problem Statement

The goal of this assignment is to use different machine learning models to identify the type of defect in a steel plate. Steel manufacturing involves continuous inspection of plate surfaces to detect defect that affect the structural integrity and product quality.

Each defect type in our selected dataset typically originates from a different point in the manufacturing process. Knowing the specific type lets engineers trace it back to the specific machine, process step, or material batch responsible. Furthermore, not all defects are equally severe or equally fixable. A surface stain might be cosmetic and the plate can still be sold at a lower grade; a structural scratch might compromise the plate's mechanical integrity and require scrapping.

## Dataset Description

**Source:** [Steel Plate Defect Extended Dataset (19219 + 1940)](https://www.kaggle.com/datasets/kamal2026/steel-plate-defect-extended-dataset)

The dataset contains ~21k instances of defect regions on steel plates, recording 27 numeric features that describe their geometric shape, outline, material and luminosity properties. It is created by merging Kaggle's training dataset of steel plate defect prediction competition and additional data steel plate fault from UC Irvine Machine Learning Repository.

The target labels for the predictions are 7 categories of faults that can be identified in the steel plates. In the given data instead of being described as 7 different label values to a single column, they are presented as one hot encoded values.

## Github Repository Link

https://github.com/Shikhar-srivastav/classification_models

## Classification Models' Metrics

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.4702   | 0.8519 | 0.5447    | 0.4702 | 0.4581 | 0.3919 |
| Decision Tree       | 0.4783   | 0.6833 | 0.4824    | 0.4783 | 0.4801 | 0.3233 |
| K Nearest Neighbors | 0.5324   | 0.7859 | 0.5220    | 0.5324 | 0.5215 | 0.3885 |
| Naive Bayes         | 0.4556   | 0.8343 | 0.5349    | 0.4556 | 0.4127 | 0.3707 |
| Random Forest       | 0.5707   | 0.8784 | 0.5755    | 0.5707 | 0.5647 | 0.4629 |

## Observations

| ML Model Name       | Observations                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performed well on visually/geometrically distinct classes (Stains, K_Scatch, Z_Scratch) but struggled badly on overlapping classes, especially Other_Faults and Pastry, reflecting the limits of a linear decision boundary when class distributions overlap in feature space                                                                                                                                             |
| Decision Tree       | Showed the highest raw recall on Other_Faults among all models, likely because a single deep tree can carve out irregular decision regions that hug this heterogeneous class — but this came at the cost of weaker performance on Dirtiness and Pastry, consistent with a single tree overfitting to training noise and generalizing less reliably than an ensemble                                                       |
| kNN                 | Performed strongly on Other_Faults and K_Scatch relative to other models, but was the weakest model on Pastry, suggesting Pastry samples don't form a tight, well-separated neighborhood in the scaled 27-dimensional feature space — a known weakness of distance-based methods in higher-dimensional or overlapping feature spaces                                                                                      |
| Naive Bayes         | Weakest overall performer — heavily over-predicted the Bumps class across almost every true class (e.g. 780 Other_Faults samples and 213 Pastry samples misclassified as Bumps), indicating the Gaussian feature-independence assumption does not hold for these correlated geometric/luminosity measurements, inflating Bumps recall artificially while degrading precision elsewhere                                    |
| Random Forest       | Delivered the most balanced performance across all seven classes — matched or exceeded Logistic Regression and kNN on the well-separated classes (K_Scatch, Stains) while also substantially improving on the hardest classes (Bumps, Other_Faults) compared to Logistic Regression, benefiting from the ensemble's ability to model non-linear, overlapping decision boundaries without a single tree's overfitting risk |
| Overall Winner      | **Random Forest** — it offered the best trade-off between accuracy on well-separated classes and resilience on ambiguous, overlapping ones (Bumps, Other_Faults, Pastry), while avoiding the overfitting seen in the single Decision Tree and the systematic bias seen in Naive Bayes.                                                                                                                                    |

## Web Application Link

https://classificationmodels-zzgjpbrvybn6bifxvt7tsn.streamlit.app
