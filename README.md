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

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1  | MCC |
| ------------------- | -------- | --- | --------- | ------ | --- | --- |
| Logistic Regression |          |     |           |        |     |     |
| Decision Tree       |          |     |           |        |     |     |
| kNN                 |          |     |           |        |     |     |
| Naive Bayes         |          |     |           |        |     |     |
| Random Forest       |          |     |           |        |     |     |

## Observations

| ML Model Name       | Observations |
| ------------------- | ------------ |
| Logistic Regression |              |
| Decision Tree       |              |
| kNN                 |              |
| Naive Bayes         |              |
| Random Forest       |              |

## Web Application Link
