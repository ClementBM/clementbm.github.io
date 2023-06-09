---
layout: post
title:  "Prevision of the Instragram engagement"
excerpt: "Heterogeneous data"
date:   2023-06-08
categories: [project]
tags: [xgboost, hyperopt, databricks, scikit-learn, databricks, pyspark, sparql]
---

XGBoost doesn't require a lot of preprocessing

XGBoost is optimized to leverage multiple cores.

XGBoost tend to overfit a little bit out of the box.

Take a first weak tree (subsequent trees corrected the remaining errors) model and then improve it with boosting, trying to predict the error between the model prediction and the actual data.

XGBoost specifically addresses the error modelization by finding where the model makes the most errors, and then try to minimize the error where its higher.

Hyperparameter
* tree structure (depth)
* regularization
  * pay less attention to certain column
  * model weights
  * classification weights for imbalance data
* Number of trees, estimators, linked to the learning rate

It seems that the learning rate is not that important.

Taking the residuals and trying to correct them.

Working well with missing values and categorical values.

# Data collection

# Data exploration

# Data preparation
No language model, stay simple for a baseline.

Make the data tabular to enable XGBoost to process it.

# Model selection

# Hyperparameter tuning
Hyperopt tuning

Stepwise tuning if not a lot of time:
* Tree first tuning
* Regularization tuning then after


# Evaluation

# TODO
* collect data
  * with instagram api
  * with ELK
  * from existing database

# Source
* [Instagram Influencer Dataset](https://github.com/ksb2043/instagram_influencer_dataset)
  * https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
* 681: XGBoost, The Ultimate Classifier, with Matt Harrison