---
layout: post
title:  "Prevision of the Instragram engagement"
excerpt: "Heterogeneous data"
date:   2023-06-08
categories: [project]
tags: [xgboost, hyperopt, databricks, scikit-learn, databricks, pyspark, sparql]
---


Advantages of XGBoost:
* XGBoost doesn't require a lot of preprocessing
* XGBoost is optimized to leverage multiple cores and on GPUs
* Working well with missing values and categorical values
* capture non linear data, complex relation in the data
* can get relationships, interaction bewteen the features/variables
* making EDA with XGBoost, get new insight
* not easily interpretable, not a white box model
* implemented in C++ under the hood

Limitations of XGBoost:
* XGBoost tend to overfit a little bit out of the box
* Doesn't work well on non tabular data, like text, image or audio


**How does it work?**

Take a first weak tree (subsequent trees corrected the remaining errors) model and then improve it with boosting, trying to predict the error between the model prediction and the actual data.

XGBoost specifically addresses the error modelization by finding where the model makes the most errors, and then try to minimize the error where its higher.

Taking the residuals and trying to correct them.

**What are the hyperparameters?**

* tree structure (depth)
* regularization
  * pay less attention to certain column
  * model weights
  * classification weights for imbalance data
* Number of trees, estimators, linked to the learning rate

It seems that the learning rate is not that important.


# Data collection

# Data exploration
## [Yellowbrick](https://rebeccabilbro.github.io/xgboost-and-yellowbrick/)

Vizualization library
A little more advance that scikit learn


## XGB FYR, for explaining feature relation

Pair features that go on after the other in trees, might indicate there is relationship between those features.

Take the output of XGBoost and look for potential interactions in the data

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

# Interpretation

# Deploy into production


# Monitoring
Take care of not having drift

# TODO
* collect data
  * with instagram api
  * with ELK
  * from existing database

# Source
* [Instagram Influencer Dataset](https://github.com/ksb2043/instagram_influencer_dataset)
  * https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
* 681: XGBoost, The Ultimate Classifier, with Matt Harrison