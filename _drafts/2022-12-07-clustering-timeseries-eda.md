---
layout: post
title:  "Find insight in time series data using clustering"
excerpt: "France energy production/consumption insight time series clustering"
date:   2022-12-07
categories: [study]
tags: [time series, dtw, clustering]
---

- [Hypothesis](#hypothesis)
- [Dynamic Time Warping metric](#dynamic-time-warping-metric)
- [DBA DTW Barycenter Averaging](#dba-dtw-barycenter-averaging)
- [On clustering](#on-clustering)
- [Silhouette](#silhouette)
- [Webstat](#webstat)
- [References](#references)

# Hypothesis
Electricity generation, consumption data sources are not always used at 100% of their capacities due to:
* maintenance
* failures
* power consumption needs

# Dynamic Time Warping metric

Dynamic time warping (DTW) was introduced in order to overcome some of the restrictions of simpler similarity measures such as Euclidean distance.
It is one of the most popular measures, The one-nearest neighbor classifier with the DTW metric is often considered to be the baseline algorithm for time series classification.
The DTW constructs a *mn* matrix of squared distances between points of both time series, which is then used as a cost matrix when searching for the cheapest path between (1,1) and (m,n). Path cost determines the similarity.

It's possible to normalize DTW given the step pattern, dividing the distance by n, m or n+m, depending on the step pattern and slope weighting.
Possible to compare time series of different lengths.
Time complexity is $$O(nm)$$
It's not invariant to scaling, so the scaling methods will influence the metric and therefore the clustering results.

The dynamic time warping score is defined as the minimum cost among all the warping paths:
$$
DTW(X,Y) = \min_{p \in \mathcal{P}} C_p(X,Y)
$$

where $$\mathcal{P}$$ is the set of warping paths.

# DBA DTW Barycenter Averaging
DTW barycenter Averaging method estimated through Excpectation-Maximization algorithm.

DBA was originally presented in [1]. This implementation is based on a idea from [2] (Majorize-Minimize Mean Algorithm).

# On clustering
With clustering we try to find subgroups within the dataset. This is an unsupervised problem because we are trying to discover structure, distinct cluster for instance, on the basis of a data set. On the other hand, the goal in supervised problems, is to try to predict some outcome vector or label.
Both clustering and PCA seek to simplify the data via a small number of summaries, but their mechanisms are different:
* PCA looks to find a low-dimensional representation of the observations that explain a good fraction of the variance
* Clustering looks to find homogeneous subgroups among the observations 

{% gist 109210aa102732b41dd8635b4b6a054e %}

# Silhouette
float: Mean Silhouette Coefficient for all samples.

{% gist 109210aa102732b41dd8635b4b6a054e slicing.py %}


[1] [Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis". Computational and Applied Mathematics 20: 53-65.](http://www.sciencedirect.com/science/article/pii/0377042787901257)
[2] [Wikipedia entry on the Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))



# [Webstat](https://api.gouv.fr/les-api/webstat)
Webstat est le portail statistique de la Banque de France. L'API Webstat permet d'accéder à plus de 35.000 séries statistiques de la Banque de France et de ses partenaires institutionnels. Obtenez simplement les données économiques et financières sur les entreprises françaises, la conjoncture régionale, le crédit et l'épargne, la monnaie ou la balance des paiements. Principales fonctionnalités:

# References
[1]	F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
[2]	D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods for Averaging in Dynamic Time Warping Spaces. Pattern Recognition, 74, 340-358.

https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
https://pyts.readthedocs.io/en/stable/auto_examples/approximation/plot_paa.html
https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html
https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

