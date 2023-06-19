---
layout: post
title:  "Find insight in time series data using clustering"
excerpt: "insight time series clustering"
date:   2022-12-07
categories: [study]
tags: [time series, dtw, clustering]
---

- [Why clustering time series?](#why-clustering-time-series)
- [The metric: Dynamic Time Warping](#the-metric-dynamic-time-warping)
- [The prototype: DBA DTW Barycenter Averaging](#the-prototype-dba-dtw-barycenter-averaging)
- [The preprocesing method: mean variance scaler](#the-preprocesing-method-mean-variance-scaler)
- [The clustering algorithm: DBA(DTW Barycenter Averaging)-k-means](#the-clustering-algorithm-dbadtw-barycenter-averaging-k-means)
- [The clustering evaluation: the Silhouette score](#the-clustering-evaluation-the-silhouette-score)
- [The use case, data Webstat](#the-use-case-data-webstat)
- [References](#references)

# Why clustering time series?
With clustering we try to find subgroups within the dataset. This is an unsupervised problem because we are trying to discover structure, distinct cluster for instance, on the basis of a data set.

Both clustering and PCA seek to simplify the data via a small number of summaries, but their mechanisms are different:
* PCA looks to find a low-dimensional representation of the observations that explain a good fraction of the variance
* Clustering looks to find homogeneous subgroups among the observations 

# The metric: Dynamic Time Warping

One of the most popular measures for time series, dynamic time warping (DTW) was introduced in order to overcome some of the restrictions of simpler similarity measures such as Euclidean distance:
1. that only works with time series of equal lengths
   1. this is not necessarily an advantage, as it has been shown that performing linear reinterpolation to obtain equal length may be appropriate if m and n do not vary significantly
2. that compares the values of both time series at each point independently (as the values of time series are often time correlated with different lags)

The one-nearest neighbor classifier with the DTW metric is often considered to be the baseline algorithm for time series classification.

The DTW constructs a *mn* matrix of squared distances between points of both time series, which is then used as a cost matrix when searching for the cheapest path between (1,1) and (m,n). Path cost determines the similarity.

* It's possible to normalize DTW given the step pattern, dividing the distance by n, m or n+m, depending on the step pattern and slope weighting.
* It's possible to compare time series of different lengths.
* It's sensitive to scaling, so the scaling methods will influence the metric and therefore the clustering results
* For multivariate times series the *Dynamic Time Wraping* "metric" is sensitive to the normalization method
* Time complexity is $$O(nm)$$

The dynamic time warping score is defined as the minimum cost among all the warping paths:

$$
DTW(X,Y) = \min_{p \in \mathcal{P}} C_p(X,Y)
$$

where $$\mathcal{P}$$ is the set of warping paths.

# The prototype: DBA DTW Barycenter Averaging
DTW Barycenter Averaging method is estimated through Expectation-Maximization algorithm.

DBA was originally presented in [1]. This implementation is based on a idea from [2] (Majorize-Minimize Mean Algorithm).

The procedure is called DTW Barycenter Averaging, and is an iterative, global method. The latter means that the order in which the series enter the prototyping function does not affect the outcome.

DBA requires a series to be used as reference (centroid), and it usually begins by randomly electing one of the series in the data. On each iteration, the DTW alignement between each series in the cluser $$C$$ and the centroid is computed.

{% gist 109210aa102732b41dd8635b4b6a054e %}

# The preprocesing method: mean variance scaler
time series are preprocessed using TimeSeriesScalerMeanVariance. This scaler is such that each output time series has zero mean and unit variance. The assumption here is that the range of a given time series is uninformative and one only wants to compare shapes in an amplitude-invariant manner.

This means that one cannot scale barycenters back to data range because each time series is scaled independently and there is hence no such thing as an overall data range.

https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py

# The clustering algorithm: DBA(DTW Barycenter Averaging)-k-means
K nearest neighboor
One nearest neigboor

The data is explicitly assigned to one and only one cluster. The total number of desired clusters must be specified beforehand, which can be a limiting factor.

k-means, is a partitional procedure that can be stated as a combinatorial optimization problems that minimize the intra-cluster distance while maximimzing the inter-cluster distance. Finding the global optimum would require enumerating all possible gorupins, seomething which is infeasible even for relatively small datasets. Therefore iterative freedy descent strategies are used instead, which examine a small fraction of the search space until convergence, but could converge to local optima.

Partitional clustering algorithms commonly work in the following way. First, $$k$$ centroids are randomly initialized, usually by choosing $$k$$ objects from the dataset at random; these are assigned to individual clusters. The distance between all objects in the data and all centroid is calculated, and each object is assigned to the cluster of its closest centroid. A prototyping function is applied to each cluster to update the corresponding centroid. Then, the distances and centroids are updated iteratively until a certain number of iterations have elapsed or no objects changes clusters any more.

Partitional clustering procedures are stochastic due to their random start. Thus, it is common practice to test different random starts to evaluate local optima and choose the best result out of all repetitions. It tends to produce spherical clusters, but has a lower complexity, so it can be applied to very large datasets.

# The clustering evaluation: the Silhouette score
float: Mean Silhouette Coefficient for all samples.

For any data point $$ i \in C_I $$ let $$a(\cdot)$$ be a measure of how well $$i$$ is assigned to its cluster, the smaller the value the better the assignement:

$$
a(i) = { 1 \over | C_I | - 1 } \sum_{j \in C_I, i \neq j} d(i,j)
$$

Then let $$b(\cdot)$$ be the mean dissimilarity of point $$i$$ to its neigbooring cluster $$C_N$$, the mean distance of $$i$$ to all points in the closest cluster, where $$C_N$$ is a subset of $$C_J$$ and $$C_N \neq C_I$$

$$
b(i) = \min_{J \neq I} { 1 \over | C_J | }\sum_{j \in C_J} d(i,j)
$$

And finally, let $$s(\cdot)$$ be the silouhette value for any given data point $$i$$:

$$
s(i) = { b(i) - a(i) \over \max\{ a(i), b(i) \} } \text{, if }|C_I| > 1
$$

Then mean of the silouhette scores across all data points $$i$$, is a measure of how grouped all the points are, how well the clusters are well formed.

$$
mean \left ( s(i) \right ) = { 1 \over n }\sum_{i} { b(i) - a(i) \over \max \{ a(i), b(i) \} }
$$

Standardization of cluster evaluation metrics by using the cluster validity indices (CVIs).

CVI are either crips or fuzzy partitions. CVIs can be internal, external or relative depending on how they are computed. Internal CVIs only consider the partitioned data and try to define a measure of cluster purity, whereas external CVIs compare the obtained partition to the correct one.

In many cases, these CVIs can be used to evaluate the result of a clustering algorithm regardless of how the clustering works internally, or how the partition came to be. The silouhette index, is an example of an internal CI.

> Knowing which CVI will work best cannot be determined a priori, so they should be tested for each specific application. Many CVIs can be utilized and compared to each other, maybe using a majority vote to decide on a final result, but here is no best CVI, and it is important to conceptually understand what a given CVI measerues in order to appropriately interpret its results.


> Furthermore, it should be noted that, due to additional distance and/or centroid calculations, computing CVIs can be prohibitive in some cases. For example, the SIlhouette index effectively needs the whole distance matrix between the original series to be calculated.

{% gist 109210aa102732b41dd8635b4b6a054e slicing.py %}


* [Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis". Computational and Applied Mathematics 20: 53-65.](http://www.sciencedirect.com/science/article/pii/0377042787901257)
* [Wikipedia entry on the Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))


# The use case, data [Webstat](https://api.gouv.fr/les-api/webstat)
Webstat est le portail statistique de la Banque de France. L'API Webstat permet d'accéder à plus de 35.000 séries statistiques de la Banque de France et de ses partenaires institutionnels. Obtenez simplement les données économiques et financières sur les entreprises françaises, la conjoncture régionale, le crédit et l'épargne, la monnaie ou la balance des paiements.

Principales fonctionnalités: 

# References
* F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
* D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods for Averaging in Dynamic Time Warping Spaces. Pattern Recognition, 74, 340-358.
* https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
* https://pyts.readthedocs.io/en/stable/auto_examples/approximation/plot_paa.html
* https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html
* https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

