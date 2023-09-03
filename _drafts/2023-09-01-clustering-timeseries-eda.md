---
layout: post
title:  "Find insight in time series data using clustering"
excerpt: "insight time series clustering"
date:   2023-09-01
categories: [study]
tags: [time series, dtw, clustering]
---

Clustering is an unsupervised data mining technique for organizing data series into groups based on their similarity. In this post we make a focus on clustering large number of Time Series. Though the objective of the technique remains the same (maximize data similarity within clusters and minimize it across clusters) the implementation details are adapted to this particular data type.

> Clustering is the practice of finding hidden patterns or similar groups in data. (Roelofsen, 2018)

- [Why clustering time series?](#why-clustering-time-series)
- [The Theory](#the-theory)
  - [1. Preprocessing method: Mean Variance Scaler](#1-preprocessing-method-mean-variance-scaler)
  - [2. Metric: Dynamic Time Warping](#2-metric-dynamic-time-warping)
  - [3. Prototyping with DTW Barycenter Averaging](#3-prototyping-with-dtw-barycenter-averaging)
  - [4. The algorithm: DBA-k-means](#4-the-algorithm-dba-k-means)
  - [5. Evaluation with the Silhouette Score](#5-evaluation-with-the-silhouette-score)
- [The Application, Webstat data](#the-application-webstat-data)
  - [Finding the best number of cluster](#finding-the-best-number-of-cluster)
  - [tf idf for comparing most proeminent tokens by cluster taken the time series names](#tf-idf-for-comparing-most-proeminent-tokens-by-cluster-taken-the-time-series-names)
- [References](#references)

# Why clustering time series?
Clustering is employed to identify subgroups within a dataset, constituting an unsupervised approach as it entails uncovering inherent structures, such as distinct clusters, within the dataset.

Both clustering and PCA (Principal Component Analysis) aim to simplify data by summarizing it, yet their methods diverge:
* PCA aims to discover a low-dimensional representation of the data points that can explain a good fraction of the variance.
* Clustering, on the other hand, aims to identify homogeneous subgroups among the data points.

In the context of time-series visualization, clustering serves as an exploratory technique where we create clusters while considering the entire time series as a unified entity. The primary steps involved in this process are as follows:

1. Determining a distance measure to quantify the similarity between observations 
2. Prototype, choose a centroid that summarizes the characteristics of all series in a cluster: mean, median, ...
3. Preprocessing method: min-max, standardization, ...
4. Choosing the algorithm to obtain the cluster, most common are partitional or hierarchical
5. And evaluate the results via cluster validity indices

# The Theory
## 1. Preprocessing method: Mean Variance Scaler

As the chosen metric is sensitive to scaling, we have to carefully preprocess the time series.
This scaler is such that each output time series has zero mean and unit variance. The assumption here is that the range of a given time series is uninformative and one only wants to compare shapes in an amplitude-invariant manner.

time series are preprocessed using TimeSeriesScalerMeanVariance. 

This means that one cannot scale barycenters back to data range because each time series is scaled independently and there is hence no such thing as an overall data range.

https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py

## 2. Metric: Dynamic Time Warping

Dynamic Time Warping (DTW), among the most widely utilized measures for time series analysis, was developed to address certain limitations of simpler similarity metrics like the Euclidean distance:

1. It has the ability to handle time series of varying lengths.
2. Unlike straightforward metrics, DTW doesn't merely compare the values of both time series point by point; it takes into account the temporal correlation between values, which often occurs with different lags.

DTW constructs a square matrix of size $$m \times n$$, calculating squared distances between corresponding points of the two time series. This matrix serves as a cost matrix when searching for the most economical path between points $$(1,1)$$ and $$(m,n)$$, and this path's cost quantifies the similarity.

The dynamic time warping score is defined as the minimum cost among all the warping paths:

$$
DTW(X,Y) = \min_{p \in \mathcal{P}} C_p(X,Y)
$$

where $$\mathcal{P}$$ is the set of warping paths.

Here are some noteworthy characteristics and considerations regarding DTW:

* It is sensitive to scaling, meaning that the scaling techniques applied will influence the metric and, consequently, the outcomes of clustering.
* The time complexity of DTW is proportional to $$O(nm)$$, where $$n$$ and $$m$$ represent the lengths of the compared time series.
* DTW permits the comparison of time series with dissimilar lengths. This characteristic may not necessarily be an advantage, as studies have demonstrated that employing linear reinterpolation to obtain equal length may be appropriate when the lengths of the time series, denoted as $$m$$ and $$n$$, do not vary significantly.


## 3. Prototyping with DTW Barycenter Averaging
In the clustering domain, a prototype refers to a single time series that provides a condensed representation of all the series within a cluster. In our context, we employ the DBA method to derive this prototype.

The DTW Barycenter Averaging method, or DBA method, necessitates the selection of a reference series, often referred to as a centroid, typically chosen randomly from the dataset. During each iteration of this method, the DTW alignment is calculated between the chosen centroid and each series  of the cluster $$C$$.

DBA is an iterative, global method. The latter means that the order in which the series enter the prototyping function does not affect the outcome. It is estimated through Expectation-Maximization algorithm.

{% gist 109210aa102732b41dd8635b4b6a054e %}

## 4. The algorithm: DBA-k-means
K-means enable to find clusters in an iterative way. The clusters may not be the best ones, but there are generally ok. Clustering is an NP hard problem, which means that the execution time is exponential. K-means is also non-probabilistic, which means, that we don't have an indicator of confidence about the belonging of a data point to a cluster (crips vs fuzzy partitions?)

The one-nearest neighbor classifier employing the DTW metric is often regarded as the foundational algorithm for time series classification.

K nearest neighboor
One nearest neigboor

The data is explicitly assigned to one and only one cluster. The total number of desired clusters must be specified beforehand, which can be a limiting factor.

k-means, is a partitional procedure that can be stated as a combinatorial optimization problems that minimize the intra-cluster distance while maximimzing the inter-cluster distance. Finding the global optimum would require enumerating all possible gorupins, seomething which is infeasible even for relatively small datasets. Therefore iterative freedy descent strategies are used instead, which examine a small fraction of the search space until convergence, but could converge to local optima.

Partitional clustering algorithms commonly work in the following way. First, $$k$$ centroids are randomly initialized, usually by choosing $$k$$ objects from the dataset at random; these are assigned to individual clusters. The distance between all objects in the data and all centroid is calculated, and each object is assigned to the cluster of its closest centroid. A prototyping function is applied to each cluster to update the corresponding centroid. Then, the distances and centroids are updated iteratively until a certain number of iterations have elapsed or no objects changes clusters any more.

Partitional clustering procedures are stochastic due to their random start. Thus, it is common practice to test different random starts to evaluate local optima and choose the best result out of all repetitions. It tends to produce spherical clusters, but has a lower complexity, so it can be applied to very large datasets.

## 5. Evaluation with the Silhouette Score
float: Mean Silhouette Coefficient for all samples.

Cluster validity indices (CVIs) is standardize framework for cluster evaluation metrics.

CVI are either crips or fuzzy partitions. CVIs can be internal, external or relative depending on how they are computed. Internal CVIs only consider the partitioned data and try to define a measure of cluster purity, whereas external CVIs compare the obtained partition to the correct one.

In many cases, these CVIs can be used to evaluate the result of a clustering algorithm regardless of how the clustering works internally, or how the partition came to be. The silouhette index, is an example of an internal CI.

> Knowing which CVI will work best cannot be determined a priori, so they should be tested for each specific application. Many CVIs can be utilized and compared to each other, maybe using a majority vote to decide on a final result, but here is no best CVI, and it is important to conceptually understand what a given CVI measerues in order to appropriately interpret its results.

> Furthermore, it should be noted that, due to additional distance and/or centroid calculations, computing CVIs can be prohibitive in some cases. For example, the Silhouette index effectively needs the whole distance matrix between the original series to be calculated.


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

{% gist 109210aa102732b41dd8635b4b6a054e slicing.py %}

* [Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis". Computational and Applied Mathematics 20: 53-65.](http://www.sciencedirect.com/science/article/pii/0377042787901257)
* [Wikipedia entry on the Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))

# The Application, [Webstat](https://api.gouv.fr/les-api/webstat) data
Webstat est le portail statistique de la Banque de France. L'API Webstat permet d'accéder à plus de 35.000 séries statistiques de la Banque de France et de ses partenaires institutionnels. Obtenez simplement les données économiques et financières sur les entreprises françaises, la conjoncture régionale, le crédit et l'épargne, la monnaie ou la balance des paiements.



## Finding the best number of cluster
TODO: schéma intéressant sur https://nbviewer.org/github/pycaret/examples/blob/main/PyCaret%202%20Clustering.ipynb

![Elbow curve of the number of clusters](/assets/2023-09-01/elbow-curve.png)


![Plot of the time series clusters with their centroïd](/assets/2023-09-01/ts-clusters.png)


## tf idf for comparing most proeminent tokens by cluster taken the time series names

TODO: supprimer certain tokens? "nombre cumulé sur 12 mois", "Flux cumulés 12 mois glissants", "Cumul annuel"
TODO: modifier la fonction regex: retirer les () et [] en premier, ensuite découper selon les virgules
TODO: ajouter le score tfidf
TODO: signification acronymes: CVS, CJO, 

|   cluster |   size | series                                                                                                                     |
|----------:|-------:|:---------------------------------------------------------------------------------------------------------------------------|
|         1 |     28 | Défaillances, ~~nombre cumulé sur 12 mois~~, Corse, Indice de ventes au détail, Unités légales                                 |
|         2 |     44 | Défaillances, ~~nombre cumulé sur 12 mois~~, ~~Flux cumulés 12 mois glissants~~, Unités légales, (CVS)                             |
|         3 |    112 | (CVS-CJO), Unités légales, ~~nombre cumulé sur 12 mois~~, [hors bâtiment], indice du prix à la production industrielle         |
|         4 |     26 | (CVS-CJO), [hors bâtiment], Indice de ventes au détail, (CVS), (CJO)                                                       |
|         5 |    179 | Défaillances, ~~nombre cumulé sur 12 mois~~, Unités légales, Construction, (CVS-CJO)                                           |
|         6 |     33 | Défaillances, ~~nombre cumulé sur 12 mois~~, Industrie, (CVS), Unités légales                                                  |
|         7 |     20 | (flux monétaire), Emissions nettes, Valeur nominale, Valeur de marché, (yc opérations d'apport en nature)                  |
|         8 |     42 | Défaillances, ~~nombre cumulé sur 12 mois~~, Transports et entreposage, Valeur de marché, Activités financières et d'assurance |
|         9 |     12 | Indice de production industrielle, (CJO), [hors bâtiment], Emissions nettes, de titres de dette                            |
|        10 |     21 | (CJO), Indice de ventes au détail, Finlande, Indice de production industrielle, Chypre                                     |



# References
* F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
* D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods for Averaging in Dynamic Time Warping Spaces. Pattern Recognition, 74, 340-358.
* https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
* https://pyts.readthedocs.io/en/stable/auto_examples/approximation/plot_paa.html
* https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html
* https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

