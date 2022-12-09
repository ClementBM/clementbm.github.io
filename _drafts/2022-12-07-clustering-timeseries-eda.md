---
layout: post
title:  "Find insight in time series data using clustering"
excerpt: "France energy production/consumption insight time series clustering"
date:   2022-12-07
categories: [study]
tags: [time series, dtw, clustering]
---

## Dynamic Time Warping metric

Dynamic time warping (DTW) was introduced in order to overcome some of the restrictions of simpler similarity measures such as Euclidean distance.
It is one of the most popular measures, The one-nearest leighbor classifier with the DTW metric is often considered to be the baseline algorithm for time series classification.
The DTW costructs a *mn* matrix of squared distances between points of both time series, which is then used as a cost matrix when searching for the cheapest path between (1,1) and (m,n). Path cost determines the similarity.

It's possible to normalize DTW given the step pattern, dividing the distance by n, m or n+m, depending on the step pattern and slope weighting.
Possible to compare time series of different lengths.
Time complexity is O(nm)
It's not invariant to scaling, so the scaling methods will influence calculated metric and therefore the clustering results.

The dynamic time warping score is defined as the minimum cost among all the warping paths:
$$
DTW(X,Y) = \min_{p \in \mathcal{P}} C_p(X,Y)
$$

where $$\mathcal{P}$$ is the set of warping paths.

## DBA DTW Barycenter Averaging
DTW barycenter Averaging method estimated through Excpectation-Maximization algorithm.

DBA was originally presented in [1]. This implementation is based on a idea from [2] (Majorize-Minimize Mean Algorithm).





# References
[1]	F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
[2]	D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods for Averaging in Dynamic Time Warping Spaces. Pattern Recognition, 74, 340-358.

https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
https://pyts.readthedocs.io/en/stable/auto_examples/approximation/plot_paa.html
https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html
https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

https://www.insee.fr/fr/statistiques/3973175#documentation
https://odre.opendatasoft.com/explore/dataset/eco2mix-metropoles-tr/information/?disjunctive.libelle_metropole&disjunctive.nature&sort=-date&refine.date_heure=2022&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJsaW5lIiwiZnVuYyI6IlNVTSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNFODUyNTIiLCJ5QXhpcyI6ImNvbnNvbW1hdGlvbiJ9XSwieEF4aXMiOiJkYXRlX2hldXJlIiwibWF4cG9pbnRzIjoyMDAsInRpbWVzY2FsZSI6Im1pbnV0ZSIsInNvcnQiOiIiLCJzZXJpZXNCcmVha2Rvd25UaW1lc2NhbGUiOiIiLCJjb25maWciOnsiZGF0YXNldCI6ImVjbzJtaXgtbWV0cm9wb2xlcy10ciIsIm9wdGlvbnMiOnsiZGlzanVuY3RpdmUubGliZWxsZV9tZXRyb3BvbGUiOnRydWUsImRpc2p1bmN0aXZlLm5hdHVyZSI6dHJ1ZSwic29ydCI6Ii1kYXRlIiwicmVmaW5lLmRhdGVfaGV1cmUiOiIyMDIyIn19fV0sImRpc3BsYXlMZWdlbmQiOnRydWUsImFsaWduTW9udGgiOnRydWV9
https://www.data.gouv.fr/fr/datasets/donnees-du-signal-ecowatt-a-partir-du-01-09-2022/
https://opendata.edf.fr/explore/?sort=modified&disjunctive.theme&disjunctive.publisher&disjunctive.keyword
https://www.rte-france.com/eco2mix/synthese-des-donnees?type=consommation


https://www.monde-diplomatique.fr/2021/02/DEBREGEAS/62795
https://www.monde-diplomatique.fr/2021/11/BERNIER/64005
https://www.monde-diplomatique.fr/2021/05/RIMBERT/63053