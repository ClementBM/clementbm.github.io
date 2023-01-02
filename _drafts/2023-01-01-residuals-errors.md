---
layout: post
title:  "Prediction error in time series"
excerpt: "France energy consumption prediction"
date:   2022-12-17
categories: [study]
tags: [time series, prediction]
---


https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html

bartlett_confintbool, default True

    Confidence intervals for ACF values are generally placed at 2 standard errors around r_k. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2 in [2]. For the ACF of raw data, the standard error at a lag k is found as if the right model was an MA(k-1). This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an MA of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett’s formula. For more details on Bartlett formula result, see section 7.2 in [1].


## Residuals analysis
The residuals are the prediction errors
$$
\hat{W_t} = (X_t - \hat{X_t})
$$
