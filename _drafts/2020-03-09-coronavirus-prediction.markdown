---
layout: post
title:  "Coronavirus prediction based on current data"
excerpt: "Predictions with Sigmoid and SIR based on coronavirus data"
date:   2020-03-09
categories: [Prediction, Coronavirus]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

## Prediction with S curve

Taking the sigmoid function as the S curve:
$$
y(x)
=
{
    c
    \over
    1 + e^{-k(x - x_0)}
}
+ y0
$$

## Prediction with gaussian

Taking the gaussian:
$$
c
{
    e^{
        {
            -(x - x_0)^2
            \over
            2 \sigma^2
        }
    }
}
$$

## Prediction with SIR model
To understand the trend of infection, we will use mathematical epidemic model. We start to discuss the trend using a basic model named SIR model.

### Constant
* $$N$$: (fixed) population of N individuals

### Functions
* $$S(t)$$ are those susceptible but not yet infected with the disease
* $$I(t)$$ is the number of infectious individuals
* $$R(t)$$ are those individuals who have recovered from the disease and now have immunity to it

### Coefficients
* $$\beta$$ describes the effective contact rate of the disease: an infected individual comes into contact with βN other individuals per unit time (of which the fraction that are susceptible to contracting the disease is S/N) [see](https://en.wikipedia.org/wiki/Transmission_risks_and_rates)
* $$\gamma$$ is the mean recovery rate: that is, 1/γ is the mean period of time during which an infected individual can pass it on.
(simply the rate of recovery or death). If the duration of the infection is denoted D, then γ = 1/D, since an individual experiences one recovery in D units of time. 

$$
\begin{align}
    { d S \over d t}
    &=
    - { \beta SI \over N}\\
    { d I \over d t}
    &=
    { \beta SI \over N}\ - \gamma I\\
    { d R \over d t}
    &=
    \gamma I
\end{align}
$$

# Sources

* https://kitchingroup.cheme.cmu.edu/blog/2013/02/18/Fitting-a-numerical-ODE-solution-to-data/
* https://mail.python.org/pipermail/scipy-user/2008-March/016057.html
http://sherrytowers.com/2013/01/29/neiu-lecture-vi-fitting-the-parameters-of-an-sir-model-to-influenza-data/
* http://www.cidrap.umn.edu/news-perspective/2020/02/study-72000-covid-19-patients-finds-23-death-rate
* https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

* https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca

* https://ourworldindata.org/coronavirus

**Testing**
* https://www.worldometers.info/coronavirus/covid-19-testing/
* https://www.cdc.gov/coronavirus/2019-ncov/testing-in-us.html

**Italy**
* https://github.com/pcm-dpc/COVID-19/blob/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csvhttps://www.cdc.gov/coronavirus/2019-ncov/testing-in-us.html

https://www.theatlantic.com/technology/archive/2020/03/how-understand-your-states-coronavirus-numbers/607921/