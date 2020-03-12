---
layout: post
title:  "Age distribution analysis on Coronavirus"
excerpt: "Analysis of the age distribution of Coronavirus confirmed cases in the Republic of Korea"
date:   2020-03-08
categories: [Distribution, Coronavirus]
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

South Korea is testing people for COVID-19 at an impressive rate and faster than all other countries who have been reporting confirmed cases. This shows an impressive and significant lab capacity. The nation offers a unique opportunity for countries that are dealing with the new coronavirus.

In this post, I use publicly available data from the Official information of Coronavirus disease 2019 in South Korea, whose repository is located [here](https://github.com/jihoo-kim/Coronavirus-Dataset). You can find the entire notebook at this [link](https://github.com/ClementBM/Experiment_Coronavius/blob/master/notebook/Coronavirus_Korea_Distribution.ipynb).

## Importing the data
The database contains about 7513 people having caught the disease, whose states are "released", "isolated state" or "deceased". What's really interesting about this dataset is that number of cases have information of age and gender. After dropping patient without age or gender, the resulting dataset still have about 710 people.

Let's take a quick look at the first rows of the outcoming dataset.

| id | sex | birth_year | country | region | infection_reason | infection_order | infected_by | contact_number | confirmed_date | released_date | deceased_date | state | age |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 1 | female |1984.0 | China | filtered at airport | visit to Wuhan | 1.0 | NaN | 45.0 | 2020-01-20 | 2020-02-06 | NaN | released | 36.0 |
| 2 | male | 1964.0 | Korea | filtered at airport | visit to Wuhan | 1.0 | NaN | 75.0 | 2020-01-24 | 2020-02-05 | NaN | released | 56.0 |
| 3 | male | 1966.0 | Korea | capital area | visit to Wuhan | 1.0 | NaN | 16.0 | 2020-01-26 | 2020-02-12 | NaN | released | 54.0 |
| 4 | male | 1964.0 | Korea | capital area | visit to Wuhan | 1.0 | NaN | 95.0 | 2020-01-27 | 2020-02-09 | NaN | released | 56.0 |

## Age distribution of cases by gender
In this section we observe the a distribution of confirmed cases by sex and by state: deceased, isolated and released.
![Age distribution of cases by gender](/assets/2020-03-08/distribution-cases.png)
To get a better perspective, the following barplot shows the corresponding number of cases for each category.
![Cases count by gender and state](/assets/2020-03-08/barplot-cases.png)

## Age pyramid of Republic of Korea
To generalize the previous observation to other countries, one could balanced the age distribution by the pyramid of age. For information purpose, here is a representation of the age pyramid in South Korea.
![Korea age pyramid](/assets/2020-03-08/korea-age-pyramid.png)

## Proportion of Covid-19 cases by gender and by age, normalized by the age pyramid
For all cases ("released", "isolated" and "deceased"), here is the age distribution normalized/balanced by the previous age pyramid.
![Age distribution of cases](/assets/2020-03-08/normed-disitribution-cases.png)
If the virus had spread uniformaly across the population, this distribution should be quite flat. However it doesn't seem to be the case. We have a huge proportion of very old people having caught the virus ! Let's see zoom in the data to see how many centenaries are took a census in South Korea:

| Age range | male (count) | female (count) | male (%) | female (%) |
| -- | -- | -- | -- | -- | -- |
| 100+ | 517 | 3350 | 0.002% | 0.0131% |

And how many of them had contracted the virus:

| patient_id | sex | birth_year | state | age | age_range |
| -- | -- | -- | -- | -- | -- |
| 5661 | female | 1905 | isolated | 115 | 100+ |
| 6070 | male | 1903 | isolated | 117 | 100+ |
| 6845 | female | 1913 | isolated | 107 | 100+ |
| 7219 | female | 1920 | isolated | 100 | 100+ |

Apparently how have a guiness record here ! Well, more seriously, despite the fact that

![Age distribution of deceased cases](/assets/2020-03-08/normed-disitribution-deceased-cases.png)

# Sources
* [Wiki](https://en.wikipedia.org/wiki/2020_coronavirus_outbreak_in_South_Korea)
* [South Korea](https://abcnews.go.com/international/massive-coronavirus-testing-program-south-korea-underscores-nimble/story?id=69226222)
* Intro has been written with the help of [gpt-2](https://github.com/openai/gpt-2)