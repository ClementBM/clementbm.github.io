---
layout: post
title:  "Analyzing a study on the correlation between chocolate consumption and cognitive function"
excerpt: ""
date:   2023-01-25
categories: [study]
tags: [chocolate, correlation]
---

![Chocolate consumption](/assets/2023-01-25/chocolate-consumption-by-nobel-laureate.png){: width="100%"  }

In this post, we briefly analyse a note on [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf), by *Franz H. Messerli*.

The author of the study stated that given the results of a study showing that rats improved cognitive performance after the administration of a cocoa polyphenolic, it should enhance human cognition or at least reducing natural senility due to aging.

> Improved cognitive performance with the administration of a cocoa polyphenolic extract has even been reported in aged WistarUnilever rats.
>
> Bisson JF, Nejdi A, Rozan P, Hidalgo S, Lalonde R, Messaoudi M. Effects of long-term administration of a cocoa polyphenolic extract (Acticoa powder) on cognitive performances in aged rats.

As stated in the note, the author take the *total number of Nobel laureates per capita* as a "surrogate" to cognitive performance. Dependance between Nobel Prices per capita and cognitive performance of a population, without knowing exactly what cognitive performance means and how is it measured. It presupposed that the number of *Nobel laureates per capita* is a dependent (concordant ?) of the average cognitive performance of a country.

> Conceivably, however, the total number of Nobel laureates per capita could serve as a surrogate end point reflecting the proportion with superior cognitive function and thereby give us some measure of the overall cognitive function of a given country.
>
> Author

- [Note short description](#note-short-description)
- [Study limitations given by the note](#study-limitations-given-by-the-note)
  - [Dependence between chocolate laureate consumption and mean consumption](#dependence-between-chocolate-laureate-consumption-and-mean-consumption)
  - [Time dependent variables](#time-dependent-variables)
  - [Correlation VS Causation](#correlation-vs-causation)
- [Methodology analysis](#methodology-analysis)
  - [Nobel Price per capita as an indicator](#nobel-price-per-capita-as-an-indicator)
    - [Mean aged of Nobel Prices winners](#mean-aged-of-nobel-prices-winners)
    - [Cognitive function](#cognitive-function)
    - [Sources of Flavonoids](#sources-of-flavonoids)
    - [Nobel Laureates](#nobel-laureates)
  - [Access to chocolate](#access-to-chocolate)
    - [Price - Wealth](#price---wealth)
    - [Habits - Culture](#habits---culture)
    - [Quality - Concentration of cocoa polyphenolic](#quality---concentration-of-cocoa-polyphenolic)
  - [Reproductibility and sampled data](#reproductibility-and-sampled-data)
    - [Only 23 countries out of more than 200](#only-23-countries-out-of-more-than-200)
    - [Chocolate consumption data from a Swiss Food Company](#chocolate-consumption-data-from-a-swiss-food-company)
  - [Dependency metric pearson coefficient](#dependency-metric-pearson-coefficient)
- [Retry](#retry)
- [Sources](#sources)


# Note short description
The source of the data:
* per capita Nobel laureates
* per capita yearly chocolate consumption (swiss company private? data)
* limited on 22 countries

* Hypothesis 1: chocolate consumption improve significantly cognitive function
* Hypothesis 2: chocolate consumption **does not significantly** improve cognitive function

> Obviously, these findings are hypothesis-generating only and will have to be tested in a prospective, randomized trial

> It remains to be determined whether the consumption of chocolate is the underlying mechanism for the observed association with improved cognitive function.


# Study limitations given by the note

## Dependence between chocolate laureate consumption and mean consumption

> The present data are based on country averages, and the specific chocolate intake of individual Nobel laureates of the past and present remains unknown.
> 
> The cumulative dose of chocolate that is needed to sufficiently increase the odds of being asked to travel to Stockholm is uncertain.

## Time dependent variables
> This research is evolving, since both the number of Nobel laureates and chocolate consumption are time-dependent variables and change from year to year.

## Correlation VS Causation
> Of course, a correlation between X and Y does not prove causation but indicates that either X influences Y, Y influences X, or X and Y are influenced by a common underlying mechanism.

# Methodology analysis

## Nobel Price per capita as an indicator
Nobel price is about Chemistry, Literature, Peace, Physics, and Physiology or Medicine. It doesn't encompass Mathematic since Alfred Nobel wasn't a fan of mathematicians. 

Include Abel Prize, mathematic outside of the Nobel scope

Nobel Peace Prize, biaised until 1990, Nobel Peace Center in Oslo
> From the 1990s onwards, many prizes were awarded for human rights, and the peace prize became truly global

Geir Lundestad, Secretary of the Norwegian Nobel Committee in 2006, said, "The greatest omission in our 106-year history is undoubtedly that Mahatma Gandhi never received the Nobel Peace prize. Gandhi could do without the Nobel Peace prize, [but] whether Nobel committee can do without Gandhi is the question"

### Mean aged of Nobel Prices winners
A short exploratory analysis of the population of nobel price laureates shows that laureates have around sixty years old or more.

Moving average of laureates, only 6% of laureates are female.

### Cognitive function
Cognitive function is certainly dependent on a lot of variables.

> For optimal cognitive function and intellectual achievement, it's crucial to maintain a balanced diet, engage in regular physical activity, get enough sleep, manage stress levels, and pursue intellectual stimulation through education and learning opportunities. These factors, along with a supportive environment and access to resources, are more likely to contribute to intellectual achievement than chocolate consumption alone.

![alt](/assets/2023-01-25/cognitive-function-Jane-s-Paulsen.jpg)

### Sources of Flavonoids

ChatGPT
> The potential benefits of flavonoids can also be obtained from other dietary sources like fruits, vegetables, and teas.

### Nobel Laureates

ChatGPT
> Intellectual or academic achievements.
> 
> There are numerous other factors at play, such as investment in education, research infrastructure, socioeconomic conditions, and cultural values, that have a more significant impact on the number of Nobel laureates in a given country.

## Access to chocolate
As chocolate is a non-necessary goods, unlike rice or eggs, inhabitant in difficult situations, may not priorities chocolate consumption if they are already struggling to find a decent everyday meal.

ChatGPT
> Chocolate consumption is primarily influenced by cultural, economic, and personal preferences.

### Price - Wealth
If chocolate consumption is mainly dependent on average wealth, than chocolate consumption wouldn't be a "fiable" measure because of its high dependent nature. Wealth sensitive.

### Habits - Culture
Chocolate consumption may also be culture sensitive.

Price and habits might "influence" chocolate consumption but theses factors are not a big deal.

### Quality - Concentration of cocoa polyphenolic
Average consumption per capita need to .. 
High consumption of chocolate with a low polyphenolic concentration, might invalidate study. At the contrary a low consumption of chocolate with a high polyphenolic concentration... 

ChatGPT comments on:
> Moreover, it's worth considering that chocolate is often consumed in various forms, ranging from dark chocolate with high cocoa content to milk chocolate with added sugars and fats. The health benefits associated with chocolate are primarily attributed to dark chocolate with higher cocoa content, as it contains a higher concentration of flavonoids and fewer added ingredients.

## Reproductibility and sampled data
Giving the data used during the study is not available, it's hard to reproduce the same result.

### Only 23 countries out of more than 200
Adapt r statistic, relative to this country sample?

### Chocolate consumption data from a Swiss Food Company

## Dependency metric pearson coefficient
r=0.791, P<0.0001

Missing a confidence interval

# Retry

![alt](/assets/2023-01-25/chocolate-consumption-per-capita.png)

![alt](/assets/2023-01-25/nobel-country-per-capita.png)

# Sources
https://www.radiofrance.fr/franceinter/podcasts/la-terre-au-carre/la-terre-au-carre-du-mardi-10-janvier-2023-8979117

* [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf)
* [Evolution of National Nobel Prize Shares in the 20th Century (by Juergen Schmidhuber)](https://people.idsia.ch/~juergen/all.html)https://people.idsia.ch/~juergen/nobelshare.pdf
* https://www.liberation.fr/sciences/des-prix-nobel-toujours-tres-masculins-et-de-plus-en-plus-ages-20211011_YKTNPXRUZVB7FBWZCANHTWHAOQ/
* https://public.opendatasoft.com/explore/dataset/nobel-prize-laureates/table/?flg=fr&disjunctive.category
* https://en.wikipedia.org/wiki/List_of_countries_by_Nobel_laureates_per_capita
* https://www.fao.org/faostat/en/#data
* https://ourworldindata.org/grapher/chocolate-consumption-per-person?time=2008&country=AUS~DMA~NAM
* https://damecacao.com/chocolate-statistics/#12
* [La vraisemblance](https://www.youtube.com/watch?v=P-AHaAP8fIk)
* [Initiation à la statistique bayésienne](https://www.youtube.com/watch?v=5hN_plbtPjw)
* https://www.numbeo.com/food-prices/
* https://ourworldindata.org/grapher/cost-calorie-sufficient-diet?country=USA~BHR~GBR~IND~CHN~BRA~ZAF~FRA~TCD
* https://ideas.ted.com/the-steep-price-we-pay-for-cheap-chocolate/
* https://altoida.com/blog/defining-the-6-key-domains-of-cognitive-function/
