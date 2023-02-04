---
layout: post
title:  "Analyzing a study on the correlation between chocolate consumption and cognitive function"
excerpt: ""
date:   2023-01-25
categories: [study]
tags: [chocolate, correlation]
---

![Chocolate consumption](/assets/2023-01-25/chocolate_consumption.png){: width="100%"  }

In this post, we briefly analyse a note on [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf), by *Franz H. Messerli*.

The author of the study stated that given the results of a study showing that rats improved cognitive performance after the administration of a cocoa polyphenolic, that it could be rather similar for human, enhancing or at least reducing natural senility due to aging.

> Improved cognitive performance with the administration of a cocoa polyphenolic extract has even been reported in aged WistarUnilever rats.
>
> Bisson JF, Nejdi A, Rozan P, Hidalgo S, Lalonde R, Messaoudi M. Effects of long-term administration of a cocoa polyphenolic extract (Acticoa powder) on cognitive performances in aged rats.

As stated in the note, the author take the *total number of Nobel laureates per capita* as a "surrogate". Concordance, dependance between Nobel Prices per capita and cognitive performance of a population, without knowing exactly what cognitive performance means and how is it measured.

> Conceivably, however, the total number of Nobel laureates per capita could serve as a surrogate end point reflecting the proportion with superior cognitive function and thereby give us some measure of the overall cognitive function of a given country.
>
> Author

The source of the data:
* per capita Nobel laureates
* per capita yearly chocolate consumption (swiss company private? data)
* limited on 22 countries

* Hypothesis 1: chocolate consumption improve significantly cognitive function
* Hypothesis 2: chocolate consumption **does not significantly** improve cognitive function


> Of course, a correlation between X and Y does not prove causation but indicates that either X influences Y, Y influences X, or X and Y are influenced by a common underlying mechanism.

> Obviously, these findings are hypothesis-generating only and will have to be tested in a prospective, randomized trial

## Study Limitations

> The present data are based on country averages, and the specific chocolate intake of individual Nobel laureates of the past and present remains unknown.
> 
> The cumulative dose of chocolate that is needed to sufficiently increase the odds of being asked to travel to Stockholm is uncertain.
> 
> This research is evolving, since both the number of Nobel laureates and chocolate consumption are time-dependent variables and change from year to year.

## Conclusions

> It remains to be determined whether the consumption of chocolate is the underlying mechanism for the observed association with improved cognitive function.

# Comments

## Nobel Price Country unequal distribution 
Show a distribution of Nobel Price by country.

![alt](/assets/2023-01-25/chocolate_consumption_by_country.png)

Nobel Peace Prize, biaised until 1990
Include Abel Prize, mathematic outside of the Nobel scope

Geir Lundestad, Secretary of the Norwegian Nobel Committee in 2006, said, "The greatest omission in our 106-year history is undoubtedly that Mahatma Gandhi never received the Nobel Peace prize. Gandhi could do without the Nobel Peace prize, [but] whether Nobel committee can do without Gandhi is the question"

Nobel Peace Center in Oslo
> From the 1990s onwards, many prizes were awarded for human rights, and the peace prize became truly global
 
![...](/assets/2023-01-25/nobel-country-per-capita.png)

## Mean aged of Nobel Prices winners
A short exploratory analysis of the population of nobel price laureates shows that laureates have around sixty years old or more.

Moyenne glissante of laureates

Only 6% of laureates are female.

## Access to chocolate
Price, Quantity, Quality, Black ?

Culture

## Reproductibility
Giving the data used during the study is not available, it's hard to reproduce the same result.

## Chosen metric: pearson coefficient
r=0.791, P<0.0001

Missing a confidence interval

## Only 23 countries out of 200
Adapt r statistic, relative to this country sample?

## Ambitious conclusion
(Chemistry, Literature, Peace, Physics, and Physiology or Medicine)

Cognitive function is certainly dependant on a lot of variables.


# Sources
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

![alt](/assets/2023-01-25/nelson-mandela.jpg)