---
layout: post
title:  "Analyzing a study on the correlation between chocolate consumption and cognitive function"
excerpt: ""
date:   2023-01-25
categories: [study]
tags: [chocolate, correlation]
---

![Chocolate consumption](/assets/2023-01-25/chocolate-consumption-by-nobel-laureate.png){: width="100%"  }

In this post, we briefly analyze a note on [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf), by *Franz H. Messerli*.

According to another study showing that rats improved their cognitive performances after the administration of a cocoa polyphenolic,
the author tested the hypothesis whether chocolate consumption enhance human cognition.

As stated in the note, the author took the *total number of Nobel laureates per capita* as a "surrogate" to cognitive performance.

**Table of contents**
- [Description of the note](#description-of-the-note)
- [Cognitive function and Nobel Price per capita](#cognitive-function-and-nobel-price-per-capita)
  - [Indicator of cognitive performance](#indicator-of-cognitive-performance)
  - [Average age of Nobel Prices' winners](#average-age-of-nobel-prices-winners)
  - [Nobel Laureates](#nobel-laureates)
- [Chocolate and Flavonoids](#chocolate-and-flavonoids)
  - [Sources of Flavonoids](#sources-of-flavonoids)
  - [Price - Wealth](#price---wealth)
  - [Habits - Culture](#habits---culture)
  - [Quality - Concentration of cocoa polyphenolic](#quality---concentration-of-cocoa-polyphenolic)
- [Reproductibility and sampled data](#reproductibility-and-sampled-data)
  - [Only 23 countries out of more than 200](#only-23-countries-out-of-more-than-200)
  - [Chocolate consumption data from a Swiss Food Company](#chocolate-consumption-data-from-a-swiss-food-company)
  - [Retry attempt](#retry-attempt)
- [Dependency metric pearson coefficient](#dependency-metric-pearson-coefficient)
- [Conclusion](#conclusion)
- [Sources](#sources)

In the following section, I make a brief description of the three pages note of interest, published in the New England Journal of Medecine in 2012.

# Description of the note

* **Hypothesis 1**: chocolate consumption improve significantly cognitive function
* **Hypothesis 2**: chocolate consumption **does not significantly** improve cognitive function

The source of the data:
* Per capita Nobel laureates
* Per capita yearly chocolate consumption (swiss company private? data) of 23 countries

The results of the study:
* Pearson correlation factor $$r = 0.791$$
* Confidence $$\text{p-value} < 0.0001$$

The correlation result of $$r = 0.791$$ tend to go in the direction of a linear correlation between the chocolate consumption and the number of laureates per capita.

We can already comment that the small number of countries of the dataset could be an issue. Are the set of 23 countries a good representation of the variety of all (~200) countries?

Thereafter three other limitations are listed by the note's author himself.

**Dependence between chocolate laureate consumption and mean consumption**

> The present data are based on country averages, and the specific chocolate intake of individual Nobel laureates of the past and present remains unknown.
> 
> The cumulative dose of chocolate that is needed to sufficiently increase the odds of being asked to travel to Stockholm is uncertain.

**Time dependent variables**
> This research is evolving, since both the number of Nobel laureates and chocolate consumption are time-dependent variables and change from year to year.

**Correlation VS Causation**
> Of course, a correlation between X and Y does not prove causation but indicates that either X influences Y, Y influences X, or X and Y are influenced by a common underlying mechanism.

# Cognitive function and Nobel Price per capita
According to the dictionnary, **cognition** is the "mental action or process of acquiring knowledge and understanding through thought, experience, and the senses". Cognition is multi-dimensional in the sense that it encompases all aspects of intellectual functions. Wikipedia's page on cognition lists multiple examples of such intellectual functions: perception, attention, thought, imagination, intelligence, memory, judgment and evaluation, reasoning, problem-solving and decision-making, comprehension and production of language.

The following diagram by *Jane S. Paulsen* shows some of them grouped by category:

![alt](/assets/2023-01-25/cognitive-function-Jane-s-Paulsen.jpg)

Cognitive function of a human being may be dependent on diet, physical activity, sleep, stress levels and intellectual stimulation.

All of these mighty possible factors are dependant at different proportion of the wealth of a person, and by extent to his/her access to learning/cultural resources.

It is not clear to what proportion each of these factors influence the cognitive function, but it is clear that the dark chocolate consumption along with the diet are not the only factors that have an impact on the cognitive performance of a person.

## Indicator of cognitive performance
Is the **Nobel laureates per capita** a good indicator of cognitive performance?

The Nobel price rewards notable contribution of scientists (accross the globe) in the advance of the knowledge. Rewards span accross the fields of chemistry, literature, peace, physics and medicine. It doesn't encompass the field of mathematic since Alfred Nobel, the creator of the Nobel's price, wasn't really in good terms with mathematicians.

Caution shoud be taken specifically on the Nobel Peace Prize. Indeed, according to the Nobel Peace Center in Oslo, this prize is biaised toward the western countries until the 1990's: "From the 1990s onwards, many prizes were awarded for human rights, and the peace prize became truly global". As for the field of mathematic, we could argue that the study is missing the Abel Prize, the Nobel equivalent in the scope of mathematic.

The note presupposed that the number of **Nobel laureates per capita** is concordant with the average cognitive performance of a country.

> Conceivably, however, the total number of Nobel laureates per capita could serve as a surrogate end point reflecting the proportion with superior cognitive function and thereby give us some measure of the overall cognitive function of a given country.
> 
> Author

## Average age of Nobel Prices' winners
A short exploratory analysis of the population of nobel price laureates shows that laureates have around sixty years old or more.

Moving average of laureates, only 6% of laureates are female.

## Nobel Laureates

ChatGPT
> Intellectual or academic achievements.
> 
> There are numerous other factors at play, such as investment in education, research infrastructure, socioeconomic conditions, and cultural values, that have a more significant impact on the number of Nobel laureates in a given country.

# Chocolate and Flavonoids
> Improved cognitive performance with the administration of a cocoa polyphenolic extract has even been reported in aged WistarUnilever rats.
>
> Bisson JF, Nejdi A, Rozan P, Hidalgo S, Lalonde R, Messaoudi M. Effects of long-term administration of a cocoa polyphenolic extract (Acticoa powder) on cognitive performances in aged rats.

or at least reduce the natural senility due to aging.

As chocolate is a non-necessary goods, unlike rice or eggs, inhabitant in difficult situations, may not priorities chocolate consumption if they are already struggling to find a decent everyday meal.

The consumption of chocolate is probably most influenced by cultural, economic, and personal preferences.

## Sources of Flavonoids

Flavonoids are not the exclusivity of the dark chocolate. They can also be found in fruits and vegetables, red wine, and teas, among other.

## Price - Wealth
If chocolate consumption is mainly dependent on average wealth, than chocolate consumption wouldn't be a "fiable" measure because of its high dependent nature. Wealth sensitive.

## Habits - Culture
Chocolate consumption may also be culture sensitive.

Price and habits might "influence" chocolate consumption but theses factors are not a big deal.

## Quality - Concentration of cocoa polyphenolic
Average consumption per capita need to .. 
High consumption of chocolate with a low polyphenolic concentration, might invalidate study. At the contrary a low consumption of chocolate with a high polyphenolic concentration... 

It's worth considering that chocolate consumed in various forms, ranging from dark chocolate with high cocoa content to milk chocolate with added sugars and fats.

The health benefits associated with chocolate are primarily attributed to dark chocolate with higher cocoa content, as it contains a higher concentration of flavonoids and fewer added ingredients.

# Reproductibility and sampled data
> Obviously, these findings are hypothesis-generating only and will have to be tested in a prospective, randomized trial
> 
> It remains to be determined whether the consumption of chocolate is the underlying mechanism for the observed association with improved cognitive function.

Giving the data used during the study is not available, it's hard to reproduce the same result.

## Only 23 countries out of more than 200
Adapt r statistic, relative to this country sample?

## Chocolate consumption data from a Swiss Food Company

## Retry attempt

![alt](/assets/2023-01-25/chocolate-consumption-per-capita.png)

![alt](/assets/2023-01-25/nobel-country-per-capita.png)

# Dependency metric pearson coefficient
r=0.791, P<0.0001

Missing a confidence interval

# Conclusion
Geir Lundestad, Secretary of the Norwegian Nobel Committee in 2006, said, "The greatest omission in our 106-year history is undoubtedly that Mahatma Gandhi never received the Nobel Peace prize. Gandhi could do without the Nobel Peace prize, [but] whether Nobel committee can do without Gandhi is the question"

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
