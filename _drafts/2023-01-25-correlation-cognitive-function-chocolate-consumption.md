---
layout: post
title:  "Analyzing a study on the correlation between chocolate consumption and cognitive function"
excerpt: ""
date:   2023-01-25
categories: [study]
tags: [chocolate, correlation]
---

![Chocolate consumption](/assets/2023-01-25/chocolate-diva.jpg){: width="50%" style="margin-left: auto;margin-right: auto;display: block;"  }

When I was studying at the university, I once made an english report on how to rethink the strategy of a sinking company. By way of this use case, we had to find new game planes for selling, innovate to find new customers and fulfill new customer needs.

The whole exercise was certainly boring, but as I was lurking on the internet in lack of inspiration, I found this note on [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf), by *Franz H. Messerli* published in the New England Journal of Medecine in 2012. In this paper, the author wanted to test whether chocolate consumption enhance human cognition. Indeed, according to a prior study, rats improved their cognitive performances after the administration of a cocoa polyphenolic.

As a last word of the report, I adviced the fake managing directors to eat more chocolate as to improve their cognitive functions and help them finding disruptive ideas to enhance their business. Sounds like a good idea, isn't it? (well... maybe not).

**Table of contents**
- [Description of the note](#description-of-the-note)
- [A little improvement of the statistical evidence](#a-little-improvement-of-the-statistical-evidence)
- [Research Design](#research-design)
  - [Dark chocolate is not the only factor at play](#dark-chocolate-is-not-the-only-factor-at-play)
  - [Is Nobel laureates per capita a fair indicator of the cognitive performance of country?](#is-nobel-laureates-per-capita-a-fair-indicator-of-the-cognitive-performance-of-country)
  - [Two indicators (cocoa, cognition) of a confounding variable (country's wealth)?](#two-indicators-cocoa-cognition-of-a-confounding-variable-countrys-wealth)
- [Execution](#execution)
  - [A very poor sample of wealthy countries](#a-very-poor-sample-of-wealthy-countries)
  - [Dependency metric pearson coefficient](#dependency-metric-pearson-coefficient)
- [Retry attempt and conclusion](#retry-attempt-and-conclusion)
- [Resources](#resources)
  - [Cost of foods and chocolate](#cost-of-foods-and-chocolate)
  - [Nobel Prices](#nobel-prices)
  - [Cognitive function](#cognitive-function)
  - [Statistics](#statistics)
  - [Data Sources](#data-sources)
  - [Chocolate international trades](#chocolate-international-trades)


# Description of the note
The study was purely observational as no clinical trial was lead. The author maked use of open data. 
The *total number of Nobel laureates per capita* were taken as a "surrogate" to cognitive performance. Chocolate consumption estimation came from three different data sources published by chocolate players: chocolate manufacturers, federations and associations.

![Chocolate consumption](/assets/2023-01-25/chocolate-consumption-by-nobel-laureate.png){: width="100%"  }

**The hypothesis**: chocolate consumption improves cognitive function

**The data:**
* Per capita Nobel laureates on the last century
* Per capita yearly chocolate consumption of 23 countries

**The results:**
* Pearson correlation factor $$r = 0.791$$
* Confidence $$\text{p-value} < 0.0001$$
  * Null hypothesis: the correlation is equal to 0
  * Alternate hypothesis: the correlation is not equal to 0 

A correlation coefficient of $$r = 0.791$$ indicates a positive and linear correlation between the chocolate consumption and the number of laureates per capita.

The p-value is really low $$ p < 0.0001 $$, which is extremely significant, we are pretty sure that we can reject the null hypothesis, that the correlation is not equal to zero. In other word, it is highly unlikely that the observed correlation is only due to chance.

However proving that the correlation is significantly different than zero doesn't give that much information about the alternative hypothesis. Why is there a correlation? Which are the reasons causing it? And what is the strength of the correlation?. Actually, we could have a similar p-value on a tiny correlation ratio although it would need a larger sample size. (TODO: calculate the power for p=0.0001, and effect of 0.0 to 0.1?)
To have a better grasp of the correlation at stake, let's calculate the confidence interval.

# A little improvement of the statistical evidence
Defining the confidence interval at 95%, as the following probability:

$$
\text{P}( r \in [r_1, r_2] ) \ge 95% 
$$

In the case of the Pearson's coefficient, we define the parameter $$z$$, that is the Fisher transformation:

$$
z(r) = { 1 \over 2 } \ln \left( { 1 + r \over 1 - r } \right) = \text{arctanh} (r)
$$

$$z$$ approximately follows a normal distribution with
* $$mean(z) = \text{arctanh} (\rho) $$ and
* standard error $$s_z = { 1 \over \sqrt{n -3}}$$

As $$z$$ has a normal distribution, the confidence interval of $$z$$ for 95% is:

$$
[ z_1 = z - 2 s_z; z_2 = z + 2 s_z]
$$

Then the confidence interval of $$r$$ is:

$$
r_i = { \exp{(2 z_i)} - 1 \over \exp{(2 z_i)} + 1 }\text{, with } i \in [1, 2]
$$

Hereafter are the numerical calculation in python (skipping imports for readability):

```python
r = 0.791
z = math.log((1 + r) / (1 - r)) / 2
# z = 1.074

n = 23
s_z = 1 / math.sqrt(n - 3)
# s_z = 0.226
```

```python
z_1, z_2 = z - 2 * s_z, z + 2 * s_z
# z_1, z_2 = 0.741, 1.407
```

```python
def z_to_r(z):
  return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)

r_1, r_2 = z_to_r(z_1), z_to_r(z_2)
# r_1, r_2 = 0.630, 0.887
```

Finally the confidence interval is:

$$
\text{P}( r \in [0.630, 0.887] ) \ge 95% 
$$

The confidence interval doesn't cross the zero frontier, which is somewhat reassuring against the point of the author, that the chocolate does have a positive effect on human cognition functions. By the way, the author of the study didn't mention the confidence interval of the correlation coefficient in this note.

If the confidence interval gives us a little more faith in the author's postulate, there are a tons of other objections that can be made:
* Isn't the chocolate consumption just an indicator of the wealth of a country? (which is also the case with the "Nobel laureates per capita")
* Is the set of countries enough to draw conclusions?
* Is "nobel laureates per capita" a direct indicator of the mean cognitive performance of a country?

# Research Design
## Dark chocolate is not the only factor at play

According to the dictionnary, **cognition** is the "mental action or process of acquiring knowledge and understanding through thought, experience, and the senses". Cognition is multi-dimensional in the sense that it encompases all aspects of intellectual functions. Wikipedia's page on cognition lists multiple examples of such intellectual functions: perception, attention, thought, imagination, intelligence, memory, judgment and evaluation, reasoning, problem-solving and decision-making, comprehension and production of language.

The following diagram by *Jane S. Paulsen* shows some of them grouped by category:

![alt](/assets/2023-01-25/cognitive-function-Jane-s-Paulsen.jpg)

Cognitive function of a human may be influenced by **diet**, **physical activity**, **sleep**, **stress levels** and **intellectual stimulation.** All of these factors are dependent at different proportion of the **wealth of a person**, and by extent to his/her access to **learning/cultural resources**. The environment in which a person lives inside his/her country, have a significant impact on his/her intellectual/academic achievements. Investment in education, research infrastructure, socioeconomic conditions, and cultural values are predominent.

Who knows in what proportions theses factors influence cognitive functions. Alimentary diet and in a smaller extent the consumption of dark chocolate are only a few among multiples factors that influences cognitive performance of a person.

If the study was experimental, the bio-statistician would have had to design the experiment so that after making a sumup of all the variables having an influence on the system at hand, he/she would:
* find a solution/situation where all these variables are fixed except the two being tested
* make the causing variable change, and observe the joint evolution of the other one

Unfortunately, medical hypothesis testing requires a lot of time and resources. It's natural trying to get insight with already available indicators beforing choosing to give the go or nogo. As a consequence, doing a fair list of the variables influencing the cognitive function before calculating the correlation seems reasonable too. So that the result of the correlation, whatever this value is, can be put in perspective. How the other influencing variables (diet, physical activity, sleep, stress levels) fluctuate in the selected sample of countries and how do they impact the cognitive functions? Who knows...

It's also worth considering that cocoa is consumed in various forms, ranging from dark chocolate with high cocoa content to milk chocolate with added sugars and fats. The health benefits associated with chocolate are primarily attributed to dark chocolate with higher cocoa content, as it contains a higher concentration of flavonoids and fewer added ingredients. High consumption of chocolate with a low polyphenolic concentration, as well as a low consumption with a high polyphenolic concentration migth weakened the goodness of the indicator. Besides, the average consumption is not necessarily consistent with the laureates' consumption. By the way, the Flavonoids are not the exclusivity of the dark chocolate, they can also be found in fruits and vegetables, red wine, and teas, among other.

We begin to feel the complexity of the study, and that a "simple" correlation coefficient is probably not a sufficient statistic to give some light on the author's postulate.

## Is Nobel laureates per capita a fair indicator of the cognitive performance of country?

> Conceivably, however, the total number of Nobel laureates per capita could serve as a surrogate end point reflecting the proportion with superior cognitive function and thereby give us some measure of the overall cognitive function of a given country.
> 
> Note's author


The Nobel price rewards notable contribution of scientists (accross the globe) in the advance of the knowledge. Rewards span accross the fields of chemistry, literature, peace, physics and medicine. It doesn't encompass the field of mathematic since Alfred Nobel, the creator of the Nobel's price, wasn't really in good terms with mathematicians.

Caution shoud be taken specifically on the Nobel Peace Prize. Indeed, according to the Nobel Peace Center in Oslo, this prize is biaised toward the western countries until the 1990's: "From the 1990s onwards, many prizes were awarded for human rights, and the peace prize became truly global". As for the field of mathematic, we could argue that the study is missing the Abel Prize, the Nobel equivalent in the scope of mathematic.

The note presupposed that the number of **Nobel laureates per capita** is concordant with the average cognitive performance of a country. However a short exploratory analysis of the population of nobel price laureates shows that laureates have around sixty years old or more and that only 6% of laureates are female. Not very representative...

Aren't there better indicators of the cognitive performance of a country? Well, there surely is. We could take the PISA worldwide ranking as an indicator. The Programme for International Student Assessment is the OECD's programme which measures 15-year-olds' ability to use their reading, mathematics and science knowledge and skills to meat real-life challenges. 

## Two indicators (cocoa, cognition) of a confounding variable (country's wealth)?

> Of course, a correlation between X and Y does not prove causation but indicates that either X influences Y, Y influences X, or X and Y are influenced by a common underlying mechanism.
>
> Note's author

Nonetheless the good starting point of the study, and because the *consumption of chocolate* and the *Nobel laureates per capita* are both indicators of the wealth of a (occidental) country, trying to prove that chocolate improve cognition on national statistics, comes to proves that rich countries are actually rich.


The presence of confounding variables can affect the validity of hypothesis testing results, as they can affect the relationship between the variables being tested. The cocoa supposedly have a positive effect on the cognitive function. But isn't the country wealth strongly impacting both phenomenons and their indicators? The following diagram illustrates this by showing a conceptual model of the variables:

![Conceptual model of the cocoa consumption effect on cognitive functions](/assets/2023-01-25/cocoa-cognition-conceptual-model.png){: style="margin-left: auto;margin-right: auto;display: block;"}


It's like trying to find a correlation between the garbage indicator and the life expectancy, both are strongly related to the economic state of a country. There is a good chance for them to be correlated. Another good example of the omitted variable bias is the correlation between ice cream consumption and sunburn. There IS a correlation, but does that mean ice cream consumption causes sunburn?

![Conceptual model of the ice cream consumption effect on sunburns](/assets/2023-01-25/icecream-sunburn-conceptual-model.png){: style="margin-left: auto;margin-right: auto;display: block;"}

Even if chocolate intake is strongly correlated with cognitive function, it doesn't mean that an increase in chocolate consumption do enhance cognition. The correlation coeffient says that people eating more chocolate have better chance to have higher cognitive function; it doesn't say that you'll enhance your cognitive fonctions by eating more chocolate. Unlike the other study with Wistar-Unilever rats, the current study didn't intervene on the chocolate consumption of volunteers to see what would happen.

> The present data are based on country averages, and the specific chocolate intake of individual Nobel laureates of the past and present remains unknown.
>
> Note's author

In addition, chocolate is a rather luxury foods, it was known by the Mayas as the "Theobroma cacao" (food of the gods). As a non-necessary goods, unlike rice or eggs, inhabitant struggling to find a decent everyday meal may not prioritize chocolate consumption. Even for wealthy inhabitants the local gastronomy has a huge impact on their consumption. For instance, the Chinese middle class doesn't consume that much of chocolate, unlike red wine, because they simply don't eat a lot of sweet in general. Last fact but not the least, remember which country invent the compass, the gunpowder, the papermaking and printing? Well, China! And without any chocolate! This fact emphasize the great care to put everythings in an historical perspective. Did you see where China lies on the correlation diagram?

Chocolate consumption is probably an epiphenomenon of the wealth of (occidental) countries, which happen to have an inclination toward sweety foods and therefore are likely to appreciate the Maya beverage. At the same time, the wealthiest countries in the world for the last century (1900-2000) also happen to be in a large majority occidental (be carefull with the [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox)!)


# Execution
## A very poor sample of wealthy countries

At the level of the execution, the small number of countries could be an issue.
* Are there enough countries in the chosen sample?
* Is the set of 23 countries a good representation of the diversity of the 195 recognized countries?

> Obviously, these findings are hypothesis-generating only and will have to be tested in a prospective, randomized trial
> 
> It remains to be determined whether the consumption of chocolate is the underlying mechanism for the observed association with improved cognitive function.

Hypothesis testing relies on analyzing sample data to draw conclusions about all the population. The accuracy of the conclusions is influenced by the representativeness and the size of the sample. If the sample is not truly representative of the population or is too small, the results may not be generalizable.

Generally we estimate the number of sample needed for being powered. To calculate the power of the study we can use:

$$
H_0: \rho = 0 \\
H_a: \rho = r \ne 0
$$

$$
n = C_{\beta} { 4 \over \left [ \ln { 1 - r  \over 1 + r } \right ]^2 } + 3
$$

Thus, with 23 sample size, we have almost 99.8% chance of concluding that the correlation coefficient is different than zero when the correlation coefficient is 0.791.

With only 23 countries out of 193


23/193 ~ 11.9%

If I make an experiment on 10% of the total population, sample size / total population size = 10% !!

Sampling ratio: ratio between the sample and the population size.

**Degree of accuracy desired**: Related to the subject of Power Analysis (which is beyond the scope of this site), this method requires the researcher to consider the acceptable margin of error and the confidence interval for their study.  The online resource from Raosoft and Raven Analytics uses this principle.

**Degree of variability** (homogeneity/heterogeneity) in the population: As the degree of variability in the population increases, so too should the size of the sample increase.  The ability of the researcher to take this into account is dependent upon knowledge of the population parameters.

**Sampling ratio** (sample size to population size): Generally speaking, the smaller the population, the larger the sampling ratio needed.  For populations under 1,000, a minimum ratio of 30 percent (300 individuals) is advisable to ensure representativeness of the sample.  For larger populations, such as a population of 10,000, a comparatively small minimum ratio of 10 percent (1,000) of individuals is required to ensure representativeness of the sample.

Rule of thumb:
> For populations under 1000, you need sampling ratio of 30% to be really accurate.


Timespan is heterogenous and diverse.

**Time dependent variables**
> This research is evolving, since both the number of Nobel laureates and chocolate consumption are time-dependent variables and change from year to year.

Considering that the chocolate consumption really democratize at the end of the 19th century. As it was before, exclusively for the "elite", apart from the latin americans.. ?

* first awards given in 1901 until 2011, the date of the note's publication, span of 110 years
* [chocosuisse](https://www.chocosuisse.ch/fr/), [theobroma-cacao](https://www.theobroma-cacao.de/wissen/wirtschaft/international/konsum) and [caobisco](https://caobisco.eu/), on a timespan of approximately 8 years, from 2004 to 2012.

## Dependency metric pearson coefficient

Only considering linear correlation, what about Spearman coefficient?


L'intensité de la liaison linéaire entre deux variables continues peut être mesuré par le coefficient de corrélation linéaire (ou r de Pearson).
La liaison:
* nulle si le coefficient de corrélation est 0 (nuage de points circulaires ou parallèle à un des deux axes correspondant aux variables)
* parfaite si le coefficient de corrélation est de + ou - 1 (nuage de points rectiligne)
* forte si le coefficient de corrélation est supérieur en valeur absolue à 0.8 (nuage elliptique allongé)

Le coefficient de corrélation linéaire est positif lorque les deux variables évoluent dans le mêe sens: les deux augment  ou diminuent ensemble. Un coefficient de corrélation négatif indique une variation inverse: l'une augmente quand l'autre diminue.

Mais une liaison non linéaire, a fortiori non monotone, n'est pas toujours mesurable par le coefficient de corrélation linéaire de Pearson. C'est ainsi le cas d'une liaison parabolique (de degré 2). De plus en présence de valeurs eceptionnelles, de points aberrants, même une liaison linéaire peut ne pas être détectée par le coefficient de Pearson.
Exemple: d'Anscombe

Tandis que le coefficient de corrélation linéaire de Pearson n'est utilisable qu'avec des variables continues, le coefficient de corrélation de rangs (ou rho) de Spearman permet de mesurer la liaison entredeux variables qui peuvent être continues, discrète ou ordinales. Même pour des variables continues le rho de Spearman est préférable au r de Pearson quand les variables ont des valeurs extrêmes ou ne suivent pas une loi normale. En outre, le rho de Spearman détecte bien toutes les liaison monotones, même non linéaire.

On a toujours intérêt à comparer les deux coefficient de Pearson et Spearman, le second étant utilisable dans un plus grand nombre de situations:
* si r > rho: on est peut être en présence de valeurs exceptionnelles (?????)
* si r < rho: on est peut être en présence d'une liaison non linéaire

Ce qui fait la robustesse de ce test est que le rho de Spearman est calculé sur les rangs des valeurs des variables, et non sur les valeurs elles-mêmes, ce qui lui permet de s'affranchir de l'hypothèse contraignante de normalité des variables: c'est un test non-paramétrique.
Le test de Spearman est considéré comme imparfait lorsqu'une variables présente de nombreux ex-aequo dans la population.

# Retry attempt and conclusion

Giving the data used during the study is not available, it's hard to reproduce the same result.


Problem of reproducibility, the author not making available data used during his testing...

![alt](/assets/2023-01-25/chocolate-consumption-per-capita.png)

![alt](/assets/2023-01-25/nobel-country-per-capita.png)

Geir Lundestad, Secretary of the Norwegian Nobel Committee in 2006, said, "The greatest omission in our 106-year history is undoubtedly that Mahatma Gandhi never received the Nobel Peace prize. Gandhi could do without the Nobel Peace prize, [but] whether Nobel committee can do without Gandhi is the question"

> As the LLM or foundational language models exhibit impressive generative capabilities, as the production of language is a part of the multitudes of cognitive functions, we could surely ask ourselves, is ChatGPT improving our cognitive functions, and therefore augment our likelihood to obtain a nobel prices?


> "If you can’t change the world with cookies, how can you change the world?”
> 
> Pat Murphy

??? Change for the worse ???

# Resources
* [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf)

## Cost of foods and chocolate
* [Worldwide Food Prices](https://www.numbeo.com/food-prices/)
* [Daily cost of a calorie sufficient diet, 2017](https://ourworldindata.org/grapher/cost-calorie-sufficient-diet?country=USA~BHR~GBR~IND~CHN~BRA~ZAF~FRA~TCD)
* [The steep price we pay for cheap chocolate](https://ideas.ted.com/the-steep-price-we-pay-for-cheap-chocolate/)
* [La France, 7e au rang des consommateurs de chocolat](https://www.agro-media.fr/dossier/france-7e-rang-consommateurs-de-chocolat-24500.html)

## Nobel Prices
* [Evolution of National Nobel Prize Shares in the 20th Century (by Juergen Schmidhuber)](https://people.idsia.ch/~juergen/all.html)(https://people.idsia.ch/~juergen/nobelshare.pdf)
* [Des prix Nobel toujours très masculins et de plus en plus âgés](https://www.liberation.fr/sciences/des-prix-nobel-toujours-tres-masculins-et-de-plus-en-plus-ages-20211011_YKTNPXRUZVB7FBWZCANHTWHAOQ/)

## Cognitive function
* [Defining the 6 Key Domains of Cognitive Function](https://altoida.com/blog/defining-the-6-key-domains-of-cognitive-function/)

## Statistics
* [La vraisemblance](https://www.youtube.com/watch?v=P-AHaAP8fIk)
* [Initiation à la statistique bayésienne](https://www.youtube.com/watch?v=5hN_plbtPjw)
* [Marginal effect with logistic regression](https://medium.com/analytics-vidhya/logistic-regression-using-python-a5044843a504)
* [Marginal Effects: Definition](https://www.statisticshowto.com/marginal-effects/)
* [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* [What is a confounding variable?](https://www.scribbr.com/methodology/confounding-variables/)

## Data Sources
* [Nobel Prize - Laureates](https://public.opendatasoft.com/explore/dataset/nobel-prize-laureates/table/?flg=fr&disjunctive.category)
* [List of countries by Nobel laureates per capita](https://en.wikipedia.org/wiki/List_of_countries_by_Nobel_laureates_per_capita)
* [Food and Agriculture Organization of the UN](https://www.fao.org/faostat/en/#data)
* [Cocoa ean consumtion per person, 2008](https://ourworldindata.org/grapher/chocolate-consumption-per-person?time=2008&country=AUS~DMA~NAM)
* [31 Current Chocolate Statistics (Market Data 2023)](https://damecacao.com/chocolate-statistics)

## Chocolate international trades
* [Le chocolat, une catastrophe écologique et sociale ? (Podcast)](https://www.radiofrance.fr/franceinter/podcasts/la-terre-au-carre/la-terre-au-carre-du-mardi-10-janvier-2023-8979117)
* [Les circuits du chocolat (Podcast)](https://www.radiofrance.fr/franceculture/podcasts/affaires-etrangeres/les-circuits-du-chocolat-2187978)
* [Chocolat 100% droit (Book)](https://www.goodreads.com/book/show/170291254-chocolat-100-droit)
