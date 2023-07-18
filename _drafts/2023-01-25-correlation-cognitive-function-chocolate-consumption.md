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

The author wanted to test whether chocolate consumption enhance human cognition. Indeed, according to a prior study, rats improved their cognitive performances after the administration of a cocoa polyphenolic.

As no clinical trial was lead, the study was purely observational. The author maked use of open data. The *total number of Nobel laureates per capita* were taken as a "surrogate" to cognitive performance (first awards given in 1901 until 2011, the date of the note's publication, span of 110 years), and chocolate consumption estimation came from three different sources: [chocosuisse](https://www.chocosuisse.ch/fr/), [theobroma-cacao](https://www.theobroma-cacao.de/wissen/wirtschaft/international/konsum) and [caobisco](https://caobisco.eu/), on a timespan of approximately 8 years, from 2004 to 2012.

> "If you can’t change the world with cookies, how can you change the world?”
> 
> Pat Murphy

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
  - [Correlation between average consumption and laureate's consumption](#correlation-between-average-consumption-and-laureates-consumption)
- [Sample data](#sample-data)
  - [Only 23 countries out of more than 200](#only-23-countries-out-of-more-than-200)
  - [Timespan diversity](#timespan-diversity)
  - [Chocolate consumption data](#chocolate-consumption-data)
- [Retry attempt](#retry-attempt)
- [Dependency metric pearson coefficient](#dependency-metric-pearson-coefficient)
- [Underlying mechanism](#underlying-mechanism)
- [An humble improvement](#an-humble-improvement)
  - [Is the number of sample countries enough?](#is-the-number-of-sample-countries-enough)
  - [Grouping](#grouping)
  - [Use cross validation](#use-cross-validation)
  - [Confounding variable](#confounding-variable)
  - [Spearman coefficient](#spearman-coefficient)
- [Conclusion](#conclusion)
- [Sources](#sources)

In the following section, I make a brief description of the three pages note of interest, published in the New England Journal of Medecine in 2012.

# Description of the note

**The hypothesis**: chocolate consumption improves cognitive function

**The data:**
* Per capita Nobel laureates
* Per capita yearly chocolate consumption of 23 countries

**The results:**
* Pearson correlation factor $$r = 0.791$$
* Confidence $$\text{p-value} < 0.0001$$
  * Null hypothesis: the correlation is equal to 0
  * Alternate hypothesis: the correlation is not equal to 0 

The correlation result of $$r = 0.791$$ tend to go in the direction of a linear correlation between the chocolate consumption and the number of laureates per capita.

We can already comment that the small number of countries of the dataset could be an issue. Two questions can be raised:
* Is the set of 23 countries a good representation of the diversity of the 195 recognized countries?
* Are there enough countries in the chosen sample?

The p-value is really low $$ p < 0.0001 $$, which is extremely significant, we are pretty sure that we can reject the null hypothesis, that the correlation is not equal to zero. In other word, it is highly unlikely that the observed correlation is only due to chance.

However, on a such little set of countries, it could be valuable to get the confidence interval of the correlation coefficient $$r$$. By the way, do you have any use cases when there aren't good reasons to get a confidence interval?

> Have confidence in intervals!

Defining the confidence interval at 95%, as the following probability:

$$
\text{P}( r \in [r_1, r_2] ) \ge 95% 
$$

In the case of the Pearson's coefficient, we define the parameter $$z$$ the Fisher transformation:

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

Hereafter are the numerical calculation in python.

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

The confidence interval doesn't cross the zero frontier, which is somewhat reassuring against the point of the author, that the chocolate does have an effect on human cognition functions.

# Cognitive function and Nobel Price per capita
According to the dictionnary, **cognition** is the "mental action or process of acquiring knowledge and understanding through thought, experience, and the senses". Cognition is multi-dimensional in the sense that it encompases all aspects of intellectual functions. Wikipedia's page on cognition lists multiple examples of such intellectual functions: perception, attention, thought, imagination, intelligence, memory, judgment and evaluation, reasoning, problem-solving and decision-making, comprehension and production of language.

> As the LLM or foundational language models exhibit impressive generative capabilities, as the production of language is a part of the multitudes of cognitive functions, we could surely ask ourselves, is ChatGPT improving our cognitive functions, and therefore augment our likelihood to obtain a nobel prices?

The following diagram by *Jane S. Paulsen* shows some of them grouped by category:

![alt](/assets/2023-01-25/cognitive-function-Jane-s-Paulsen.jpg)

We can ask ChatGPT, what are the factors likely to influence cognitive function of humans.

Cognitive function of a human being may be dependent on diet, physical activity, sleep, stress levels and intellectual stimulation.

All of these mighty possible factors are dependant at different proportion of the wealth of a person, and by extent to his/her access to learning/cultural resources.

It is not clear to what proportion each of these factors influence the cognitive function, but it is clear that the dark chocolate consumption along with the diet are not the only factors that have an impact on the cognitive performance of a person.

Well, as we could imagine, dark chocolate along with the alimentary diet of a person, is not the only factor influencing her cognitive functions. Well then, it's a really common paradigm in medical hypothesis testing. 

Commonly what the bio-statistician do is:
* make a sumup of all the variables having an influence on the system at hand
* finnd a solution/situation where all theses variabl are fixed except the two being tested
* make that variable chhange, and observe the joint evolution of the other

However, even we might have done the listing of the variables influencing the cognitive function, how do we constrain all the other variables (diet, say physical activity, sleep, stress levels ...), are they constants among the chosen sample of countries?

Trying to answer this question, raise a ton of other questions. We begin to feel the complexity of the study, and that a "simple" correlation coefficient is probably not a sufficient statistic to give some light on this.

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

## Correlation between average consumption and laureate's consumption


**Dependence between chocolate laureate consumption and mean consumption**

> The present data are based on country averages, and the specific chocolate intake of individual Nobel laureates of the past and present remains unknown.
> 
> The cumulative dose of chocolate that is needed to sufficiently increase the odds of being asked to travel to Stockholm is uncertain.


# Sample data
> Obviously, these findings are hypothesis-generating only and will have to be tested in a prospective, randomized trial
> 
> It remains to be determined whether the consumption of chocolate is the underlying mechanism for the observed association with improved cognitive function.

Giving the data used during the study is not available, it's hard to reproduce the same result.

## Only 23 countries out of more than 200
Unrepresentative samples: Hypothesis testing results can be limited by the representativeness of the sample, as unrepresentative samples can lead to biased results.

Limited to sample data: Hypothesis testing relies on analyzing sample data to draw conclusions about population parameters. The accuracy of the conclusions is influenced by the representativeness and size of the sample. If the sample is not truly representative of the population or is too small, the results may not be generalizable.

Adapt r statistic, relative to this country sample?

How do we estimate the number of sample needed for being powered.
Calculate the power of the study:

$$
H_0: \rho = 0 \\
H_a: \rho = r \ne 0
$$

$$
n = C_{\beta} { 4 \over \left [ \ln { 1 - r  \over 1 + r } \right ]^2 } + 3
$$

Thus, with 23 sample size, we have almost 99.8% chance of concluding that the correlation coefficient is different than zero when the correlation coefficient is 0.791.

## Timespan diversity

**Time dependent variables**
> This research is evolving, since both the number of Nobel laureates and chocolate consumption are time-dependent variables and change from year to year.

Considering that the chocolate consumption really democratize at the end of the 19th century. As it was before, exclusively for the "elite", apart from the latin americans.. ?


## Chocolate consumption data
Try to find dark chocolate consumption rather than.

# Retry attempt

Problem of reproducibility, the author not making available data used during his testing...

![alt](/assets/2023-01-25/chocolate-consumption-per-capita.png)

![alt](/assets/2023-01-25/nobel-country-per-capita.png)

# Dependency metric pearson coefficient

Only considering linear correlation, what about Spearman coefficient?

# Underlying mechanism
Results of significance tests are based on probabilities and as such cannot be expressed with full certainty. When a test shows that a difference is statistically significant, then it simply suggests that the difference is probably not due to chance.

Test do not explain the reasons as to why does the difference exist, say between the means of the two samples. They simply indicate whether the difference is due to fluctuations of sampling or because of other reasons but the tests do not tell us as to which is/are the other reason(s) causing the difference.

**Correlation VS Causation**
> Of course, a correlation between X and Y does not prove causation but indicates that either X influences Y, Y influences X, or X and Y are influenced by a common underlying mechanism.

On observe au 18ème siècle une remise en question du principe de causalité. La causalité apparait plus comme une caractéristique de la pensée humaine que comme une notion inhérente aux phénomènes naturels.

Auguste Comte est d'ailleurs catégorique à ce propos: pour lui la recherche des causes relève de la seule métaphysique "en considérant comme absoluement inaccessible et vide de sens pour nous nous la recherche de ce qu'on appelle les causes, soit premières, soit finales". L'important est de pouvoir déterminer des variables en amont de la maladie sur lesquelles on est susceptible d'agir dans un but de prévention ou de guérison.

En simplifiant à l'extrême, la méthode expérimental consiste à:
* faire un bilan des variables ayant une actions sur le système étudié
* trouver une situation dans laquelle toutes ces variables sont fixés, sauf deux
* à faire varier l'une de ces deux variables et à observer l'évolution conjointe de l'autre

After fitting a regression model on let's say probability that a given person will suffer a heart attack, given that person's weight, cholesterol, and so on, it's tempting to interpret each variable on its own: reduse weight, cholestorel, ... and your heart attack risk will decrease by 30%.
But that's not what the model says. The model says that people with cholesterol an weight within a certain range have 30% lower risk of heart attack; it doesn't say that if you put an overweight person on a diet and exercise routine, that person will be less likely to have a heart attack. You didn't collect data on that! You didn't intervene and change the weight and cholestoerol levels of volunteers to see what would happen.

There could be a confounding variable here. Perhaps obesity and high cholesterol levels are merely symptoms of some other factor that also causes heart attacks; exercise and statin pills may fix them but perhaps not the heart attacks. The regression model says lower cholesterol means fewer heart attacks, but that's correlation, not causation.

# An humble improvement

## Is the number of sample countries enough?

23/193 ~ 11.9%

If I make an experiment on 10% of the total population, sample size / total population size = 10% !!

Sampling ratio: ratio between the sample and the population size.

**Degree of accuracy desired**: Related to the subject of Power Analysis (which is beyond the scope of this site), this method requires the researcher to consider the acceptable margin of error and the confidence interval for their study.  The online resource from Raosoft and Raven Analytics uses this principle.

**Degree of variability** (homogeneity/heterogeneity) in the population: As the degree of variability in the population increases, so too should the size of the sample increase.  The ability of the researcher to take this into account is dependent upon knowledge of the population parameters.

**Sampling ratio** (sample size to population size): Generally speaking, the smaller the population, the larger the sampling ratio needed.  For populations under 1,000, a minimum ratio of 30 percent (300 individuals) is advisable to ensure representativeness of the sample.  For larger populations, such as a population of 10,000, a comparatively small minimum ratio of 10 percent (1,000) of individuals is required to ensure representativeness of the sample.

Rule of thumb:
> For populations under 1000, you need sampling ratio of 30% to be really accurate.

## Grouping
As the number of variables affecting the cognitive performance are multiple, we might want to group the coutries by socio-economic similarity before making a correlation analysis.

Although the grouping won't flatten the wealth level of the countries among each group. In consequence, the correlation test may still find some link between the two variables of interset.

Moreover, isn't it a very naty trick, that somewhat is like data manipulation.

## Use cross validation
To test how well your model fits the data, use a separate dataset or a procedure such as cross-validation.

## Confounding variable
Watch out for confounding variables that could cause misleading or reversed results, a in Simpson's paradox.

TODO: page 76

Confounding variables: The presence of confounding variables can affect the validity of hypothesis testing results, as they can affect the relationship between the variables being tested.

## Spearman coefficient

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

# Conclusion
Geir Lundestad, Secretary of the Norwegian Nobel Committee in 2006, said, "The greatest omission in our 106-year history is undoubtedly that Mahatma Gandhi never received the Nobel Peace prize. Gandhi could do without the Nobel Peace prize, [but] whether Nobel committee can do without Gandhi is the question"

# Sources
* [Radio France Podcast](https://www.radiofrance.fr/franceinter/podcasts/la-terre-au-carre/la-terre-au-carre-du-mardi-10-janvier-2023-8979117)
* [Chocolate Consumption, Cognitive Function, and Nobel Laureates](https://www.biostat.jhsph.edu/courses/bio621/misc/Chocolate%20consumption%20cognitive%20function%20and%20nobel%20laurates%20(NEJM).pdf)
* [Evolution of National Nobel Prize Shares in the 20th Century (by Juergen Schmidhuber)](https://people.idsia.ch/~juergen/all.html)
* https://people.idsia.ch/~juergen/nobelshare.pdf
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
* [marginal effect](https://medium.com/analytics-vidhya/logistic-regression-using-python-a5044843a504)
* [marginal effects](https://www.statisticshowto.com/marginal-effects/)
* [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* 