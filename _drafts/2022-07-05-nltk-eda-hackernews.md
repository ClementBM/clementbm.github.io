---
layout: post
title:  "Exploratory data analysis on hacker news stories"
excerpt: "How to perform perform a quick analysis on hacker news top stories with NLTK"
date:   2022-07-05
categories: [EDA, NLP, tokenizer]
tags: [NLTK, json]
---

![Original fruit](/assets/2022-07-05/pexels-irina-kaminskaya-12633634.jpg)


## Concordance
```python
corpus_metric.story_text.concordance("language")
```

## Frequency distribution
```python
corpus_metric.frequency_distribution.most_common(20)
corpus_metric.frequency_distribution.plot(20, cumulative=True)
```

![alt](/assets/2022-07-05/frequency-distribution.png)

## Lexical dispersion plot

```python
corpus_metric.story_text.dispersion_plot(
    [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Tesla",
    ]
)
```

![alt](/assets/2022-07-05/lexical-dispersion-plot.png)

## Word Cloud

![alt](/assets/2022-07-05/word-cloud.png)

## Recurrent pattern

```python
topstories[topstories["title"].str.contains("Ask HN")]["title"]
```


# Sources
* 