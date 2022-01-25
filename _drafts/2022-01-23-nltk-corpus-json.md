---
layout: post
title:  "How to load a corpus with nltk from json"
excerpt: "EDA on hacker news top stories with nltk"
date:   2022-01-23
categories: [EDA, NLP, ]
tags: [nltk, json]
---

# 
![Split of an apple](split-apple.jpg)

In this post we'll use nltk features to perform a quick overview analysis of hacker news top stories dataset.
Natural Language Took Kit (NLTK), give us a nice starting point to perform exploratory data analysis on textual data.


> Note: this is NOT "re" you're likely used to. The regex module is an alternative to the standard re module that supports Unicode codepoint properties with the \p{} syntax.
> You may have to "pip install regx"

# Dataset

https://github.com/HackerNews/API

## Load data

## Save the dataset in a convenient format

# Integrate with nltk
https://www.nltk.org/howto/corpus.html


https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk

https://gist.github.com/JeremyEnglert/3eda4a123244c37b669472d9e8166ea6

Le module json est livré avec une méthode appelée loads(), le s dans loads() signifie string. Puisque nous voulons convertir des données de chaîne en JSON, nous utiliserons cette méthode

# Custom Tokenizer, inherits from `TokenizerI`

`nltk.tokenize.api.TokenizerI`

# Custom Corpus, inherits from `CorpusReader`

`nltk.corpus.reader.api.CorpusReader`

# Explore with nltk

```python
corpus_metric.frequency_distribution.most_common(20)

corpus_metric.story_text.collocations()
corpus_metric.story_text.collocations(window_size=3)  # does not work ?

corpus_metric.story_text.concordance("language")

corpus_metric.story_text.dispersion_plot(
    [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Tesla",
    ]
)

corpus_metric.frequency_distribution.plot(20, cumulative=True)
```
# Create a simple Corpus Metrics class


# Sources
* https://igraph.org/python/tutorial/latest/install.html
* https://regex101.com
* https://www.tensorflow.org/text/guide/subwords_tokenizer?hl=en#optional_the_algorithm