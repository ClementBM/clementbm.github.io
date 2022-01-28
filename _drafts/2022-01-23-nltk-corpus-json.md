---
layout: post
title:  "Exploratory data analysis on hacker news stories with NLTK"
excerpt: "How to load a json corpus with NLTK and perform a quick analysis on hacker news top stories"
date:   2022-01-23
categories: [EDA, NLP, tokenizer]
tags: [NLTK, json]
---

![Split of an apple](/assets/2022-01-23/split-apple.jpg)

In this post we'll use NLTK features to perform a quick overview analysis of hacker news top stories dataset.

The whole code for this project is located at [https://github.com/ClementBM/hackernews-eda](https://github.com/ClementBM/hackernews-eda)
 
- [What's NLTK for?](#whats-nltk-for)
- [Dataset](#dataset)
  - [Load data](#load-data)
  - [Save the dataset in a convenient format](#save-the-dataset-in-a-convenient-format)
- [Integrate with NLTK](#integrate-with-nltk)
  - [Custom Tokenizer, inherits from `TokenizerI`](#custom-tokenizer-inherits-from-tokenizeri)
  - [Custom Corpus, inherits from `CorpusReader`](#custom-corpus-inherits-from-corpusreader)
- [Explore with NLTK](#explore-with-nltk)
- [Descriptive metrics](#descriptive-metrics)
  - [Create a simple Corpus Metrics class](#create-a-simple-corpus-metrics-class)
- [WordCloud](#wordcloud)
- [Sources](#sources)

# What's NLTK for?

NLTK is a python library that enables you to play with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.

It's a free, open source and community-driven project.

Natural Language Tookkit (NLTK), give us a nice starting point to perform exploratory data analysis on textual data.

# Dataset
[Hacker News](https://news.ycombinator.com/) (sometimes abbreviated as HN) is a social news website focusing on computer science and entrepreneurship, similar to [slashdot](https://slashdot.org/). It is run by Paul Graham's investment fund and startup incubator, Y Combinator. In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity."

The word hacker in "Hacker News" is used in its original meaning and refers to the hacker culture which consists of people who enjoy tinkering with technology.

Since HackerNews open its [API](https://github.com/HackerNews/API), data is available in near real time.

## Load data
The real time of top 500 stories is available at `https://hacker-news.firebaseio.com/v0/topstories.json`.

Then every story is retreived via the following end point `https://hacker-news.firebaseio.com/v0/item/{item_id}.json`, `{item_id}`.

```python
import json

import requests
from tqdm import tqdm


hn_topstories_url = (
    "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
)
hn_get_item_url = (
    "https://hacker-news.firebaseio.com/v0/item/{item_id}.json?print=pretty"
)

topstory_result = requests.get(hn_topstories_url)
topstory_ids = json.loads(topstory_result.text)

data = list()
for topstory_id in tqdm(topstory_ids):
    result = requests.get(hn_get_item_url.format(item_id=topstory_id))
    data.append(json.loads(result.text))
```

We use the `loads()` method from `json` module. The `s` in load**s**() means string. As requests.get return a string we use this method.

## Save the dataset in a convenient format
Then we save the dataset in disk as zip to save space.

```python
import pandas as pd
from pathlib import Path

TOPSTORIES_PATH = Path(__file__).parent / "hn_topstories.zip"

data_df = pd.json_normalize(data)
data_df.to_pickle(TOPSTORIES_PATH)
```

You can then read it calling `pd.read_pickle(TOPSTORIES_PATH)`

# Integrate with NLTK
The nltk.corpus package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora, which are listed [here](https://www.nltk.org/nltk_data/).

https://www.nltk.org/howto/corpus.html

Each corpus reader class is specialized to handle a specific corpus format. In addition, the nltk.corpus package automatically creates a set of corpus reader instances that can be used to access the corpora in the NLTK data package

https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk

https://gist.github.com/JeremyEnglert/3eda4a123244c37b669472d9e8166ea6

* `̀TokenizerI`
* `CorpusReader`

## Custom Tokenizer, inherits from `TokenizerI`

`nltk.tokenize.api.TokenizerI`

A processing interface for tokenizing a string.
Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).

override `tokenize()` method

> Note: this is NOT "re" you're likely used to. The regex module is an alternative to the standard re module that supports Unicode codepoint properties with the \p{} syntax.
> You may have to "pip install regx"

**Tokenizer specifications**

| Step | Description |
|--|--|
| Replace html entities | |
| Find word according to regex patterns | |
| Filter (remove) words that are punctuation | |

## Custom Corpus, inherits from `CorpusReader`
A base class for "corpus reader" classes, each of which can be
used to read a specific corpus format.  Each individual corpus
reader instance is used to read a specific corpus, consisting of
one or more files under a common root directory.  Each file is
identified by its ``file identifier``, which is the relative path
to the file from the root directory.

A separate subclass is defined for each corpus format.  These
subclasses define one or more methods that provide 'views' on the
corpus contents, such as ``words()`` (for a list of words) and
``parsed_sents()`` (for a list of parsed sentences).  Called with
no arguments, these methods will return the contents of the entire
corpus.  For most corpora, these methods define one or more
selection arguments, such as ``fileids`` or ``categories``, which can
be used to select which portion of the corpus should be returned.

`nltk.corpus.reader.api.CorpusReader`

# Explore with NLTK

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

# Descriptive metrics

| Metric | Formula | Tokenizer sensitive |
|--|--|--|
| Item Count | $$ n $$ | No |
| Item Unique Count | $$ n_{unique} $$ | No |
| Duplicate Item Proportion | $$ n - n_{unique} \over n $$ | No |
| Dictionary Length | $$ d $$ | Yes |
| Lemmatized Dictionary Length | $$d_u$$ | Yes |
| In vocabulary Token Proportion | $$ d_i \over d_u $$ | Yes |
| Out of vocabulary Token Proportion (wordnet vocabulary) <br> Domain specific metric | $$ {d_o \over d_u} = 1 -  { d_i \over  d_u }$$ | Yes |
| Token Count | $$ t $$ | Yes |
| Lexical Diversity | $$ d \over t $$ | Yes |
| Hapaxes Proportion | $$ d_h \over d $$ | Yes |
| Uppercase Items (title or scope) Proportion | $$ n_{upper} \over n_{unique} $$ | No |
| Numerical Token Proportion | $$ d_{numerical} \over d $$ | Yes |
| Average Item Length | $$ \bar{n} $$ | Yes |
| Standard Deviation Item Length | $$ s_{n} $$ | Yes |
| Median Item Length | $$ \tilde{n} $$ | Yes |
| Minimum/Maximum Item Length | $$ min(n), max(n) $$ | Yes |

## Create a simple Corpus Metrics class

```python
"n" for nouns
"v" for verbs
"a" for adjectives
"r" for adverbs
"s" for satellite adjectives
```
# WordCloud

# Sources
* https://igraph.org/python/tutorial/latest/install.html
* https://regex101.com
* https://www.tensorflow.org/text/guide/subwords_tokenizer?hl=en#optional_the_algorithm
* https://www.nltk.org/book_1ed