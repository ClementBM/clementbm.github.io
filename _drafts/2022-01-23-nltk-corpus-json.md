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
- [HackerNews Dataset](#hackernews-dataset)
  - [Load the data](#load-the-data)
  - [Save the dataset in a convenient format](#save-the-dataset-in-a-convenient-format)
- [Integrate with NLTK](#integrate-with-nltk)
  - [Define a custom Tokenizer, inherits from `TokenizerI`](#define-a-custom-tokenizer-inherits-from-tokenizeri)
  - [Define a custom CorpusReader, inherits from `CorpusReader`](#define-a-custom-corpusreader-inherits-from-corpusreader)
    - [Data Access Methods](#data-access-methods)
    - [Corpus Views](#corpus-views)
- [Explore with NLTK](#explore-with-nltk)
  - [Concordance](#concordance)
  - [Frequency distribution](#frequency-distribution)
  - [Lexical dispersion plot](#lexical-dispersion-plot)
  - [Word Cloud](#word-cloud)
  - [Recurrent pattern](#recurrent-pattern)
  - [Collocations with Dunning likelihood method](#collocations-with-dunning-likelihood-method)
- [Descriptive metrics](#descriptive-metrics)
  - [Create a simple Corpus Metrics class](#create-a-simple-corpus-metrics-class)
- [WordCloud](#wordcloud)
- [Sources](#sources)

# What's NLTK for?

NLTK is a python library that enables you to play with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.

It's a free, open source and community-driven project.

Natural Language Toolkit (NLTK), give us more than a goog starting point to perform exploratory data analysis on textual data. NLTK is one the reference when dealing with natural language processing in python. Other well known libraries are `spaCy`, `Standford Core NLP`, `pattern` and `Textblob`.

If you don't care about the technical part, and are just interested by the analytical one, feel free to jump right to [Explore with NLTK](#explore-with-nltk).

# HackerNews Dataset
[Hacker News](https://news.ycombinator.com/) (sometimes abbreviated as HN) is a social news website focusing on computer science and entrepreneurship, similar to [slashdot](https://slashdot.org/). In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity."

The word hacker in "Hacker News" is used in its original meaning and refers to the hacker culture which consists of people who enjoy tinkering with technology.

Since HackerNews open its [API](https://github.com/HackerNews/API), data is available in near real time.

## Load the data
The real time of top 500 stories is available at `https://hacker-news.firebaseio.com/v0/topstories.json`.

Every story is retreived via the following end point `https://hacker-news.firebaseio.com/v0/item/{item_id}.json`, `{item_id}` being the `id` of the story in this case.

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

We use the `loads()` method from `json` module. The `s` in load**s**() means string. As `requests.get` return a string we use this method.

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

`NLTK` has a reader class that takes an `jsonl` file as input. We take inspiration from this class and save the base corpus file with the same format.

A common json like format used to save corpus, is as json line. HuggingFace also provides readers that takes json line files.

> Link to json line website specification

> Python library to save as jsonl file

# Integrate with NLTK
The `nltk.corpus` package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora, which are listed [here](https://www.nltk.org/nltk_data/).

Each corpus reader class is specialized to handle a specific corpus format. In addition, the nltk.corpus package automatically creates a set of corpus reader instances that can be used to access the corpora in the NLTK data package distribution.

Unlike most corpora, which consist of a set of files, we'll show here how to build a CorpusReader class that handles corpus in a json format.

You would need to create your own corpus reader if you want:

* To access a corpus that is not included in the NLTK data distribution.
* To access a full copy of a corpus for which the NLTK data distribution only provides a sample.
* To access a corpus using a customized corpus reader (e.g., with a customized tokenizer).

The base class `CorpusReader` only defines a few general-purpose methods for listing and accessing the files that make up a corpus. Corpora vary widely in the types of content they include.
It is up to the subclasses to define data access methods that provide access to the information in the corpus. However, corpus reader subclasses should be consistent in their definitions of these data access methods wherever possible.

All the details about the nltk corpus is available at [this link](https://www.nltk.org/howto/corpus.html).

In the following sections we see how to subclass `̀TokenizerI` and `CorpusReader`.

## Define a custom Tokenizer, inherits from `TokenizerI`

NLTK has an interface for tokenizing a string `nltk.tokenize.api.TokenizerI`.

Subclasses must define `tokenize()` or `tokenize_sents()` (or both).

**Tokenizer specifications**

| Step | Description |
|--|--|
| Replace html entities | Remove html entities like &gt; or &nbsp; and convert them to their corresponding unicode character. |
| Find word according to regex patterns | |
| Filter (remove) words that are punctuation | remove punctuation from `string.punctuation` **'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{\|}~'** and **——–’‘“”×** |

The NTLK api define a processing interface for tokenizing a string.
```python
class TokenizerI(ABC):    
    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of *s*.

        :rtype: List[str]
        """
```

A subclass `StoryTokenizer` subclassing `TokenizerI`
```python
class StoryTokenizer(TokenizerI):
    def tokenize(self, text: str) -> typing.List[str]:
        """Tokenize the input text.

        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; joining this list returns\
        the original string if `preserve_case=False`.
        """
```

> Note: this is NOT "re" you're likely used to. The regex module is an alternative to the standard re module that supports Unicode codepoint properties with the \p{} syntax.
> You may have to "pip install regx"

## Define a custom CorpusReader, inherits from `CorpusReader`
NTLK api defines `nltk.corpus.reader.api.CorpusReader`, a base class for "corpus reader" classes. Each corpus reader instance can be used to read a specific corpus format consisting of one or more files under a common root directory. Each file is identified by its `file identifier`, which is the relative path to the file from the root directory.

These subclasses define one or more methods that provide 'views' on the corpus contents, such as `words()` (for a list of words) and `parsed_sents()` (for a list of parsed sentences). Called with no arguments, these methods will return the contents of the entire corpus. For most corpora, these methods define one or more selection arguments, such as `fileids` or `categories`, which can be used to select which portion of the corpus should be returned.

In general CorpusReader gives us way to access properties of the corpus such as
* raw text
* a list of words
* a list of sentences
* a list of paragraphs

NLTK data package also includes various type of corpora such as lexicons and word lists, as well as tagged corpora, in case, the Corpus object gives a set of additional methods is text data annotated such as:
* a list of annotated words, in the form of a list of word and tag couple (as a tuple)
* a list of annotated sentences

### Data Access Methods

Individual corpus reader subclasses typically extend this basic set of file-access methods with one or more data access methods, which provide easy access to the data contained in the corpus. The signatures for data access methods often have the basic form:

```python
corpus_reader.some_data_access(fileids=None, ...options...)
```

Some of the common data access methods, and their return types, are:

| Method | Return type |
|---|---|
| I{corpus}.words() | list of str |
| I{corpus}.sents() | list of (list of str) |
| I{corpus}.paras() | list of (list of (list of str)) |
| I{corpus}.chunked_sents() | list of (Tree w/ (str,str) leaves) |
| I{corpus}.raw() | str (unprocessed corpus contents) |

### Corpus Views

```python
class StoryCorpusReader(CorpusReader):
    CorpusView = StreamBackedCorpusView
```

An important feature of NLTK’s corpus readers is that many of them access the underlying data files using “corpus views.” A corpus view is an object that acts like a simple data structure (such as a list), but does not store the data elements in memory; instead, data elements are read from the underlying data files on an as-needed basis.

By only loading items from the file on an as-needed basis, corpus views maintain both memory efficiency and responsiveness

The most common corpus view is the `StreamBackedCorpusView`, which acts as a read-only list of tokens. 

> When writing a corpus reader for a corpus that is never expected to be very large, it can sometimes be appropriate to read the files directly, rather than using a corpus view.

The heart of a `StreamBackedCorpusView` is its block reader function, which reads zero or more tokens from a stream, and returns them as a list. A very simple example of a block reader is:

```python
def simple_block_reader(stream):
    return stream.readline().split()
```

You can create your own corpus view in one of two ways:

1. Call the `StreamBackedCorpusView` constructor, and provide your block reader function via the block_reader argument.
2. Subclass `StreamBackedCorpusView`, and override the `read_block()` method.

The first option is usually easier, but the second option can allow you to write a single `read_block` method whose behavior can be customized by different parameters to the subclass’s constructor. For an example of this design pattern, see the `TaggedCorpusView` class, which is used by `TaggedCorpusView`.

Corpus views have also the following properties: Concatenation, Slicing, Multiple Iterators


The most similar is the [twitter dataset](https://www.nltk.org/howto/corpus.html#twitter-samples), Tweets are stored as a line-separated JSON.

# Explore with NLTK

## Concordance
```python
corpus_metric.story_text.concordance("language")
```

## Frequency distribution
```python
corpus_metric.frequency_distribution.most_common(20)
corpus_metric.frequency_distribution.plot(20, cumulative=True)
```

![alt](/assets/2022-01-23/frequency-distribution.png)

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

![alt](/assets/2022-01-23/lexical-dispersion-plot.png)

## Word Cloud

![alt](/assets/2022-01-23/word-cloud.png)

## Recurrent pattern

```python
topstories[topstories["title"].str.contains("Ask HN")]["title"]
```

## Collocations with Dunning likelihood method

```python
corpus_metric.story_text.collocations() # Dunning likelihood collocation
# three word window: corpus_metric.story_text.collocations(window_size=3)
```

open source; command line; Dark Souls; Nhat Hanh; Points Guy; Suite
legacy; Thich Nhat; largest chip; American Airlines; black holes; SICP
JavaScript; modern language; Google Analytics; source Ask

```python
topstories[topstories["title"].str.contains(" source")]["title"]
```

# Descriptive metrics


We define a corpus $$\mathcal{C} = {d_i} $$ as a list of document. The number of document can be seen as the cardinality of the corpus $$\vert \mathcal{C} \vert = N$$.

In this case, each document has only one text field, the title.

Each document has a set of text attribute/field. Each field is indexed with j from 1 to M, such $$a_{ij}$$ is the j$$^{th}$$ field of the i$$^{th}$$ document.

We define a tokenizer function $$ f_{tokenizer} $$ as $$ f_{tokenizer} : a_{ij} \rightarrow [t_1, ..., t_k, ... , t_L] $$, with $${t_k}$$ the list of tokens corresponding to the $$ a_{ij} $$ field.

We define $$ \mathcal{T} = t_{ijk} $$ the token k for the field j of the ith document.
We can also note $$ \mathcal{T}_j = t_{ik} $$ the list of token for a given field j.

We define the dictionary $$ \mathcal{D} = { t_i } $$ given a tokenizer function $$ f_{tokenizer} $$, as the list of unique tokens generated by a tokenizer on a given corpus.

| Metric | Formula | Tokenizer sensitive |
|--|--|--|
| Item Count | $$ \vert \mathcal{C} \vert $$ | No |
| Item Unique Count | $$ \vert \mathcal{C}_{unique} \vert $$ | No |
| Duplicate Item Proportion | $$ \vert \mathcal{C} \vert - \vert \mathcal{C}_{unique} \vert \over \vert \mathcal{C} \vert $$ | No |
| Dictionary Length | $$ \vert \mathcal{D} \vert $$ | Yes |
| Lemmatized Dictionary Length | $$ \vert \mathcal{D}_{lemme} \vert $$ | Yes |
| In vocabulary Token Proportion | $$ \vert \mathcal{D}_{lemme} \cap \mathcal{D}_{NLTK} \vert \over \vert \mathcal{D}_{lemme} \vert $$ | Yes |
| Out of vocabulary Token Proportion (wordnet vocabulary) <br> Domain specific metric | $$ \vert \mathcal{D}_{lemme} \vert - \vert \mathcal{D}_{lemme} \cap \mathcal{D}_{NLTK} \vert \over \vert \mathcal{D}_{lemme} \vert $$ | Yes |
| Token Count | $$ \vert \mathcal{T} \vert $$ | Yes |
| Lexical Diversity | $$ \vert \mathcal{D} \vert \over \vert \mathcal{T} \vert $$ | Yes |
| Hapaxes Proportion | $$ \vert \mathcal{D}_{hapax} \vert \over \vert \mathcal{D} \vert $$ | Yes |
| Uppercase Items Proportion | $$ n_{upper} \over n_{unique} $$ | No |
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
* https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk

* See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of Statistical Natural Language Processing. MIT press, p. 162 https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22