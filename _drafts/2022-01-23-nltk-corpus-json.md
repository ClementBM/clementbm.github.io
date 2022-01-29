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

# HackerNews Dataset
[Hacker News](https://news.ycombinator.com/) (sometimes abbreviated as HN) is a social news website focusing on computer science and entrepreneurship, similar to [slashdot](https://slashdot.org/). In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity."

The word hacker in "Hacker News" is used in its original meaning and refers to the hacker culture which consists of people who enjoy tinkering with technology.

Since HackerNews open its [API](https://github.com/HackerNews/API), data is available in near real time.

## Load the data
The real time of top 500 stories is available at `https://hacker-news.firebaseio.com/v0/topstories.json`.

Then every story is retreived via the following end point `https://hacker-news.firebaseio.com/v0/item/{item_id}.json`, `{item_id}` being the `id` of the story in this case.

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

# Integrate with NLTK
The `nltk.corpus` package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora, which are listed [here](https://www.nltk.org/nltk_data/).

The documentation at https://www.nltk.org/howto/corpus.html gives you all the details about each peculiar CorporaReader inherited instances and specificities in its implementation.

Each corpus reader class is specialized to handle a specific corpus format. In addition, the nltk.corpus package automatically creates a set of corpus reader instances that can be used to access the corpora in the NLTK data package

https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk

https://gist.github.com/JeremyEnglert/3eda4a123244c37b669472d9e8166ea6

Unlike most corpora, which consist of a set of files, we'll show here how to build a CorpusReader class that handles corpus in a json format.

* `̀TokenizerI`
* `CorpusReader`

Although the nltk.corpus module automatically creates corpus reader instances for the corpora in the NLTK data distribution, you may sometimes need to create your own corpus reader. In particular, you would need to create your own corpus reader if you want…

* To access a corpus that is not included in the NLTK data distribution.
* To access a full copy of a corpus for which the NLTK data distribution only provides a sample.
* To access a corpus using a customized corpus reader (e.g., with a customized tokenizer).

Corpus Types

Corpora vary widely in the types of content they include. This is reflected in the fact that the base class CorpusReader only defines a few general-purpose methods for listing and accessing the files that make up a corpus. It is up to the subclasses to define data access methods that provide access to the information in the corpus. However, corpus reader subclasses should be consistent in their definitions of these data access methods wherever possible.

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

NLTK data package also includes a number of lexicons and word lists. 

In general CorpusReader gives us way to access properties of the corpus such as
* raw text
* a list of words
* a list of sentences
* a list of paragraphs

There are also tagged corpora, in case, the Corpus object gives a set of additional methods is text data annotated such as:
* a list of annotated words, in the form of a list of word and tag couple (as a tuple)
* a list of annotated sentences

The most similar is the [twitter dataset](https://www.nltk.org/howto/corpus.html#twitter-samples), Tweets are stored as a line-separated JSON.

Every corpus is assumed to consist of one or more files, all located in a common root directory (or in subdirectories of that root directory)

**Data Access Methods**

Individual corpus reader subclasses typically extend this basic set of file-access methods with one or more data access methods, which provide easy access to the data contained in the corpus. The signatures for data access methods often have the basic form:

```python
corpus_reader.some_data access(fileids=None, ...options...)
```

Some of the common data access methods, and their return types, are:

* I{corpus}.words(): list of str
* I{corpus}.sents(): list of (list of str)
* I{corpus}.paras(): list of (list of (list of str))
* I{corpus}.tagged_words(): list of (str,str) tuple, only supported by corpora that include part-of-speech annotations
* I{corpus}.tagged_sents(): list of (list of (str,str)), only supported by corpora that include part-of-speech annotations
* I{corpus}.tagged_paras(): list of (list of (list of (str,str))), only supported by corpora that include part-of-speech annotations
* I{corpus}.chunked_sents(): list of (Tree w/ (str,str) leaves)
* I{corpus}.parsed_sents(): list of (Tree with str leaves)
* I{corpus}.parsed_paras(): list of (list of (Tree with str leaves))
* I{corpus}.xml(): A single xml ElementTree
* I{corpus}.raw(): str (unprocessed corpus contents)

```python
class StoryCorpusReader(CorpusReader):
    CorpusView = StreamBackedCorpusView
```

**Corpus Views**

An important feature of NLTK’s corpus readers is that many of them access the underlying data files using “corpus views.” A corpus view is an object that acts like a simple data structure (such as a list), but does not store the data elements in memory; instead, data elements are read from the underlying data files on an as-needed basis.

By only loading items from the file on an as-needed basis, corpus views maintain both memory efficiency and responsiveness

The most common corpus view is the `StreamBackedCorpusView`, which acts as a read-only list of tokens. 

> When writing a corpus reader for a corpus that is never expected to be very large, it can sometimes be appropriate to read the files directly, rather than using a corpus view.

The heart of a StreamBackedCorpusView is its block reader function, which reads zero or more tokens from a stream, and returns them as a list. A very simple example of a block reader is:

```python
def simple_block_reader(stream):
    return stream.readline().split()
```

You can create your own corpus view in one of two ways:

1. Call the StreamBackedCorpusView constructor, and provide your block reader function via the block_reader argument.
2. Subclass StreamBackedCorpusView, and override the read_block() method.

The first option is usually easier, but the second option can allow you to write a single read_block method whose behavior can be customized by different parameters to the subclass’s constructor. For an example of this design pattern, see the TaggedCorpusView class, which is used by TaggedCorpusView.

Corpus views have also the following properties: Concatenation, Slicing, Multiple Iterators

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