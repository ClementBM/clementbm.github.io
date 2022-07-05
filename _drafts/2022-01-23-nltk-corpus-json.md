---
layout: post
title:  "Extend NLTK CorpusReader"
excerpt: "How to load a json corpus with NLTK on hacker news top stories"
date:   2022-01-23
categories: [EDA, NLP, tokenizer]
tags: [NLTK, jsonl]
---

![Split of an apple](/assets/2022-01-23/split-apple.jpg)

In this post we'll extend NLTK CorpusReader class to load hacker news top stories dataset.

- [What's NLTK for?](#whats-nltk-for)
- [HackerNews Dataset](#hackernews-dataset)
  - [Load the data](#load-the-data)
  - [Save the dataset in a convenient format](#save-the-dataset-in-a-convenient-format)
- [Integrate with NLTK](#integrate-with-nltk)
  - [Define a custom Tokenizer, inherits from `TokenizerI`](#define-a-custom-tokenizer-inherits-from-tokenizeri)
  - [Define a custom CorpusReader, inherits from `CorpusReader`](#define-a-custom-corpusreader-inherits-from-corpusreader)
    - [Data Access Methods](#data-access-methods)
    - [Corpus Views](#corpus-views)
- [Sources](#sources)

The whole code for this project is located at [https://github.com/ClementBM/hackernews-eda](https://github.com/ClementBM/hackernews-eda)

# What's NLTK for?

NLTK is a reference python library that enables you to play with human language data. It provides a suite of text processing tools for classification, tokenization, stemming, tagging, parsing, and semantic reasoning. It also gives easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, Penn Treebank. As a free, open source and community-driven project, it's also quite easy to extend.

Natural Language Toolkit (NLTK), give us more than a good starting point to perform exploratory data analysis on textual data. When dealing with natural language processing in python, NLTK is one of the oldest library still in use. Other well known libraries are `spaCy`, `Standford Core NLP`, `pattern` and `Textblob`.

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
Then we save the dataset in disk as `jsonl`. Json line is a common json like format used to save corpus. Various python libraries use this format like HuggingFace, Scrapy and Spark among others.

[Link to json line website specification](https://jsonlines.org/)

```python
import pandas as pd
from pathlib import Path

TOPSTORIES_PATH = Path(__file__).parent / "hn_topstories.zip"

data_df = pd.json_normalize(data)
with open(file_path, "ab") as json_file:
    data_df.apply(
        lambda x: json_file.write(f"{x.to_json()}\n".encode("utf-8")),
        axis=1,
    )
```

`NLTK` has a reader class that takes an `jsonl` file as input. We take inspiration from this class and save the base corpus file with the same format.

# Integrate with NLTK
The `nltk.corpus` package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora, which are listed [here](https://www.nltk.org/nltk_data/).

Each corpus reader class is specialized to handle a specific corpus format. Unlike most corpora, which consist of a set of files, we'll show here how to build a CorpusReader class that handles corpus in a jsonl format, taking inspiration from the Twitter corpus.

Along the way, we'll also be able to access a corpus using a customized corpus reader (e.g., with a customized tokenizer).

The base class `CorpusReader` defines a few general-purpose methods for listing and accessing the files that make up a corpus. Corpora vary widely in the types of content they include. It is up to the subclasses to define data access methods that provide access to the information in the corpus. However, corpus reader subclasses should be consistent in their definitions of these data access methods wherever possible.

All the details about the nltk corpus is available at [this link](https://www.nltk.org/howto/corpus.html).

In the following sections we see how to subclass `̀TokenizerI` and `CorpusReader`.

## Define a custom Tokenizer, inherits from `TokenizerI`

NLTK has an interface for tokenizing a string `nltk.tokenize.api.TokenizerI`.

Subclasses must define `tokenize()` or `tokenize_sents()` (or both).

The NLTK api defines a processing interface for tokenizing a string.
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

# Sources
* https://regex101.com
* https://www.tensorflow.org/text/guide/subwords_tokenizer?hl=en#optional_the_algorithm
* https://www.nltk.org/book_1ed
* https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk
