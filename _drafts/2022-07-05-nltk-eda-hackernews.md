---
layout: post
title:  "Easy exploratory data analysis on hacker news stories"
excerpt: "How to perform perform a quick analysis on hacker news top stories with NLTK"
date:   2022-07-05
categories: [study]
tags: [NLTK, NLP, visualization]
---

![Original fruit](/assets/2022-07-05/pexels-irina-kaminskaya-12633634.jpg)

This post is about corpus exploration with NLTK, it's the continuation of the first post on [How to extend NLTK CorpusReader?]({% post_url 2022-01-23-nltk-extend-corpusreader-jsonl %}).
Working with NLTK is a great way to learn how to use the toolkit while discovering more about NLP in general.

**Table of contents**

- [Text class wrapper](#text-class-wrapper)
- [Sliceable](#sliceable)
- [Concordance](#concordance)
- [Find all regex](#find-all-regex)
- [Token distribution](#token-distribution)
- [Removing stop words](#removing-stop-words)
- [Plot frequency distribution](#plot-frequency-distribution)
- [Lexical dispersion plot](#lexical-dispersion-plot)
- [Recurrent pattern](#recurrent-pattern)
- [Go further](#go-further)
- [Sources](#sources)


First we load our corpus with the CorpusReader class creating in the [previous post]({% post_url 2022-01-23-nltk-extend-corpusreader-jsonl %})

```python
story_corpus = StoryCorpusReader()
```

We then use a NLTK native class called `Text`.

# Text class wrapper

According to the documentation, the `Text` class is a wrapper around a sequence of simple (string) tokens, which is intended to support initial exploration of texts. Its methods perform a variety of analyses on the text's contexts and display the results.

A `Text` is typically initialized from a given document or corpus. In our case we use `sentences_tokens()` from the `StoryCorpusReader` class.

```python
from nltk.text import Text
story_text = Text(story_corpus.sentences_tokens())
```

`Text` can work as a collection. That is, you can access a token via an integer index.

# Sliceable

You can also slice the collection as shown below. The following two lines give the same results

```python
story_text[3:5]
story_text.tokens[3:5]
```

```python
>>> ['noninvasive', 'optical']
```

# Concordance

If you want to explore the context of certain term/token, concordance is a great way to have a quick view.
The `concordance()` function print the surrounding of a chosen word. Word matching is not case-sensitive.

```python
story_text.concordance("language")
```

```python
>>> Displaying 5 of 5 matches:
>>>  plant A probabilistic programming language in 70 lines of Python A better way
>>> n Binary Interface ABI Ask HN What language will you pick if you are to reinve
>>> using DNSSEC The case for a modern language Consistency Sin Cannabis use produ
>>> Morello ISAs The case for a modern language part 1 OpenAPI Tools Starving Afgh
>>>  station What's the Most Efficient Language Don t Worry Be Happy Today in the 
```

# Find all regex

If you want something more granular than concordance you can use the `findall()` function. It enables you to filter the result of the search based on regular expression. It finds instances of the regular expression in the text. The text is a list of tokens, and a `regexp` pattern to match a single token must be surrounded by angle brackets.

```python
story_text.findall("<.*><.*><Google>")
```

```python
>>> co-founder leaves Google; Migrations Can Google; great resignation
>>> Google; CUDA support Google; about the Google; sign for Google;
>>> Sideloading Is Google; founders of Google; at risk Google; Idempotent
>>> API Google; alternative to Google; Up with Google; absorbing them
>>> Google; a failure Google; logic behind Google
```

# Token distribution

It's also easy to find the most common tokens of the corpus. You can use the `vocab()` function

```python
story_text.vocab().most_common(10)
```

`vocab()` output a dictionary that has a token as a key and the number of occurence as a value. The value is the number of times the token appears in the corpus.

```python
>>> [('of', 81),
>>>  ('to', 74),
>>>  ('the', 73),
>>>  ('in', 69),
>>>  ('a', 63),
>>>  ('and', 59),
>>>  ('for', 54),
>>>  ('The', 50),
>>>  ('HN', 42),
>>>  ('is', 31)]
```

To have a better idea of the most frequent word of this corpus, we would like to remove stop words, that is, words that are common across all corpora.

# Removing stop words

```python
story_vocab = story_text.vocab()
stop_words = set(stopwords.words("english"))

removable_vocab_keys = []
for vocab_key in story_vocab.keys():
    if vocab_key.casefold() in stop_words:
        removable_vocab_keys.append(vocab_key)

for removable_vocab_key in removable_vocab_keys:
    story_vocab.pop(removable_vocab_key)
```

# Plot frequency distribution

After removing the most frequent words, we look at the distribution of the most frequent words.

```python
plt.figure(figsize=(18, 12))
story_vocab.plot(20, cumulative=False, percents=False, show=False)
plt.xticks(rotation=45)
```

![alt](/assets/2022-07-05/frequency-distribution.png)

`story_text.plot()` is a shortcut for `story_text.vocab().plot()`. Be careful of aliasing, i.e. changing `story_vocab` will also modify the outcome of `story_text.vocab()`. The vocab() being an accessor of the underlying private property `_vocab`.

# Lexical dispersion plot

```python
story_text.dispersion_plot(
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

# Recurrent pattern

```python
topstories[topstories["title"].str.contains("Ask HN")]["title"]
```

# Go further

If you wish to write a program which makes use of these analyses, then you should bypass the `Text` class, and use the appropriate analysis function or class directly instead.

# Sources
* NLTK