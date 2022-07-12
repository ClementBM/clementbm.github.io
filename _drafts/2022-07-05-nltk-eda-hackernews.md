---
layout: post
title:  "Exploratory data analysis on hacker news stories"
excerpt: "How to perform perform a quick analysis on hacker news top stories with NLTK"
date:   2022-07-05
categories: [study]
tags: [NLTK, NLP, visualization]
---

![Original fruit](/assets/2022-07-05/pexels-irina-kaminskaya-12633634.jpg)

This post is about corpus exploration with NLTK, it's the suite of the first post on CorpusReader.
Working with NLTK is a great way to learn not only the toolkits but the more about EDA in NLP in general, and good practice in coding.


- [Text class wrapper](#text-class-wrapper)
- [Sliceable](#sliceable)
- [Concordance](#concordance)
- [Find all regex](#find-all-regex)
- [Token distribution](#token-distribution)
- [Removing stop words](#removing-stop-words)
- [Plot frequency distribution](#plot-frequency-distribution)
- [Lexical dispersion plot](#lexical-dispersion-plot)
- [Recurrent pattern](#recurrent-pattern)
- [Sources](#sources)


As the first we load our corpus with the CorpusReader class creating in the last post.

```python
story_corpus = StoryCorpusReader()
```

# Text class wrapper
A wrapper around a sequence of simple (string) tokens, which is intended to support initial exploration of texts (via the interactive console). Its methods perform a variety of analyses
on the text's contexts and display the results
* counting
* concordancing
* collocation discovery

If you wish to write a program which makes use of these analyses, then you should bypass the `Text` class, and use the appropriate analysis function or class directly instead.

A `Text` is typically initialized from a given document or corpus.  E.g.:

```python
from nltk.text import Text
story_text = Text(story_corpus.sentences_tokens())
```

# Sliceable
Following method give the same results
```python
story_text[3:5]
story_text.tokens[3:5]
```

```python
>>> ['noninvasive', 'optical']
```

# Concordance
Prints a concordance for `word` with the specified context window. Word matching is not case-sensitive.

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
Find instances of the regular expression in the text. The text is a list of tokens, and a regexp pattern to match a single token must be surrounded by angle brackets.

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

Find most common tokens
```python
story_text.vocab().most_common(10)
```

`vocab()` is a dictionary. key token and value the number of times the token appears in the corpus.

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

To have better idea of the most frequent word of this corpus, we would like to remove stop words, that is, words that are common across all corpus.

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


```python
plt.figure(figsize=(18, 12))
story_vocab.plot(20, cumulative=False, percents=False, show=False)
plt.xticks(rotation=45)
```

![alt](/assets/2022-07-05/frequency-distribution.png)

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


# Sources
* NLTK