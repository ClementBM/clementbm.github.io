---
layout: post
title:  "Get collocations with Dunning likelihood method"
excerpt: "Explore a corpus with NLTK and the Dunning likelihood method to find common collocations"
date:   2022-07-07
categories: [EDA, NLP, tokenizer]
tags: [NLTK, json]
---


![Grape vine fruit](/assets/2022-07-07/pexels-maur%C3%ADcio-mascaro-9192252.jpg)

# Co-occurrence
In linguistics, co-occurrence or cooccurrence is an above-chance frequency of occurrence of two terms (also known as coincidence or concurrence) from a text corpus alongside each other in a certain order. Co-occurrence in this linguistic sense can be interpreted as an indicator of semantic proximity or an idiomatic expression. Corpus linguistics and its statistic analyses reveal patterns of co-occurrences within a language and enable to work out typical collocations for its lexical items. A co-occurrence restriction is identified when linguistic elements never occur together. Analysis of these restrictions can lead to discoveries about the structure and development of a language.

Co-occurrence can be seen an extension of word counting in higher dimensions. Co-occurrence can be quantitatively described using measures like correlation or mutual information.

# Idiom

# Collocation


# Dunning likelihood

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

# Sources
* https://en.wikipedia.org/wiki/Collocation
* https://en.wikipedia.org/wiki/Co-occurrence
* See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of Statistical Natural Language Processing. MIT press, p. 162 https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22