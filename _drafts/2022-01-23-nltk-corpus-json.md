---
layout: post
title:  "How to load a corpus with nltk from json"
excerpt: "EDA on hacker news top stories with nltk"
date:   2022-01-23
categories: [EDA, NLP, ]
tags: [nltk, json]
---

In this post we'll use nltk features to perform a quick overview analysis of hacker news top stories dataset.
Natural Language Took Kit (NLTK), give us a nice starting point to perform exploratory data analysis on textual data.


https://github.com/HackerNews/API

https://www.nltk.org/howto/corpus.html

https://stackoverflow.com/questions/38179829/how-to-load-a-json-file-with-python-nltk

https://gist.github.com/JeremyEnglert/3eda4a123244c37b669472d9e8166ea6

Le module json est livré avec une méthode appelée loads(), le s dans loads() signifie string. Puisque nous voulons convertir des données de chaîne en JSON, nous utiliserons cette méthode