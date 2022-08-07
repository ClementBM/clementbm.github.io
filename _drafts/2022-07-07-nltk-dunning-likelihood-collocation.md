---
layout: post
title:  "Get collocations with Dunning likelihood method"
excerpt: "Explore a corpus with NLTK and the Dunning likelihood method to find common collocations"
date:   2022-07-07
categories: [theory]
tags: [NLTK, NLP, statistic]
---

![Grape vine fruit](/assets/2022-07-07/pexels-maur%C3%ADcio-mascaro-9192252.jpg)

# Co-occurrence

In linguistics, co-occurrence or cooccurrence is an above-chance frequency of occurrence of two or more terms in the same text (phrase, paragraph, corpus...), from a text corpus alongside each other in a certain order. Co-occurrence in this linguistic sense can be interpreted as an indicator of semantic proximity or an idiomatic expression. Corpus linguistics and its statistic analyses reveal patterns of co-occurrences within a language and enable to work out typical collocations for its lexical items. A co-occurrence restriction is identified when linguistic elements never occur together. Analysis of these restrictions can lead to discoveries about the structure and development of a language.

Co-occurrence can be seen an extension of word counting in higher dimensions. Co-occurrence can be quantitatively described using measures like correlation or mutual information.

It's possible that terms are mutally dependent when the use of the two is very frequent.
Statistic tests can prove the hypothetic dependance, like test of mutual information or coefficient verosimilitud.

Co-occurrence is the co presence statistically significative of two or multiple unit within the same contextual window.

When it's proved that there is a semantical or gramatical dependency between two words, we call it colocation.

Location are "stable" coocurrence, group of words forming one lexical unit with a typical/own/particular meaning

# Phraseme
A phraseme also called a "set phrase" or "idiomatic phrase", "multi-word expression", or "idiom" is a multi word utterance where at least one of whose components is selectionnaly constrained or restricted by linguistic convention such that it is not freely chosen.

At the contrary, there are collocations such as "infinite patience" where one of the words is chosen freely (patience) based on the meaning the speaker wishes to express while the choice of the other word (infinite) is constrained by the convention of the English language.

Both kinds of expression are phrasemes and can be contrasted with "free phrases", expressions where all of the members are chosen freely, based exclusively on their meaning and the message that the speaker wishes to communicate.

Saying, or a proverb, figure of speech, foxed expression

# Collocation
Cooccureence are close to collocationn, which is a form of idiomatic expression caused by a systematic coocurrence.

# Cooccureence and champ lexical
When two words or other linguistic unit, have a semantical relationship, cooccurrence notion is at the base of thematic, champ lexical, isotopie.

# Isotopie
in semantic and semiotiquen, isotopie is the redondancy of element in a corpus enabling to understand it?

For example, the redondancy of the first person (I), make it easy to understand that the same person is talking.
Redondancy of the same champ lexical enale us to understand that we are talking about the same theme.

# Dunning likelihood

```python
corpus_metric.story_text.collocations() # Dunning likelihood collocation
# three word window: corpus_metric.story_text.collocations(window_size=3)
```

open source; command line; Dark Souls; Nhat Hanh; Points Guy; Suite
legacy; Thich Nhat; largest chip; American Airlines; black holes; SICP
JavaScript; modern language; Google Analytics; source Ask


# Sources
* https://en.wikipedia.org/wiki/Collocation
* https://en.wikipedia.org/wiki/Co-occurrence
* See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of Statistical Natural Language Processing. MIT press, p. 162 https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22
* https://en.wikipedia.org/wiki/Pointwise_mutual_information
*  [Word association norms, mutual information, and lexicography](https://aclanthology.org/J90-1003.pdf)
*  https://www.nltk.org/howto/collocations.html
*  [Mutual information](https://towardsdatascience.com/mutual-information-prediction-as-imitation-da2cfb1e9bdd)
*  [Information theory](https://towardsdatascience.com/information-theory-a-gentle-introduction-6abaf99835ac)
*  [La vraisemblance](https://www.youtube.com/watch?v=P-AHaAP8fIk)
*  [Initiation à la statistique bayésienne](https://www.youtube.com/watch?v=5hN_plbtPjw)