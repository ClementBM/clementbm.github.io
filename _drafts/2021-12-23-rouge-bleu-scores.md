---
layout: post
title:  "ROUGE and BLEU scores for NLP model evaluation"
excerpt: "ROUGE and BLEU similarity metric for evaluating models"
date:   2021-12-23
categories: [NLP, Metrics]
tags: [ROUGE, BLEU]
---

![Blueberries and raspberries](/assets/2021-12-23/blueberries-raspberries.jpg)

In this post, I explain two common metrics used in the field of NLG (Natural Language Generation) and MT (Machine Translation).

**BLUE** score was first created to automatically evaluate machine translation, while **ROUGE** was created a little later inspired by BLUE to score a task of auto summurization.
Both metrics are calculated using n-gram co-occurrence statistic and they both range from 0 to 1, 1 meaning sentences are exactly the same.

Despite their relative simplicity, BLEU and ROUGE similarity metrics are quite reliable because they were proven to highly correlate with human judgements.

## BLEU
BLEU score stands for **B**i**l**ingual **E**valuation **U**nderstudy.

When evaluating machine translation, multiple caracterics are taken into account:
* adequacy
* fidelity
* fluency

Unigram matches tend to measure adequacy while longer n-grams matches account for fluency.

$$
BLEU_N =
BP \cdot
\exp{\left(
    \sum_{n=1}^N w_n \log p_n
\right)}
$$

$$
BP = 
    \left\{
        \begin{array}{ll}
            1 & \text{if } \ell_{hyp} \gt \ell_{ref} \\
            e^{1 - { \ell_{ref} \over \ell_{hyp} }} & \text{if } \ell_{hyp} \le \ell_{ref}
        \end{array}
    \right.
$$

$$
p_n =
{
    \sum_{n\text{-}gram \in hypothesis}
Count_{match}(n\text{-}gram)
    \over
    \sum_{n\text{-}gram \in hypothesis} Count(n\text{-}gram)
}=
{
    \sum_{n\text{-}gram \in hypothesis} Count_{match}(n\text{-}gram)
    \over
    \ell_{hyp}^{n\text{-}gram}
}
$$

| Type | Sentence | Length |
|--|--|--|
| Reference | The way to make people trustworthy is to trust them. | $$\ell_{ref}^{unigram} = 10$$ |
| Hypothesis | To make people trustworthy, you need to trust them. | $$\ell_{hyp}^{unigram} = 9$$ |

For this example we take parameters as the base line score, described in the paper, with $$ N = 4 $$, with a uniform distribution, therefore taking $$w_n = { 1 \over 4 }$$.


$$
BLEU_{N=4} =
BP \cdot
\exp{\left(
    \sum_{n=1}^{N=4} { 1 \over 4 } \log p_n
\right)}
$$

| n-gram | 1-gram | 2-gram | 3-gram | 4-gram |
|--|:--:|:--:|:--:|:--:|
| $$p_n$$ | $${ 7 \over 9 }$$ | $${ 5 \over 8 }$$ | $${ 3 \over 7 }$$ | $${ 1 \over 6 }$$ |

$$
BP = e^{1 - { \ell_{ref} \over \ell_{hyp} }} = e^{ - { 1 \over 9 }}
$$

![Bleu score unigrams](/assets/2021-12-23/bleu-unigrams.png)

![Bleu score bigrams](/assets/2021-12-23/bleu-bigrams.png)

## ROUGE
ROUGE score stands for **R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation.

Evaluation of summarization involves mesures of
* coherence
* conciseness
* grammaticality
* readability
* content

![Rouge-1 score unigrams](/assets/2021-12-23/rouge-unigrams.png)

$$
ROUGE_1=
{
    7 \over 8
} = 0.875
$$

### ROUGE-L


$$
\left\{
    \begin{array}{ll}
        R_{LCS} &= 
        {
            LCS(reference, hypothesis)
            \over
            m_{unigram}
        } \\
        P_{LCS} &=
        {
            LCS(reference, hypothesis)
            \over
            n_{unigram}
        } \\
        ROUGE_{LCS} &=
        { 
            (1 + \beta^2) R_{LCS} P_{LCS}
            \over
            R_{LCS} + \beta^2 P_{LCS}
        }
    \end{array}
\right.
$$


$$
\left\{
    \begin{array}{ll}
        R_{LCS} &= { 7 \over 8 } \\
        P_{LCS} &= { 7 \over 10 } \\
        ROUGE_{LCS} &=
        {
            (1 + \beta^2)  { 7 \over 8 } { 7 \over 10 }
            \over
            { 7 \over 8 } + \beta^2 { 7 \over 10 }
        } =
        {
            (1 + \beta^2) 49
            \over
            70 + \beta^2 56
        }
    \end{array}
\right.
$$

$$
ROUGE_{LCS}=
{
    (1 + 1^2)  49 \over  70 +1^2 56
}= 
{ 
    49 \over 63 
}
\approx 0.778
$$

## BLEU VS ROUGE

| BLEU score | ROUGE score |
|---|---|
| The more references sentences the better | The more references sentences the better |
| Precision oriented score | Recall oriented score, in this n-gram version |
| One version | Multiple versions |
| Can be computed on choices of n-grams | Can be computed on choices of n-grams |
| Initially for translation evaluation (**B**i**l**ingual **E**valuation **U**nderstudy) | Initially for summary evaluations (**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation) |
| Inexpensive automatic evaluation | Inexpensive automatic evaluation |
| correlates highly with human evaluation | correlates highly with human evaluation |
| count the number of overlapping units such as n-gram, word sequences, and word pairs between hypothesis and references | count the number of overlapping units such as n-gram, word sequences, and word pairs between hypothesis and references |
| rely on tokenization and word filtering, text normalization | rely on tokenization and word filtering, text normalization  |
| does not cater for different words that have the same meaning — as it measures syntactical matches rather than semantics | |

# Sources
* [Html Proofer](https://www.supertechcrew.com/jekyll-check-for-broken-links/)