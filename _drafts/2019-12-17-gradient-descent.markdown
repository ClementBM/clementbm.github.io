---
layout: post
title:  "Gradient descent explained"
date:   2019-12-17
categories: [gradient descent, neural network]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Gradient Descent

This article doesn't go through the details of how to implement a neural network.
It aims to give you the main theory about how it works, and it could show you hopefully on a different angle how things linked together.

Gradient is obtained by using the back propagation algorithm, indeed back propagation is a differentiation algorithm.

An important aspect of the design of a deep neural networks is the choice of the cost function. Cost functions for neural networks are quasi same as those for oher parametric models such as linear models.
In most cases, our parametric model defines a distribution $$p(y|x;\theta)$$ and we simply use the principle of maximum likelihood? This means we use the cross entropy between the training data and the model's prediction as the cost function.

> Learning Conditional Distributions with Maximum likelihood.

Most modern neural network are trained using maximum likelihood. This means that the cost function is simply the negative log-likelihood, equivalently described as the cross-entropy between the training data and the model distribution.

## Introduction
Backpropagation, for "backward propagationn of errors" is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weight.
It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.
Backpropagation was first invented in the 1970s as a general optimization method for performing automatic differentiation of complex nested functions.

> Presentation
During the lecture of this post, mister *legal* alien :alien: will take a look at :telescope:  the demonstration and make comments when he wants to.

## Notations
This part defined the notations that will be used in the later demonstration. It may sound verbose but I prefer to make the notations explicit so that everyone with a basic mathematic knowledge should be able to understand this article.

### $$X$$ *as* the sample
The dataset X is reprensented as $$\underline{\underline{X}}$$, an $$\text{M}$$-by-$$\text{N}$$ matrix, i.e. $$\underline{\underline{X}} \in \mathbb{R}^{\text{M} \times \text{N}}$$

$$
\underline{\underline{X}} =
\begin{pmatrix}
    x_{11} & \dots  & x_{1j}  & \dots  & x_{1\text{N}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    x_{i1} & \dots  & x_{ij}  & \dots  & x_{i\text{N}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    x_{\text{M}1} & \dots  & x_{\text{M}j}  & \dots  & x_{\text{MN}}
\end{pmatrix}
$$

$$\underline{x_i}$$ is the $$i^{\text{th}}$$ sample value of the dataset, i.e. : $$
\underline{x_i} = 
\begin{pmatrix}
    x_{i1} & \dots  & x_{ij}  & \dots  & x_{i\text{N}} \\
\end{pmatrix}$$

$$\underline{X_j}$$ is the expression of the $$j^{\text{th}}$$ explanatory variable, i.e. : 
$$
\underline{X_j} = 
\begin{pmatrix}
    x_{1j} \\
    \vdots \\
    x_{ij} \\
    \vdots \\
    x_{\text{M}j}
\end{pmatrix}
$$

> :alien: *alien says* :speech_balloon:\
M the number of samples we have\
N the dimension of the input, or the number of explanatory variables\
**Vectors** are underlined like this: $$\underline{x}$$\
**Matrices** are double-underlined like this: $$\underline{\underline{X}}$$

### $$Y$$ *as* the target
The variable to be predicted or the dependant variable is defined as $$\underline{\underline{Y}}$$, an $$\text{M}$$-by-$$\text{P}$$ matrix, i.e. $$\underline{\underline{Y}} \in \mathbb{R}^{\text{M} \times \text{P}}$$
$$
\underline{\underline{Y}} = 
\begin{pmatrix}
    y_{11} & \dots  & y_{1j}  & \dots  & y_{1\text{P}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    y_{i1} & \dots  & y_{ij}  & \dots  & y_{i\text{P}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    y_{\text{M}1} & \dots  & y_{\text{M}j}  & \dots  & y_{\text{MP}}
\end{pmatrix}
$$

$$\underline{y_i}$$ is the $$i^{\text{th}}$$ sample of the output target, i.e. $$
\underline{y_i} = 
\begin{pmatrix}
    y_{i1} & \dots  & y_{ij}  & \dots  & y_{i\text{P}} \\
\end{pmatrix}
$$
$$\underline{Y_j}$$ is the expression of the $$j^{\text{th}}$$ dependant variable, i.e. : 
$$
\underline{Y_j} = 
\begin{pmatrix}
    y_{1j} \\
    \vdots \\
    y_{ij} \\
    \vdots \\
    y_{\text{M}j}
\end{pmatrix}
$$

> :alien: *alien says* :speech_balloon:\
> M the number of samples we have\
> P the dimension of the target

### $$\mathcal{D}$$ as dataset
The dataset is just the combinations of X and Y, that is

$$
\mathcal{D} = \big \{(\underline{x_i}, \underline{y_i}) \big \}, \forall i \in [1..\text{M}]
$$

### $$NN$$ *as* Neural Network
A dense neural network can be defined by a composite transfer fonction $$F$$ composed of functions $$f_{\Theta_{\ell}}$$, $$ \forall \ell \in [1..\text{L}]: \big [ f_{\Theta_1}, ... , f_{\Theta_{\ell}}, ...,f_{\Theta_{\text{L}}} \big ] $$ and input parameters $$\underline{x_i}$$:

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} \rightarrow \mathbb{R}^{\text{P}} \\
    \underline{x_i} \rightarrow F(\underline{x_i}) = \underline{\hat{y_i}}
\end{array}
$$

:triangular_flag_on_post: *to simplify* :triangular_flag_on_post:
> $$\underline{x}$$ is equivalent to $$\underline{x_i}$$\
$$\underline{\hat{y}}$$ is equivalent to $$\underline{\hat{y_i}}$$

so we can write

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} \rightarrow \mathbb{R}^{\text{P}} \\
    \underline{x} \rightarrow F(\underline{x}) = \underline{\hat{y}}
\end{array}
$$

:telescope: *with* :telescope:
> $$\underline{\hat{y}}$$ the calculated output of the neural network. The objective is to have  $$\underline{\hat{y}}$$ as close as possible to $$\underline{y}$$.\
$$F$$ the composite transfer fonction.

For the $$\ell^{\text{th}}$$ layer we have

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{\ell-1}} \rightarrow \mathbb{R}^{\text{H}_\ell} \\
    \underline{a_{\ell-1}} \rightarrow f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = \underline{a_{\ell}}
\end{array}
$$

For the first layer

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} = \mathbb{R}^{\text{H}_0} \rightarrow \mathbb{R}^{\text{H}_1} \\
    \underline{x} \rightarrow f_{\Theta_1}(\underline{x}) = \underline{a_{1}}
\end{array}
$$

And the last one

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{\text{L}-1}} \rightarrow \mathbb{R}^{\text{H}_{\text{L}}} = \mathbb{R}^{\text{P}} \\
    \underline{a_{\text{L}-1}} \rightarrow f_{\Theta_{\text{L}}}(\underline{a_{\text{L}-1}}) 
        =\underline{a_{\text{L}}}
        =\underline{\hat{y}}
\end{array}
$$

:telescope: *with* :telescope:
> $$\text{H}_{\ell-1}$$ the size of the $$\ell-1^{\text{th}}$$ hidden layer
$$\text{H}_{\ell}$$ the size of the $$\ell^{\text{th}}$$ hidden layer\
$$f_{\Theta_{\ell}}$$ the transfer fonction for the $$\ell^{\text{th}}$$ layer\
$$\underline{a_{\ell-1}}$$ the input of the $$\ell^{\text{th}}$$ layer\
$$\underline{a_{\ell}}$$ the output of the $$\ell^{\text{th}}$$ layer

We'll dig into the details of the transfer fonction a little bit more later. For now, we try to consider the neural network at a granular point of view. So, given the transfer fonction $$f_{\Theta_{\ell}}$$, the whole network should look like the composition of the $$\text{P}$$ transfer fonctions listed in $$F$$

$$
    F(\underline{x})
    =
    f_{\Theta_{\text{L}}}(\dots (f_{\Theta_{\ell}}(\dots (f_{\Theta_2}(f_{\Theta_1}(\underline{x}))) \dots)) \dots)
$$

Composing functions is like chaining them. The output of the inner function becomes the input of the outer function. Given two functions $$f$$ and $$g$$, the composite function $$h$$ resulting is $$h(x) = g(f(x))$$ which is also noted as $$h(x) = (g \circ f) (x)$$. So the later wordy equation can be written in an easier way

$$
    F(\underline{x})
    =
    (f_{\Theta_{\text{L}}} \circ \cdots \circ f_{\Theta_{\ell}} \circ \cdots f_{\Theta_1})(\underline{x})
$$

> :alien: *alien says* :speech_balloon:\
$$\text{L}$$ the number of layers of the neural network\
$$F$$ is the composite transfer fonction. There is one transfer fonction $$f_{\Theta_{\ell}}$$ per layer, each one enables passing from layer $$\ell$$-1 to layer $$\ell$$, with $$\ell \in [1..\text{L}]$$

What we've just finished to define is the feedforward propagation. As we saw this algorithm passes the inputs from one layer to the other thanks to the transfer fonctions of each layer of the neural network.

### $$\mathcal{L}$$ *as* Loss function and $$E$$ *as* Error
$$\mathcal{L}$$ is a function of output $$y$$ and of the predicted output $$\hat{y}$$. It represents a kind of difference between the expected and the actual output. There are many ways to define a loss function. Historically for neural networks the loss function used was the mean squared error (MSE). For the $$i^{\text{th}}$$ sample of the dataset we have

$$
\forall{i} \in [1..\text{M}],
\mathcal{L}_{\text{MSE}}(\underline{y_i}, \hat{\underline{y_i}})=
{1 \over \text{P}} \sum_{j=1}^{\text{P}}(\hat{y_{ij}} - y_{ij})^2
$$

However, as the MSE function (mean squared error) is not convex for neural networks, we usually prefer to use the **Cross-Entropy** loss function. In our context of multiclass classification as the output $$\underline{y_i}$$ has P dimension, it's better to specificy that we are using the **Multiclass Cross-Entropy** Loss. Another name frequently used is the **Negative Log Likelihood** or LLE as we'd like to predict the probability of a sample to be in one of P classes, given a multinomial distribution... ?

$$
\mathcal{L}_{\text{CE}}(\underline{y_i}, \hat{\underline{y_i}})=
-\sum_{j=1}^{\text{P}} {y_{ij}} \log(\hat{y_{ij}})
$$

We define the overall error as the sum of the losses over the dataset

$$
E = {1 \over \text{M}} \sum_{i=1}^{\text{M}} \mathcal{L}(\underline{y_i}, \hat{\underline{y_i}})
$$

> The loss and the error might not be the definitions frequently used, so be carefull with that.

Error follows by summing through the samples of the dataset

$$
E_{\text{CE}}=
-{1 \over \text{M}} \sum_{i=1}^{\text{M}} \sum_{j=1}^{\text{P}} {y_{ij}} \log(\hat{y_{ij}}))
$$

> :alien: *alien says* :speech_balloon:\
$$y_{ij}$$ is the $$j^{\text{th}}$$ component of the target variable in the $$i^{\text{th}}$$ sample.\
$$\hat{y_{ij}}$$ is the $$j^{\text{th}}$$ component of the predicted target variable (estimated by the neural network) in the $$i^{\text{th}}$$ sample.

### A *as* activation
Before defining the gradient descent algorithm, we should present a more detailed version of the transfer function to show the role of the parameter $$\Theta$$ and also define the link functions for the hidden layer (activation function) and for the last layer (output function).

Remember that for the $$\ell^{\text{th}}$$ layer we have

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{ \ell - 1}} \rightarrow \mathbb{R}^{\text{H}_{ \ell }} \\
    a_{\ell-1} \rightarrow f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = \underline{a_{\ell}}
\end{array}
$$

We define $$\underline{z_{\ell}} \in \mathbb{R}^{\text{H}_{ \ell }}$$
as
$$
\underline{z_{\ell}} = \underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}}
$$
and $$g_{\ell}: \mathbb{R}^{\text{H}_{ \ell }} \rightarrow \mathbb{R}^{\text{H}_{ \ell }}$$
as

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{ \ell }} \rightarrow \mathbb{R}^{\text{H}_{ \ell }} \\
    \underline{z_{\ell}} \rightarrow g_{\ell}(\underline{z_{\ell}}) = f_{\Theta_{\ell}}(\underline{a_{\ell-1}})
\end{array}
$$

For the first layer

$$
\begin{array}{l}
    \underline{z_1} = \underline{\underline{\Theta_1}} \underline{x}\\
    \underline{a_1} = f_{\Theta_1}(\underline{x}) = g_1(\underline{z_1})
\end{array}
$$

and for the last layer

$$
\begin{array}{l}
    \underline{z_{\text{L}}} = \underline{\underline{\Theta_{\text{L}}}} \underline{a_{\text{L-1}}}\\
    \underline{a_{\text{L}}} = f_{\Theta_{\text{L}}}(\underline{a_{\text{L-1}}}) = g_{\text{L}}(\underline{z_{\text{L}}})
\end{array}
$$

The transfer fonction is composed of a matrix multiplication of $$\underline{a_{\ell-1}}$$ with  $$\underline{\underline{\Theta_{\ell}}}$$, respectivly the input and the weigts of the $$\ell^{\text{th}}$$ layer, and the activation function $$g_{\ell} $$

$$
g_{\ell} : \mathbb{R^{\text{H}_{\ell}}} \rightarrow \mathbb{R^{\text{H}_{\ell}}}
$$

$$
f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = g_{\ell}(\underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}}) = \underline{a_{\ell}}
$$

For the first layer

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} \rightarrow \mathbb{R}^{\text{H}_1} \\
    \underline{x} \rightarrow 
    g_1(\underline{\underline{\Theta_1}} \underline{x})
\end{array}
$$

$$
\forall i \in [1..\text{H}_1], \forall j \in [1..\text{N}]:
\underline{\underline{\Theta_1}} = 
\begin{pmatrix}
    \theta_{1, 11} & \dots  & \theta_{1, 1j}  & \dots  & \theta_{1, 1\text{N}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{1, i1} & \dots  & \theta_{1, ij}  & \dots  & \theta_{1, i\text{N}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{1, \text{H}_11} & \dots  & \theta_{1, \text{H}_1j}  & \dots  & \theta_{1, {\text{H}_1}\text{N}}
\end{pmatrix}
,
\underline{\underline{\Theta_1}} \in \mathbb{R}^{\text{H}_1 \times \text{N}}
$$

The $$\ell^{th}$$ layer

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{\ell-1}} \rightarrow \mathbb{R}^{\text{H}_{\ell}} \\
    \underline{a_{\ell-1}} \rightarrow 
    g_{\ell}(\underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}})
\end{array}
$$

$$
\forall i \in [1..\text{H}_{\ell}],
\forall j \in [1..\text{H}_{\ell-1}],
\forall \ell \in [1..\text{L}]:
\underline{\underline{\Theta_{\ell}}} = 
\begin{pmatrix}
    \theta_{\ell,11} & \dots  & \theta_{\ell,1j}  & \dots  & \theta_{\ell,1\text{H}_{\ell-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\ell,i1} & \dots  & \theta_{\ell,ij}  & \dots  & \theta_{\ell,i\text{H}_{\ell-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\ell,\text{H}_{\ell}1} & \dots  & \theta_{\ell,\text{H}_{\ell}j}  & \dots  & \theta_{\ell,{\text{H}_{\ell}}\text{H}_{\ell-1}}
\end{pmatrix}
,
\underline{\underline{\Theta_{\ell}}} \in \mathbb{R}^{\text{H}_{\ell} \times \text{H}_{\ell-1}}
$$

And the last one

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{\text{L}-1}} \rightarrow \mathbb{R}^{\text{P}} \\
    \underline{a_{\text{L}-1}} \rightarrow
    g_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}} \underline{a_{\text{L}-1}})
\end{array}
$$

$$
\forall i \in [1..\text{P}], \forall j \in [1..\text{H}_{\text{L}-1}]:
\underline{\underline{\Theta_{\text{L}}}} = 
\begin{pmatrix}
    \theta_{\text{L},11} & \dots  & \theta_{\text{L},1j}  & \dots  & \theta_{\text{L},1\text{H}_{\text{L}-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\text{L},i1} & \dots  & \theta_{\text{L},ij}  & \dots  & \theta_{\text{L},i\text{H}_{\text{L}-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\text{L},\text{P}1} & \dots  & \theta_{\text{L},\text{P}j}  & \dots  & \theta_{\text{L},{\text{P}}\text{H}_{\text{L}-1}}
\end{pmatrix}
,
\underline{\underline{\Theta_{\text{P}}}} \in \mathbb{R}^{\text{P} \times \text{H}_{\text{L}-1}}
$$

> :alien: *alien says* :speech_balloon:\
$$g_{\ell}$$ is the activation function of the $$\ell^{\text{th}}$$ hidden layer.\
$$g_{\text{L}}$$ is the output function of the last layer.

In many neural network and in this article we will take the sigmoid function for the activation and the softmax for the output.


### Gradient Descent *as* the Optimization Algorithm
Up to now, we defined the datasets composed of $$\underline{\underline{X}}$$ and $$\underline{\underline{Y}}$$, the forward propagation algorithm which calculates an estimation of the target $$\underline{\underline{\hat{Y}}}$$, and an error function $$E_{\text{CE}}$$. Recall that we want $$\underline{\underline{\hat{Y}}}$$ to be as close as possible to $${\underline{\underline{Y}}}$$. The error function $$E_{\text{CE}}$$ comes in handy by giving us a metric to quantify as far are we from the ground truth $${\underline{\underline{Y}}}$$. And so, "as close as possible" can be mathematically translated into "minimizing the error function $$E_{\text{CE}}$$"

$$
\arg \min_{\Theta} E_{\text{CE}}(\Theta)
$$

How do we minimize a fonction ? Well... with an optimization algorithm !
In this post we will use the gradient descent. Gradient descent belong to the family of the `first-order optimization algorithms`. It enables us to minimize the error $$E_{\text{CE}}$$ using the gradient with respect to the weights $$\Theta$$, so at t$$^{\text{th}}$$ iteration

$$
\underline{\underline{\Theta_{\ell}}}^{(t+1)} = \underline{\underline{\Theta_{\ell}}}^{(t)} - \alpha{\partial E(\Theta_{\ell}^{(t)}) \over \partial\Theta_{\ell}}
$$

> :alien: *alien says* :speech_balloon:\
$$\alpha$$ is the learning rate of the gradient descent algorithm\
$$E_{\text{CE}}$$ is a non linear error function. It depends on $$\underline{\underline{X}}$$ and $$\Theta_{\ell}$$, and it must defined and differentiable in the neighborhood of $$\Theta_{\ell}^{(t)}$$\
$${\partial E(\Theta_{\ell}^{(t)}) \over \partial\Theta_{\ell}}$$ is the partial derivative of $$E_{\text{CE}}$$ according to $$\Theta_{\ell}$$ the tuning parameter

### Breaking the last layer
First things first, the last layer of the network:

$$
    \forall i \in [1..\text{P}] \text{, }
    \forall j \in [1..\text{H}_{\text{L-1}}] \text{ and }
    \forall k \in [1..\text{M}]
$$

$$
    {\partial E_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial \over \partial\Theta_{\text{L},ij}}
    \Big (
    -{1 \over \text{M}} \sum_{k=1}^{\text{M}} \sum_{i=1}^{\text{P}} {y_{ki}} \log(\hat{y_{ki}})
    \Big )
$$

Before derivating the error let's simplify using the chain rule of derivation along with $$\underline{a_{\text{L}}}$$ and $$\underline{z_{\text{L}}}$$, because $$E_{\text{CE}}$$ depends on $$a_{\text{L}}$$ which is equals to $$\hat{y}$$, and $$a_{\text{L}}$$ depends on $$z_{\text{L}}$$

$$
    {\partial E_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial E_{\text{CE}} \over \partial \underline{a_{\text{L}}}}
    {\partial \underline{a_{\text{L}}} \over \partial \underline{z_{\text{L}}}}
    {\partial \underline{z_{\text{L}}} \over \partial \Theta_{\text{L},ij}}
$$

We continue simplifying by taking only one sample before calculate the derivative.
The previous equation becomes

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial E_{\text{CE}} \over \partial \underline{a_{\text{L}}}}
    {\partial \underline{a_{\text{L}}} \over \partial \underline{z_{\text{L}}}}
    {\partial \underline{z_{\text{L}}} \over \partial \Theta_{\text{L},ij}}
$$

To make the calculation of the derivative even simpler we pick scalar instead of vectors so that:

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial E_{\text{CE}} \over \partial a_{\text{L},i}}
    {\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
    {\partial z_{\text{L},i} \over \partial \Theta_{\text{L},ij}}
$$

with

$$
\underline{a_{\text{L}-1}} \in \mathbb{R}^{H_{\text{L}-1}}
$$

$$
\underline{z_{\text{L}}}, \underline{a_{\text{L}}} \in \mathbb{R}^{H_{\text{P}}}
$$

**First**

$$
\begin{equation*}
{\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}=
{\partial \over \partial a_{\text{L},i}}
\Big (
    - \sum_{k=1}^{\text{P}} {y_{k}} \log(\hat{y_{k}})
\Big )\\
=
{\partial \over \partial a_{\text{L},i}}
\Big (
    - \sum_{k=1}^{\text{P}} {y_{k}} \log(a_{\text{L},k})
\Big )
\end{equation*}
$$

we have
$$
y_k =
\begin{cases}
    0 & \text{if \(k \neq i\) } \\
    1 & \text{if \(k = i\) }
\end{cases}
$$
therefore when $$i = k$$, $$y_i = 1$$

$$
{\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}=
-
{\partial \over \partial a_{\text{L},i}}
\Big (
    \log(a_{\text{L},i})
\Big )
$$

(A une constante près à cause de la dérive du log)

**Second**

$$
\begin{equation}
{\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
=
{\partial \over \partial z_{\text{L},i}}
\Big (
    g_{\text{L}}(z_{\text{L},i})
\Big )
=
{\partial \over \partial z_{\text{L},i}}
\Big (
    \frac{\mathrm{e}^{z_{\text{L},i}}}{\sum_{k=1}^{\text{L}} \mathrm{e}^{z_{\text{L},k}}}
\Big )\\
=
\frac{ \mathrm{e}^{z_{L},i} \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},k} - \big ( \mathrm{e}^{z_{L},i} \big ) ^2}
{ \big ( \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},k} \big ) ^ 2}
=
\frac{ \mathrm{e}^{z_{L},i} }
{ \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},i} }
-
\big (
\frac{ \mathrm{e}^{z_{L},i} }
{ \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},i} }
\big ) ^2 \\
=
g_{\text{L}}(z_{\text{L},i}) - g^2_{\text{L}}(z_{\text{L},i})
\end{equation}
$$

so that

$$
\begin{equation}
{\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
=
a_{\text{L},i}(1 - a_{\text{L},i})
\end{equation}
$$

**Third**

$$
{\partial z_{\text{L},i} \over \partial\Theta_{\text{L},ij}}
=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \underline{\underline{\Theta_{\text{L}}}} \underline{a_{\text{L}-1}}
\Big )_i\\
=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \sum_{k=1}^{\text{H}_{\text{L}-1}} \theta_{\text{L},ik} a_{\text{L}-1,k}
\Big )\\
=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \theta_{\text{L},i1} a_{\text{L}-1,1} + \cdots + \theta_{\text{L},ik} a_{\text{L}-1,k} + \cdots + \theta_{\text{L},iH_{\text{L}-1}} a_{\text{L}-1,H_{\text{L}-1}}
\Big )
$$

The derivative is equal to zero when $$j \neq k$$ so for $$j = k$$

$$
{\partial z_{\text{L},i} \over \partial\Theta_{\text{L},ij}}
=
a_{\text{L}-1,j}
$$

Then putting it together with the chained derivation

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    \underbrace{
        {\partial E_{\text{CE}} \over \partial a_{\text{L},i}}
        {\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
    }_{\delta_{\text{L},i}}
    {\partial z_{\text{L},i} \over \partial \Theta_{\text{L},ij}} \\
    =
    - \frac{1}{a_{\text{L},i}} a_{\text{L},i} (1 - a_{\text{L},i}) a_{\text{L}-1,j}\\
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    (a_{\text{L},i} - 1) a_{\text{L}-1,j}
$$

As $$\underline{a_{\text{L}}}$$ and $$\underline{a_{\text{L}-1}}$$ are row vectors, the previous equation can be vectorized

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    (\underline{a_{\text{L}}} - 1)^{\text{T}} \underline{a_{\text{L}-1}}
$$

And sum up over the M training examples

$$
    {\partial E_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    \frac{1}{M} \sum_{k=1}^{M} (\underline{a_{\text{L},k}} - 1)^{\text{T}} \underline{a_{\text{L}-1,k}}
$$

### Breaking the hidden layers

Given $$i \in [1..\text{H}_{\ell}]$$, $$j \in [1..\text{H}_{\ell-1}]$$ and $$\underline{\underline{\Theta_{\ell}}} \in \mathbb{R}^{\text{H}_{\ell} \times \text{H}_{\ell - 1} }$$

$$
{\partial \mathcal{L}_{\text{CE}} 
    \over 
\partial\Theta_{\ell,ij}}
=
{\partial \mathcal{L}_{CE}
    \over 
\partial a_{\ell,i}}
\overbrace{
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
}^{\text{same for all hidden layers}}
\underbrace{
    {\partial z_{\ell,i}
        \over 
    \partial\Theta_{\ell,ij}}
}_{\text{same as the final layer}}
$$

**First**

Given $$ z_{\text{L}+1}, a_{\ell+1}  \in \mathbb{R}^{H_{\ell+1}} $$ and $$ z_{\text{L}}, a_{\ell}  \in \mathbb{R}^{H_{\ell}} $$ and $$ \delta_{\ell} \in \mathbb{R}^{H_{\ell}}$$

Again, we apply the chain rule of derivation

$$
{\partial \mathcal{L}_{CE}
    \over 
\partial a_{\ell,i}}
=
\sum^{H_{\ell+1}}_{k=1}
{\partial \mathcal{L}_{CE}
    \over 
\partial a_{\ell+1,k}}
{\partial a_{\ell+1,k}
    \over 
\partial z_{\ell+1,k}}
{\partial z_{\ell+1,k}
    \over 
\partial a_{\ell,i}}
$$

We define

$$
\delta_{\ell,i}
=
{\partial \mathcal{L}_{CE}
    \over 
\partial a_{\ell,i}}
{\partial a_{\ell,i}
    \over 
\partial z_{\ell,i}}
$$

**1.1** and **1.2**

$$
\big (
{\partial \mathcal{L}_{CE}
    \over 
\partial a_{\ell,i}}
{\partial a_{\ell,i}
    \over 
\partial z_{\ell,i}}
\big )
=
\delta_{\ell+1,k}
$$

As we go backward, $$ \delta_{\ell+1,k} $$ has already been calculated.

With $$ \underline{\delta_{\ell + 1}} \in \mathbb{R}^{H_{\ell+1}} $$

**1.3**

$$
{\partial z_{\ell+1,k}
    \over 
\partial a_{\ell,i}}
=
{\partial
    \over 
\partial a_{\ell,i}}
\big (
    \underline{\underline{\Theta_{\ell + 1}}} \underline{a_{\ell}}
\big )_k \\
=
{\partial
    \over 
\partial a_{\ell,i}}
\big (
    \sum^{H_{\ell}}_{c=1} \theta_{\ell + 1,kc} a_{\ell,c}
\big ) \\
=
{\partial
    \over 
\partial a_{\ell,i}}
\theta_{\ell + 1,k1} a_{\ell,1} + \cdots + \theta_{\ell + 1,kc} a_{\ell,c} + \cdots + \theta_{\ell + 1,kH_{\ell}} a_{\ell,H_{\ell}}
$$

the derivatives are nulls if $$ i \neq c $$ then for $$i = c$$

$$
{\partial z_{\ell+1,k}
    \over 
\partial a_{\ell,i}}
=
\theta_{\ell + 1,ki}
$$

With $$ \underline{\underline{\Theta_{\ell + 1}}} \in \mathbb{R}^{H_{\ell+1} \times H_{\ell}} $$

**Second**

$$
{\partial a_{\ell,i}
    \over 
\partial z_{\ell,i}}
=
{\partial
    \over 
\partial z_{\ell,i}}
\big (
    g_{\ell}(z_{\ell,i})
\big )\\
=
{\partial
    \over 
\partial z_{\ell,i}}
\big (
    \frac{1}
    {1 + \mathrm{e}^{z_{\ell,i}}}
\big )
$$

$$
{\partial a_{\ell,i}
    \over 
\partial z_{\ell,i}}
=
a_{\ell,i} (1 - a_{\ell,i})
$$

**Third**
Same as for the output layer

$$
{\partial z_{\ell,i}
    \over 
\partial\Theta_{\ell,ij}}
=
a_{\ell-1,j}
$$

**Finally**

$$
{\partial \mathcal{L}_{\text{CE}} 
    \over 
\partial\Theta_{\ell,ij}}
=
\big (
    \sum^{H_{\ell+1}}_{k=1}
    \delta_{\ell+1,k}
    \theta_{\ell+1,ki}
\big )
a_{\ell,i}
(1 - a_{\ell,i})
a_{\ell-1,j}
$$

It can be vectorized as follow

$$
{\partial \mathcal{L}_{\text{CE}} 
    \over 
\partial \underline{\underline{\Theta_{\ell}}}}
=
\big (
    \underline{\delta_{\ell+1}} \underline{\underline{\Theta_{\ell+1}}}
\big )^{\text{T}}
\odot
\underline{a_{\ell}}^{\text{T}}
\odot
(1 - \underline{a_{\ell}})^{\text{T}}
\underline{a_{\ell-1}}
$$

And summing up over the samples:

...


# Sources
* https://www.wikiwand.com/fr/Algorithme_du_gradient
* https://bigtheta.io/2016/02/27/the-math-behind-backpropagation.html
* https://brilliant.org/wiki/backpropagation/
* https://en.wikipedia.org/wiki/Function_of_several_real_variables
* https://gombru.github.io/2018/05/23/cross_entropy_loss/
* https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
* https://stats.stackexchange.com/questions/323896/explanation-for-mse-formula-for-vector-comparison-with-euclidean-distance
* https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_d%C3%A9rivation_des_fonctions_compos%C3%A9es
* https://en.wikipedia.org/wiki/Function_composition
* https://en.wikipedia.org/wiki/Mathematical_optimization
* https://en.wikipedia.org/wiki/Gradient_descent

**important**
* https://www.youtube.com/watch?v=-p1ldISb90Q
* http://www.cs.cornell.edu/courses/cs5740/2016sp/resources/backprop.pdf
* http://www.math.hkbu.edu.hk/~mhyipa/nndl/chap2.pdf
* https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf
* https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
* https://math.stackexchange.com/questions/137936/log-likelihood-gradient-and-hessian
* http://cs231n.github.io/neural-networks-case-study

**stuttgart**
* https://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2018/06/ML_NeuralNetworks_SS18.pdf
* https://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2018/04/ML_Intro_SS18.pdf
* https://ipvs.informatik.uni-stuttgart.de/mlr/teaching/machine-learning-ss-18/
* https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/

**log likelihood**
* https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf

**Andrew NG**
* https://www.youtube.com/watch?v=x_Eamf8MHwU

**second order optimization**
* https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/04-secondOrderOpt.pdf

**logistic regression**
* https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf

**derivative**
* http://cs231n.stanford.edu/vecDerivs.pdf

**softmax**
* https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
* https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax