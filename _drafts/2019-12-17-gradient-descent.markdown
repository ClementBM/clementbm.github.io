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
This part defined the notations that will be used in the later demonstration. It may sound verbose but I prefer to make the notations explicit so that everyone with a basic mathematic knowledge should be able to understand the article.

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
**m** the number of samples we have\
**n** the dimension of the input, or the number of explanatory variables\
**Vectors** are underlined like this: $$\underline{x}$$\
**Matrices** are double-underlined like this: $$\underline{\underline{X}}$$

### $$Y$$ *as* the target
The variable to be predicted or the dependant variable is defined as $$\underline{\underline{Y}}$$, an $$M$$-by-$$L$$ matrix, i.e. $$\underline{\underline{Y}} \in \mathbb{R}^{M \times L}$$
$$
\underline{\underline{Y}} = 
\begin{pmatrix}
    y_{11} & \dots  & y_{1j}  & \dots  & y_{1\text{L}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    y_{i1} & \dots  & y_{ij}  & \dots  & y_{i\text{L}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    y_{\text{M}1} & \dots  & y_{\text{M}j}  & \dots  & y_{\text{ML}}
\end{pmatrix}
$$

$$\underline{y_i}$$ is the $$i^{\text{th}}$$ sample of the output target, i.e. $$
\underline{y_i} = 
\begin{pmatrix}
    y_{i1} & \dots  & y_{ij}  & \dots  & y_{i\text{L}} \\
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
> **m** the number of samples we have\
> **l** the dimension of the target

### D as dataset
Combinations of X and Y

### $$NN$$ *as* Neural Network
A dense neural network can be defined by a function whose parameters are $$\underline{x_i}$$ and the composite transfer fonction $$F$$, composed of functions $$f_{\Theta_{\ell}}$$, i.e. $$  \forall \ell \in [1..\text{P}]: [f_{\Theta_1}, ... , f_{\Theta_{\ell}}, ...,f_{\Theta_{\text{P}}}] $$

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} \rightarrow \mathbb{R}^{\text{L}} \\
    \underline{x_i} \rightarrow F(\underline{x_i}) = \underline{\hat{y_i}}
\end{array}
$$

:triangular_flag_on_post: *to simplify* :triangular_flag_on_post:
> $$\underline{x}$$ is equivalent to $$\underline{x_i}$$\
$$\underline{\hat{y}}$$ is equivalent to $$\underline{\hat{y_i}}$$

so we can write

$$
\begin{array}{l}
    \mathbb{R}^{\text{N}} \rightarrow \mathbb{R}^{\text{L}} \\
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
    \mathbb{R}^{\text{N}} \equiv \mathbb{R}^{\text{H}_0} \rightarrow \mathbb{R}^{\text{H}_1} \\
    \underline{x} \rightarrow f_{\Theta_1}(\underline{x}) = \underline{a_{1}}
\end{array}
$$

And the last one

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{p-1}} \rightarrow \mathbb{R}^{\text{H}_p} \equiv \mathbb{R}^{\text{L}} \\
    \underline{a_{\text{P}-1}} \rightarrow f_{\Theta_{\text{P}}}(\underline{a_{\text{P}-1}}) 
        =\underline{a_{\text{P}}}
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
    f_{\Theta_{P}}(\dots (f_{\Theta_{\ell}}(\dots (f_{\Theta_2}(f_{\Theta_1}(\underline{x}))) \dots)) \dots)
$$

Composing functions is like chaining them. The output of the inner function becomes the input of the outer function. Given two functions $$f$$ and $$g$$, the composite function $$h$$ resulting is $$h(x) = g(f(x))$$ which is also noted as $$h(x) = (g \circ f) (x)$$. So the later wordy equation can be written in an easier way

$$
    F(\underline{x})
    =
    (f_{\Theta_{P}} \circ \cdots \circ f_{\Theta_{\ell}} \circ \cdots f_{\Theta_1})(\underline{x})
$$

> :alien: *alien says* :speech_balloon:\
$$\text{P}$$ the number of layers of the neural network\
$$F$$ is the composite transfer fonction. There is one transfer fonction $$f_{\Theta_k}$$ per layer, each one enables passing from layer $$\ell$$-1 to layer $$\ell$$, with $$\ell \in [1..\text{P}]$$

What we've just finished to define is the feedforward propagation. As we saw this algorithm passes the inputs from one layer to the other thanks to the transfer fonctions of each layer of the neural network.

### $$\mathcal{L}$$ *as* Loss function and $$E$$ *as* Error
$$\mathcal{L}$$ is a function of output $$y$$ and of the predicted output $$\hat{y}$$. It represents a kind of difference between the expected and the actual output. There are many ways to define a loss function. It can be the mean squared error, so for the i$$^{\text{th}}$$ sample of the dataset we have

$$
\forall{i} \in [1..\text{M}],
\mathcal{L}_{\text{MSE}}(\underline{y_i}, \hat{\underline{y_i}})=
{1 \over \text{L}} \sum_{j=1}^{\text{L}}(\hat{y_{ij}} - y_{ij})^2
$$

We define the error as the sum of the losses over the dataset

$$
E = {1 \over \text{M}} \sum_{i=1}^{\text{M}} \mathcal{L}(\underline{y_i}, \hat{\underline{y_i}})
$$

Error follows by summing through the samples of the dataset

$$
E_{\text{MSE}} = {1 \over \text{M}} \sum_{i=1}^{\text{M}} {1 \over \text{L}} \sum_{j=1}^{\text{L}}(\hat{y_{ij}} - y_{ij})^2
$$

However, as the MSE function (mean squared error) is not convex for neural networks, we usually prefer to use another loss function called **cross-entropy** (comes from the *maximum likelihood principle*)
Where does it comes from ?

$$
\mathcal{L}_{\text{CE}}(\underline{y_i}, \hat{\underline{y_i}})=
-\sum_{j=1}^{\text{L}} {y_{ij}} \log(\hat{y_{ij}}) + (1 - y_{ij}) \log(1 - \hat{y_{ij}})
$$

And the error follows

$$
E_{\text{CE}}=
-{1 \over \text{M}} \sum_{i=1}^{\text{M}} \sum_{j=1}^{\text{L}} {y_{ij}} \log(\hat{y_{ij}}) + (1 - y_{ij}) \log(1 - \hat{y_{ij}})
$$

> :alien: *alien says* :speech_balloon:\
$$y_{ij}$$ is the j$$^{\text{th}}$$ component of the target variable in the i$$^{\text{th}}$$ sample.\
$$\hat{y_{ij}}$$ is the j$$^{\text{th}}$$ component of the predicted target variable (estimated by the neural network) in the i$$^{\text{th}}$$ sample.

### A *as* activation
Before defining the gradient descent algorithm, we present a more detailed version of the transfer function to emphasize the role of the parameters $$\Theta$$.

Remember that for the $$\ell^{\text{th}}$$ layer we have
$$
f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = \underline{a_{\ell}}
$$
The transfer fonction is composed of a matrix multiplication of $$\underline{a_{\ell-1}}$$ with  $$\underline{\underline{\Theta_{\ell}}}$$, respectivly the input and the weigts of the $$\ell^{\text{th}}$$ layer, and the activation function $$g_{\ell} : \mathbb{R^{\text{H}_{\ell}}} \rightarrow \mathbb{R^{\text{H}_{\ell}}} $$

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
\forall i \in [1..\text{H}_k],
\forall j \in [1..\text{H}_\textit{k-1}],
\forall k \in [1..\text{P}]:
\underline{\underline{\Theta_k}} = 
\begin{pmatrix}
    \theta_{k,11} & \dots  & \theta_{k,1j}  & \dots  & \theta_{k,1\text{H}_\textit{k-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{k,i1} & \dots  & \theta_{k,ij}  & \dots  & \theta_{k,i\text{H}_\textit{k-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{k,\text{H}_k1} & \dots  & \theta_{k,\text{H}_kj}  & \dots  & \theta_{k,{\text{H}_k}\text{H}_\textit{k-1}}
\end{pmatrix}
,
\underline{\underline{\Theta_k}} \in \mathbb{R}^{\text{H}_k \times \text{H}_\textit{k-1}}
$$

And the last one

$$
\begin{array}{l}
    \mathbb{R}^{\text{H}_{\text{P}-1}} \rightarrow \mathbb{R}^{\text{L}} \\
    \underline{a_{\text{P}-1}} \rightarrow
    g_{\text{P}}(\underline{\underline{\Theta_{\text{P}}}} \underline{a_{\text{P}-1}})
\end{array}
$$

$$
\forall i \in [1..\text{L}], \forall j \in [1..\text{H}_\textit{P-1}]:
\underline{\underline{\Theta_{\text{P}}}} = 
\begin{pmatrix}
    \theta_{\text{P},11} & \dots  & \theta_{\text{P},1j}  & \dots  & \theta_{\text{P},1\text{H}_\textit{P-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\text{P},i1} & \dots  & \theta_{\text{P},ij}  & \dots  & \theta_{\text{P},i\text{H}_\textit{P-1}} \\
    \vdots & \ddots &  \vdots & \ddots & \vdots \\
    \theta_{\text{P},\text{L}1} & \dots  & \theta_{\text{P},\text{L}j}  & \dots  & \theta_{\text{P},{\text{L}}\text{H}_\textit{P-1}}
\end{pmatrix}
,
\underline{\underline{\Theta_{\text{P}}}} \in \mathbb{R}^{\text{L} \times \text{H}_\textit{P-1}}
$$

> :alien: *alien says* :speech_balloon:\
$$g_k$$ is the activation function of the k$$^{\text{th}}$$ hidden layer.\
$$g_{\text{P}}$$ is the output function of the last layer.

Let's define another usefull notation

$$
\begin{array}{l}
    \underline{z_k} = \underline{\underline{\Theta_k}} \underline{a_{k-1}}\\
    \underline{a_k} = f_{\Theta_k}(\underline{a_{k-1}}) = g_k(\underline{z_k})
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
    \underline{z_{\text{P}}} = \underline{\underline{\Theta_{\text{P}}}} \underline{a_{\text{P-1}}}\\
    \underline{a_{\text{P}}} = f_{\Theta_{\text{P}}}(\underline{a_{\text{P-1}}}) = g_{\text{P}}(\underline{z_{\text{P}}})
\end{array}
$$

In many neural network and in this article we will take the sigmoid function for the activation and the softmax for the output.


### Gradient Descent *as* the Optimization Algorithm
Up to now, we defined the datasets composed of $$\underline{\underline{X}}$$ and $$\underline{\underline{Y}}$$, the forward propagation algorithm which calculates an estimation of the target $$\underline{\underline{\hat{Y}}}$$, and an error function $$E_{\text{CE}}$$. Recall that we want $$\underline{\underline{\hat{Y}}}$$ to be as close as possible to $${\underline{\underline{Y}}}$$. The error function $$E_{\text{CE}}$$ comes in handy by giving us a metric to quantify as far are we from the ground truth $${\underline{\underline{Y}}}$$. And so, "as close as possible" can be mathematically translated into "minimizing the error function $$E_{\text{CE}}$$"

$$
\arg \min_{\Theta} E_{\text{CE}}(\Theta)
$$

How do we minimize a fonction ? Well... with an optimization algorithm !
In this post we will use the gradient descent. Gradient descent belong to the family of the `first-order optimization algorithms`. It enables us to minimize the error $$E_{\text{CE}}$$ using the gradient with respect to the weights $$\Theta$$, so at t$$^{\text{th}}$$ iteration

$$
\underline{\underline{\Theta_k}}^{(t+1)} = \underline{\underline{\Theta_k}}^{(t)} - \alpha{\partial E(\Theta_k^{(t)}) \over \partial\Theta_k}
$$

> :alien: *alien says* :speech_balloon:\
$$\alpha$$ is the learning rate of the gradient descent algorithm\
$$E_{\text{CE}}$$ is a non linear error function. It depends on $$\underline{\underline{X}}$$ and $$\Theta_k$$, and it must defined and differentiable in the neighborhood of $$\Theta_k^{(t)}$$\
$${\partial E(\Theta_k^{(t)}) \over \partial\Theta_k}$$ is the partial derivative of $$E_{\text{CE}}$$ according to $$\Theta_k$$ the tuning parameter

### J *as* Jacobian
First things first, the last layer of the network:

$$
    \forall i \in [1..\text{L}] \text{ and }
    \forall j \in [1..\text{H}_{\text{P-1}}]
$$

$$
    {\partial E_{\text{CE}} \over \partial\Theta_{\text{P},ij}}
    =
    -{1 \over \text{M}} \sum_{k=1}^{\text{M}} \sum_{r=1}^{\text{L}} {y_{kr}} \log(\hat{y_{kr}}) + (1 - y_{kr}) \log(1 - \hat{y_{kr}}))
$$

Before derivating the error let's simplify using the chain rule of derivation along with $$\underline{a_{\text{P}}}$$ and $$\underline{z_{\text{P}}}$$, because $$E_{\text{CE}}$$ depends on $$a_{\text{P}}$$ which is equals to $$\hat{y}$$, and $$a_{\text{P}}$$ depends on $$z_{\text{P}}$$

$$
    {\partial E_{\text{CE}} \over \partial\Theta_{\text{P},ij}}
    =
    {\partial E_{\text{CE}} \over \partial a_{\text{P}}}
    {\partial \underline{a_{\text{P}}} \over \partial z_{\text{P}}}
    {\partial \underline{z_{\text{P}}} \over \partial \Theta_{\text{P},ij}}
$$

First

$$
{\partial E_{\text{CE}} \over \partial a_{\text{P}}}=
{\partial \over \partial a_{\text{P}}}
\Big (
    -{1 \over \text{M}} \sum_{i=1}^{\text{M}} \sum_{j=1}^{\text{L}} {y_{ij}} \log(\hat{y_{ij}}) + (1 - y_{ij}) \log(1 - \hat{y_{ij}})
\Big )
$$

Then

$$
{\partial a_{\text{P}} \over \partial z_{\text{P}}}=
{\partial \over \partial z_{\text{P}}}
\Big (g_{\text{P}}(\underline{z_{\text{P}}}) \Big )=
g_{\text{P}}(\underline{z_{\text{P}}}) \odot (1 - g_{\text{P}}(\underline{z_{\text{P}}}))=
\underline{a_{\text{P}}} \odot (1 - \underline{a_{\text{P}}})
$$

Finally

$$
{\partial z_{\text{P}} \over \partial\Theta_{\text{P},ij}}=
{\partial \over \partial\Theta_{\text{P},ij}}
\Big (
    \underline{\underline{\Theta_{\text{P},ij}}} \underline{a_{\text{P}-1}}
\Big )=
\underline{a_{\text{P}-1}}
$$

Recursively we can write for all layers

$$
    \forall i \in [1..\text{L}],
    \forall j \in [1..\text{H}_{\text{P-1}}],
    \forall l \in [1..\text{P}]
$$

$$
    {\partial E_{\text{CE}} \over \partial\Theta_{l, ij}}
    =
    -{1 \over \text{M}} \sum_{k=1}^{\text{M}} \sum_{r=1}^{\text{L}} {y_{kr}} \log(\hat{y_{kr}}) + (1 - y_{kr}) \log(1 - \hat{y_{kr}}))
$$

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