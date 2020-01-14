---
layout: post
title:  "The (detailled) mathematics behind neural networks"
date:   2019-12-17
categories: [gradient descent, neural network]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

This post will guide you trough the main theory behind one basic  neural network. Starting from scratch, I hope it will show you how things linked together.

This article doesn't go through the details of how to implement a neural network.

Even though this article describes all the necessary details to understand a basic neural network (except the bias unit), the main part is taken by backpropagation and gradient descent as it is the most mathematic demanding.

In the first part, we'll define all the notation usefull such as the dataset, the neural network function. This notations will be used in the later demonstration. It may sometimes sound verbose but I prefer to make the notations explicit so that more readers will be able to read this article.

> **Presentation**<br>
> During the lecture of this post, mister *legal* alien :alien: will take a look at :telescope: the demonstration and make comments when he wants to.

## Notations
### $$X$$ *as* the sample
The dataset X is represented as $$\underline{\underline{X}}$$, an $$\text{M}$$-by-$$\text{N}$$ matrix, i.e. $$\underline{\underline{X}} \in \mathbb{R}^{\text{M} \times \text{N}}$$

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

> :alien: *alien says* :speech_balloon:<br>
> M the number of samples we have<br>
> N the dimension of the input, or the number of explanatory variables<br>
> **Vectors** are underlined like this: $$\underline{x}$$<br>
> **Matrices** are double-underlined like this: $$\underline{\underline{X}}$$

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

> :alien: *alien says* :speech_balloon:<br>
> M the number of samples we have<br>
> P the dimension of the target

### $$\mathcal{D}$$ as dataset
The dataset is just the combinations of X and Y, that is

$$
\mathcal{D} = \big \{(\underline{x_i}, \underline{y_i}) \big \}, \forall i \in [1..\text{M}]
$$

### $$NN$$ *as* Neural Network
A dense neural network can be defined by a composite transfer fonction $$F$$ composed of functions $$f_{\Theta_{\ell}}$$, $$ \forall \ell \in [1..\text{L}]: \big [ f_{\Theta_1}, ... , f_{\Theta_{\ell}}, ...,f_{\Theta_{\text{L}}} \big ] $$ and input parameters $$\underline{x_i}$$:

$$
\begin{align}
    \mathbb{R}^{\text{N}} &\rightarrow \mathbb{R}^{\text{P}} \\
    \underline{x_i} &\rightarrow F(\underline{x_i}) = \underline{\hat{y_i}}
\end{align}
$$

:triangular_flag_on_post: *to simplify* :triangular_flag_on_post:
> $$\underline{x_i}$$ is written $$\underline{x}$$<br>
> $$\underline{\hat{y_i}}$$ is written $$\underline{\hat{y}}$$

so we can write

$$
\begin{align}
    \mathbb{R}^{\text{N}} &\rightarrow \mathbb{R}^{\text{P}} \\
    \underline{x} &\rightarrow F(\underline{x}) = \underline{\hat{y}}
\end{align}
$$

:telescope: *with* :telescope:
> $$\underline{\hat{y}}$$ the calculated output of the neural network. The objective is to have  $$\underline{\hat{y}}$$ as close as possible to $$\underline{y}$$.<br>
> $$F$$ the composite transfer fonction.

## Feed-forward propagation
The neural network studied here is "Feed-Forward", so that one layer is fully connected to the next layer.

Starting from input $$\underline{x}$$, the first layer

$$
\begin{align}
    \mathbb{R}^{\text{N}} = \mathbb{R}^{\text{H}_0} &\rightarrow \mathbb{R}^{\text{H}_1} \\
    \underline{x} &\rightarrow f_{\Theta_1}(\underline{x}) = \underline{a_{1}}
\end{align}
$$

Then going from the $${\ell\text{-}1}^{\text{th}}$$ to the $$\ell^{\text{th}}$$ layer 

$$
\begin{align}
    \mathbb{R}^{\text{H}_{\ell-1}} &\rightarrow \mathbb{R}^{\text{H}_\ell} \\
    \underline{a_{\ell-1}} &\rightarrow f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = \underline{a_{\ell}}
\end{align}
$$

And for the last one, to the output $$\underline{\hat{y}}$$

$$
\begin{align}
    \mathbb{R}^{\text{H}_{\text{L}-1}} &\rightarrow \mathbb{R}^{\text{H}_{\text{L}}} = \mathbb{R}^{\text{P}} \\
    \underline{a_{\text{L}-1}} &\rightarrow f_{\Theta_{\text{L}}}(\underline{a_{\text{L}-1}}) 
        =\underline{a_{\text{L}}}
        =\underline{\hat{y}}
\end{align}
$$

:telescope: *with* :telescope:
> $$\text{H}_{\ell-1}$$ the size of the $$\ell-1^{\text{th}}$$ hidden layer<br>
> $$\text{H}_{\ell}$$ the size of the $$\ell^{\text{th}}$$ hidden layer<br>
> $$f_{\Theta_{\ell}}$$ the transfer fonction for the $$\ell^{\text{th}}$$ layer<br>
> $$\underline{a_{\ell-1}}$$ the input of the $$\ell^{\text{th}}$$ layer<br>
> $$\underline{a_{\ell}}$$ the output of the $$\ell^{\text{th}}$$ layer

We'll dig into the details of the transfer fonction a little bit more later. For now, we try to consider the neural network at a granular point of view. So, given the transfer fonction $$f_{\Theta_{\ell}}$$, the whole network should look like the composition of the $$\text{L}$$ transfer fonctions listed in $$F$$

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

> :alien: *alien says* :speech_balloon:<br>
> $$\text{L}$$ the number of layers of the neural network<br>
> $$F$$ is the composite transfer fonction. There is one transfer fonction $$f_{\Theta_{\ell}}$$ per layer, each one enables passing from layer $$\ell$$-1 to layer $$\ell$$, with $$\ell \in [1..\text{L}]$$

What we've just finished to define is the feedforward propagation. As we saw, this algorithm passes the inputs from one layer to the other thanks to the transfer fonctions of each layer of the neural network.

### $$\mathcal{L}$$ *as* Loss function and $$E$$ *as* Error

An important aspect of the design of a deep neural networks is the choice of the cost function.

The loss $$\mathcal{L}$$ is a function of the ground truth $$\underline{y_i}$$ and of the predicted output $$\underline{\hat{y_i}}$$. It represents a kind of difference between the expected and the actual output. There are many ways to define a loss function. Historically for neural networks the loss function used was the mean squared error (MSE). For the $$i^{\text{th}}$$ sample of the dataset we have

$$
\forall{i} \in [1..\text{M}],
\mathcal{L}_{\text{MSE}}(\underline{y_i}, \hat{\underline{y_i}})=
{1 \over \text{P}} \sum_{j=1}^{\text{P}}(\hat{y_{ij}} - y_{ij})^2
$$

However, the MSE function (mean squared error) is not convex for neural networks.

In this case, our parametric model defines a distribution
$$p(\underline{y_i}|\underline{x_i};\Theta)$$
and we use the principle of maximum likelihood. This means we use the **Negative Log Likelihood** (LLE) cost function. As we'd like to predict the probability of a sample to be in one of P classes, given a multinomial distribution, it's better to specificy we are using the **Multiclass Cross-Entropy** Loss between the training data and the model distribution/model's prediction.

$$
\boxed{
    \mathcal{L}_{\text{CE}}(\underline{y_i}, \hat{\underline{y_i}})=
    -\sum_{j=1}^{\text{P}} {y_{ij}} \log(\hat{y_{ij}})
}
$$

We define the overall error as the sum of the losses over the dataset

$$
E = {1 \over \text{M}} \sum_{i=1}^{\text{M}} \mathcal{L}(\underline{y_i}, \hat{\underline{y_i}})
$$

> The loss and the error might not be the definitions frequently used, so be carefull with that.

Error follows by summing through the samples of the dataset

$$
\boxed{
    E_{\text{CE}}=
    -{1 \over \text{M}} \sum_{i=1}^{\text{M}} \sum_{j=1}^{\text{P}} {y_{ij}} \log(\hat{y_{ij}}))
}
$$

> :alien: *alien says* :speech_balloon:<br>
> $$y_{ij}$$ is the $$j^{\text{th}}$$ component of the target variable in the $$i^{\text{th}}$$ sample.<br>
> $$\hat{y_{ij}}$$ is the $$j^{\text{th}}$$ component of the predicted target variable (estimated by the neural network) in the $$i^{\text{th}}$$ sample.

### A *as* activation
Before defining the gradient descent algorithm, we need to define a more detailed version of the transfer function to show the role of the parameter $$\Theta$$ and also define the link functions for the hidden layers (activation functions) and for the last layer (output function).

Remember that to pass from the $$\ell\text{-}1^{\text{th}}$$ to the $$\ell^{\text{th}}$$ layer we have

$$
\begin{align}
    \mathbb{R}^{\text{H}_{ \ell - 1}} &\rightarrow \mathbb{R}^{\text{H}_{ \ell }} \\
    a_{\ell-1} &\rightarrow f_{\Theta_{\ell}}(\underline{a_{\ell-1}}) = \underline{a_{\ell}}
\end{align}
$$

The transfer fonction is composed of:
* a matrix multiplication such as
$$
\underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}} = \underline{z_{\ell}}
$$
, respectivly the weigts with the input of the $$\ell^{\text{th}}$$ layer, given $$\underline{z_{\ell}} \in \mathbb{R}^{\text{H}_{ \ell }}$$
* an activation function $$g_{\ell}: \mathbb{R}^{\text{H}_{ \ell }} \rightarrow \mathbb{R}^{\text{H}_{ \ell }}$$ such as

$$
\begin{align}
    \mathbb{R}^{\text{H}_{ \ell }} &\rightarrow \mathbb{R}^{\text{H}_{ \ell }} \\
    \underline{z_{\ell}} &\rightarrow g_{\ell}(\underline{z_{\ell}})
    =
    g_{\ell}(\underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}})
    =
    f_{\Theta_{\ell}}(\underline{a_{\ell-1}})
    =
    \underline{a_{\ell}}
\end{align}
$$

From the $$\ell\text{-}1^{\text{th}}$$ to the $$\ell^{\text{th}}$$ layer, given 
$$
\underline{\underline{\Theta_{\ell}}} \in \mathbb{R}^{\text{H}_{\ell} \times \text{H}_{\ell-1}}
$$ we can write

$$
\begin{align}
    \mathbb{R}^{\text{H}_{\ell-1}} &\rightarrow \mathbb{R}^{\text{H}_{\ell}} \\
    \underline{a_{\ell-1}} &\rightarrow 
    g_{\ell}(\underline{\underline{\Theta_{\ell}}} \underline{a_{\ell-1}})
\end{align}
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
$$

For example, the **first layer** can be written like this, given 
$$
\underline{\underline{\Theta_1}} \in \mathbb{R}^{\text{H}_1 \times \text{N}}
$$

$$
\begin{align}
    \mathbb{R}^{\text{N}} &\rightarrow \mathbb{R}^{\text{H}_1} \\
    \underline{x} &\rightarrow 
    f_{\Theta_1}(\underline{x})
    =
    g_1(\underline{\underline{\Theta_1}} \underline{x})
    =
    g_1(\underline{z_1})
    =
    \underline{a_1}
\end{align}
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
$$

And for the **last layer**, given 
$$
\underline{\underline{\Theta_{\text{L}}}} \in \mathbb{R}^{\text{P} \times H_{\text{L}-1}}
$$

$$
\begin{align}
    \mathbb{R}^{\text{H}_{\text{L}-1}} &\rightarrow \mathbb{R}^{\text{P}} \\
    \underline{a_{\text{L}-1}} &\rightarrow
    f_{\Theta_{\text{L}}}(\underline{a_{\text{L-1}}})
    =
    g_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}} \underline{a_{\text{L}-1}})
    =
    g_{\text{L}}(\underline{z_{\text{L}}})
    =
    \underline{a_{\text{L}}}
\end{align}
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
$$

> :alien: *alien says* :speech_balloon:<br>
> $$g_{\ell}$$ is the activation function of the $$\ell^{\text{th}}$$ hidden layer.<br>
> $$g_{\text{L}}$$ is the output function of the last layer.

In this article we take the softmax function for the last layer and the sigmoid function for all hidden layers. In this particular case when the last layer has a softmax function, the second last layer has the same number of nodes then the last layer. It's because the softmax function give us a way to transform the output of the second last layer into probabilities of class appearence, with a sum of 1.

* the sigmoid function
$$
g_{\ell}(z_{\ell, i}) = \frac{1}{1+e^{-z_{\ell, i}}}
$$

* and the softmax function
$$
g_{\text{L}}(z_{\text{L}, i}) = \frac{\mathrm{e}^{z_{\text{L},i}}}{\sum_{k=1}^{\text{L}} \mathrm{e}^{z_{\text{L},k}}}
$$

## Backpropagation and gradient descent
### Picking gradient descent as the optimization algorithm
Backpropagation, for "backward propagationn of errors" is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weight.
It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.
Backpropagation was first invented in the 1970s as a general optimization method for performing automatic differentiation of complex nested functions.

Up to now, we defined the datasets composed of $$\underline{\underline{X}}$$ and $$\underline{\underline{Y}}$$, the forward propagation algorithm which calculates an estimation of the target $$\underline{\underline{\hat{Y}}}$$, and an error function $$E_{\text{CE}}$$. Recall that we want $$\underline{\underline{\hat{Y}}}$$ to be as close as possible to $${\underline{\underline{Y}}}$$. The error function $$E_{\text{CE}}$$ comes in handy by giving us a metric to quantify as far are we from the ground truth $${\underline{\underline{Y}}}$$. And so, "as close as possible" can be mathematically translated into "minimizing the error function $$E_{\text{CE}}$$"

$$
\arg \min_{\Theta} E_{\text{CE}}(\Theta)
$$

How do we minimize a fonction ? Well... with an optimization algorithm !
In this post we will use the gradient descent. Gradient descent belong to the family of the `first-order optimization algorithms`. It enables us to minimize the error $$E_{\text{CE}}$$ using the gradient with respect to the weights $$\Theta$$, so at t$$^{\text{th}}$$ iteration

$$
\begin{equation}
\boxed{
    \forall \ell \in [1..\text{L}],
    \underline{\underline{\Theta_{\ell}}}^{(t+1)} = \underline{\underline{\Theta_{\ell}}}^{(t)} - \alpha{\partial E(\Theta_{\ell}^{(t)}) \over \partial\Theta_{\ell}}
}
\end{equation}
$$

Gradient is obtained by using the back propagation algorithm, indeed back propagation is a differentiation algorithm.

> :alien: *alien says* :speech_balloon:<br>
> $$\alpha$$ is the learning rate of the gradient descent algorithm<br>
> $$E_{\text{CE}}$$ is a non linear error function. It depends on $$\underline{\underline{X}}$$ and $$\Theta_{\ell}$$, and it must defined and differentiable in the neighborhood of $$\Theta_{\ell}^{(t)}$$<br>
> $${\partial E(\Theta_{\ell}^{(t)}) \over \partial\Theta_{\ell}}$$ is the partial derivative of $$E_{\text{CE}}$$ according to $$\Theta_{\ell}$$ the tuning parameter

### Breaking the last layer
First things first, the last layer of the network:

$$
\begin{align}
    &\forall i \in [1..\text{P}] \text{, }
    \forall j \in [1..\text{H}_{\text{L-1}}] \text{ and }
    \forall k \in [1..\text{M}]\\
    &{\partial E_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial \over \partial\Theta_{\text{L},ij}}
    \Big (
    -{1 \over \text{M}} \sum_{k=1}^{\text{M}} \sum_{i=1}^{\text{P}} {y_{ki}} \log(\hat{y_{ki}})
    \Big )
\end{align}
$$

Before derivating, let's simplify using the chain rule of derivation along with $$\underline{a_{\text{L}}}$$ and $$\underline{z_{\text{L}}}$$, because $$E_{\text{CE}}$$ depends on $$a_{\text{L}}$$ which is equals to $$\hat{y}$$, and $$a_{\text{L}}$$ depends on $$z_{\text{L}}$$. It can be written as

$$
\begin{align}
    E_{CE}(\underline{a_{\text{L}}}) &= \dots \\
    g_{\text{L}}(\underline{z_{\text{L}}}) &= \underline{a_{\text{L}}}\\
    h_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}}) &= \underline{z_{\text{L}}}
\end{align}
$$

We can write the error for the $$t^{\text{th}}$$ iteration

$$
E_{\text{CE}}(g_{\text{L}}(h_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}}^{(t)})))
=
E_{\text{CE}} \circ g_{\text{L}} \circ h_{\text{L}} (\underline{\underline{\Theta_{\text{L}}}}^{(t)})
$$

Applying the chain rule of derivation gives

$$
\begin{equation}
    { \partial 
        (
            E_{CE} \circ g_{\text{L}} \circ h_{\text{L}}
        )
      \over
      \partial \underline{\underline{\Theta_{\text{L}}}}
    }
    \big (
        \underline{\underline{\Theta_{\text{L}}}}^{(t)}
    \big )
    =
    { 
        \partial E_{CE}
        \over
        \partial \underline{a_{\text{L}}}
    }
    \big (
        g_{\text{L}}(h_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}}^{(t)}))
    \big )
    { 
        \partial g_{\text{L}}
        \over
        \partial \underline{z_{\text{L}}}
    }
    \big (
        h_{\text{L}}(\underline{\underline{\Theta_{\text{L}}}}^{(t)})
    \big )
    { 
        \partial h_{\text{L}}
        \over
        \partial \underline{\underline{\Theta_{\text{L}}}}
    }
    \big (
        \underline{\underline{\Theta_{\text{L}}}}^{(t)}
    \big )
\end{equation}
$$

Which is commonly written more or less is the following form

$$
    {\partial E_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    {\partial E_{\text{CE}} \over \partial \underline{a_{\text{L}}}}
    {\partial \underline{a_{\text{L}}} \over \partial \underline{z_{\text{L}}}}
    {\partial \underline{z_{\text{L}}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
$$

We continue simplifying by taking only one sample so the previous equation becomes

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    {\partial \mathcal{L}_{\text{CE}} \over \partial \underline{a_{\text{L}}}}
    {\partial \underline{a_{\text{L}}} \over \partial \underline{z_{\text{L}}}}
    {\partial \underline{z_{\text{L}}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
$$

To make the calculation of the derivative even simpler we pick scalars instead of vectors so that

$$
\begin{equation}
\boxed{
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    {\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}
    {\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
    {\partial z_{\text{L},i} \over \partial \Theta_{\text{L},ij}}
}
\end{equation}
$$

with
$$
\underline{z_{\text{L}}}, \underline{a_{\text{L}}} \in \mathbb{R}^{\text{P}}
$$
and
$$
\underline{\underline{\Theta_{\text{L}}}} \in \mathbb{R}^{\text{P} \times H_{\text{L}-1}}
$$

Let's take the **first** component of the previous equation

$$
\begin{align*}
{\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}
&=
{\partial \over \partial a_{\text{L},i}}
\Big (
    - \sum_{k=1}^{\text{P}} {y_{k}} \log(\hat{y_{k}})
\Big )\\
&=
{\partial \over \partial a_{\text{L},i}}
\Big (
    - \sum_{k=1}^{\text{P}} {y_{k}} \log(a_{\text{L},k})
\Big )
\end{align*}
$$

we have
$$
y_k =
\begin{cases}
    0 & \text{if \(k \neq i\) } \\
    1 & \text{if \(k = i\) }
\end{cases}
$$
therefore when $$k = i$$, $$y_k = 1$$

$$
{\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}=
-
{\partial \over \partial a_{\text{L},i}}
\Big (
    \log(a_{\text{L},i})
\Big )
$$

So

$$
\boxed{
    {\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}=
    - \frac{1}{a_{\text{L},i}}
}
$$

> :alien: *alien says* :speech_balloon:<br>
> Having said that we omit a constant during the logarithm derivative calculation

Then we take the **second** component

$$
\begin{align}
{\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
&=
{\partial \over \partial z_{\text{L},i}}
\Big (
    g_{\text{L}}(z_{\text{L},i})
\Big )\\
&=
{\partial \over \partial z_{\text{L},i}}
\Big (
    \frac{\mathrm{e}^{z_{\text{L},i}}}{\sum_{k=1}^{\text{L}} \mathrm{e}^{z_{\text{L},k}}}
\Big )\\
&=
\frac{ \mathrm{e}^{z_{L},i} \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},k} - \big ( \mathrm{e}^{z_{L},i} \big ) ^2}
{ \big ( \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},k} \big ) ^ 2} \\
&=
\frac{ \mathrm{e}^{z_{L},i} }
{ \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},i} }
-
\big (
\frac{ \mathrm{e}^{z_{L},i} }
{ \sum_{k=1}^{\text{P}} \mathrm{e}^{z_{L},i} }
\big ) ^2 \\
&=
g_{\text{L}}(z_{\text{L},i}) - g^2_{\text{L}}(z_{\text{L},i})
\end{align}
$$

so that

$$
\begin{equation}
\boxed{
    {\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
    =
    a_{\text{L},i}(1 - a_{\text{L},i})
}
\end{equation}
$$

And the for the **third** component

$$
\begin{align}
{\partial z_{\text{L},i} \over \partial\Theta_{\text{L},ij}}
&=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \underline{\underline{\Theta_{\text{L}}}} \underline{a_{\text{L}-1}}
\Big )_i\\
&=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \sum_{k=1}^{\text{H}_{\text{L}-1}} \theta_{\text{L},ik} a_{\text{L}-1,k}
\Big )\\
&=
{\partial \over \partial\Theta_{\text{L},ij}}
\Big (
    \theta_{\text{L},i1} a_{\text{L}-1,1} + \cdots + \theta_{\text{L},ik} a_{\text{L}-1,k} + \cdots + \theta_{\text{L},iH_{\text{L}-1}} a_{\text{L}-1,H_{\text{L}-1}}
\Big )
\end{align}
$$

The derivative is equal to zero when $$j \neq k$$ so for $$j = k$$

$$
\boxed{
    {\partial z_{\text{L},i} \over \partial\Theta_{\text{L},ij}}
    =
    a_{\text{L}-1,j}
}
$$

Then putting it together with the chained derivation

$$
\begin{align}
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    &=
    \underbrace{
        {\partial \mathcal{L}_{\text{CE}} \over \partial a_{\text{L},i}}
        {\partial a_{\text{L},i} \over \partial z_{\text{L},i}}
    }_{\delta_{\text{L},i}}
    {\partial z_{\text{L},i} \over \partial \Theta_{\text{L},ij}} \\
    &=
    - \frac{1}{a_{\text{L},i}} a_{\text{L},i} (1 - a_{\text{L},i}) a_{\text{L}-1,j}
\end{align}
$$

$$
\begin{equation}
\boxed{
    {\partial \mathcal{L}_{\text{CE}} \over \partial\Theta_{\text{L},ij}}
    =
    (a_{\text{L},i} - 1) a_{\text{L}-1,j}
}
\end{equation}
$$

As $$\underline{a_{\text{L}}}$$ and $$\underline{a_{\text{L}-1}}$$ are row vectors, the previous equation can be vectorized

$$
    {\partial \mathcal{L}_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    (\underline{a_{\text{L}}} - 1)^{\text{T}} \underline{a_{\text{L}-1}}
$$

.. and summed up over the M training examples

$$
    {\partial E_{\text{CE}} \over \partial \underline{\underline{\Theta_{\text{L}}}}}
    =
    \frac{1}{M} \sum_{k=1}^{M} (\underline{a_{\text{L},k}} - 1)^{\text{T}} \underline{a_{\text{L}-1,k}}
$$

### Breaking the hidden layers
We've just finished breaking the last layer, then we can get the derivative of the penultimate one and recursivly of all the hidden layers until we reach the inputs.

Given $$i \in [1..\text{H}_{\ell}]$$, $$j \in [1..\text{H}_{\ell-1}]$$ and $$\underline{\underline{\Theta_{\ell}}} \in \mathbb{R}^{\text{H}_{\ell} \times \text{H}_{\ell - 1} }$$

$$
\begin{equation}
\boxed{
    {\partial \mathcal{L}_{\text{CE}} 
        \over 
    \partial\Theta_{\ell,ij}}
    =
    {\partial \mathcal{L}_{CE}
        \over 
    \partial a_{\ell,i}}
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
    {\partial z_{\ell,i}
        \over 
    \partial\Theta_{\ell,ij}}
}
\end{equation}
$$

Let's take the **first** component and apply the chain rule of derivation.

Given $$ z_{\text{L}+1}, a_{\ell+1}  \in \mathbb{R}^{H_{\ell+1}} $$ and $$ z_{\text{L}}, a_{\ell}  \in \mathbb{R}^{H_{\ell}} $$ and $$ \delta_{\ell} \in \mathbb{R}^{H_{\ell}}$$

$$
\begin{equation}
\boxed{
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
}
\end{equation}
$$

We define

$$
\boxed{
    \delta_{\ell,i}
    =
    {\partial \mathcal{L}_{CE}
        \over 
    \partial a_{\ell,i}}
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
}
$$

With $$ \underline{\delta_{\ell + 1}} \in \mathbb{R}^{H_{\ell+1}} $$, the **first two** components are equals to

$$
\boxed{
    \Big (
    {\partial \mathcal{L}_{CE}
        \over 
    \partial a_{\ell,i}}
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
    \Big )
    =
    \delta_{\ell+1,k}
}
$$

As we go backward, $$ \underline{\delta_{\ell+1}} $$ has already been calculated, so we have:

* For
$$\ell = \text{L}: \boxed{\delta_{\text{L},i} = a_{\text{L},i} - 1}$$

* For $$ \ell \neq \text{L}:$$
$$
\begin{equation}
    \delta_{\ell,i}
    =
    -\frac{1}{a_{\ell,i}}
    { 
        \partial a_{\ell,i}
        \over
        \partial z_{\ell, i}
    }
    =
    -\frac{1}{a_{\text{L},i}}
    { 
        \partial
        \over
        \partial z_{\ell, i}
    }
    \big (
        g_{\ell}(z_{\ell,i})
    \big )
\end{equation}
$$

For the hidden layers, we took the sigmoid function as the activation function $$g_{\ell}$$

$$
\begin{align}
    { 
        \partial
        \over
        \partial z_{\ell, i}
    }
    \big (
        g_{\ell}(z_{\ell,i})
    \big )
    &=
    { 
        \partial
        \over
        \partial z_{\ell, i}
    }
    \Big (
        \frac{1}{1+e^{-z_{\ell, i}}}
    \Big )
    \\&=
    \frac{e^{-z_{\ell, i}}}{(1+e^{-z_{\ell, i}})^2}
    \\&=
    \frac{1}{1+e^{-z_{\ell, i}}}\frac{-1+1+e^{-z_{\ell, i}}}{1+e^{-z_{\ell, i}}}
    \\&=
    \frac{1}{1+e^{-z_{\ell, i}}}(1-\frac{1}{1+e^{-z_{\ell, i}}})
\end{align}
$$

So

$$
{ 
    \partial
    \over
    \partial z_{\ell, i}
}
\big (
    g_{\ell}(z_{\ell,i})
\big )
=
a_{\ell,i} ( 1 - a_{\ell,i})
$$

And

$$
\begin{equation}
\boxed{
    \delta_{\ell,i}
    =
    a_{\ell,i} - 1
}
\end{equation}
$$

And for the **last** one

$$
\begin{align}
{\partial z_{\ell+1,k}
    \over 
\partial a_{\ell,i}}
&=
{\partial
    \over 
\partial a_{\ell,i}}
\big (
    \underline{\underline{\Theta_{\ell + 1}}} \underline{a_{\ell}}
\big )_k \\
&=
{\partial
    \over 
\partial a_{\ell,i}}
\Big (
    \sum^{H_{\ell}}_{c=1} \theta_{\ell + 1,kc} a_{\ell,c}
\Big ) \\
&=
{\partial
    \over 
\partial a_{\ell,i}}
\big (
    \theta_{\ell + 1,k1} a_{\ell,1} + \cdots + \theta_{\ell + 1,kc} a_{\ell,c} + \cdots + \theta_{\ell + 1,kH_{\ell}} a_{\ell,H_{\ell}}
\big )
\end{align}
$$

The derivatives are nulls if $$ i \neq c $$. Then for $$i = c$$

$$
\begin{equation}
\boxed{
    {\partial z_{\ell+1,k}
        \over 
    \partial a_{\ell,i}}
    =
    \theta_{\ell + 1,ki}
}
\end{equation}
$$

with $$ \underline{\underline{\Theta_{\ell + 1}}} \in \mathbb{R}^{H_{\ell+1} \times H_{\ell}} $$.

Then for the **second** component, which is the same for all hidden layers

$$
\begin{align}
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
    &=
    {\partial
        \over 
    \partial z_{\ell,i}}
    \big (
        g_{\ell}(z_{\ell,i})
    \big )\\
    &=
    {\partial
        \over 
    \partial z_{\ell,i}}
    \big (
        \frac{1}
        {1 + \mathrm{e}^{z_{\ell,i}}}
    \big )
\end{align}
$$

$$
\begin{equation}
\boxed{
    {\partial a_{\ell,i}
        \over 
    \partial z_{\ell,i}}
    =
    a_{\ell,i} (1 - a_{\ell,i})
}
\end{equation}
$$

And for the **third** component, which is the same as for the final/output layer

$$
\begin{equation}
\boxed{
    {\partial z_{\ell,i}
        \over 
    \partial\Theta_{\ell,ij}}
    =
    a_{\ell-1,j}
}
\end{equation}
$$

**Finally**

$$
\begin{equation}
\boxed{
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
}
\end{equation}
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

> :alien: *alien says* :speech_balloon:<br>
> $$\odot$$ represents the element wise product, or the Hadamart product.

And summing up over the samples: 

$$
{\partial E_{\text{CE}} 
    \over 
\partial \underline{\underline{\Theta_{\ell}}}}
=
\frac{1}{M} \sum_{k=1}^{M}
\big (
    \underline{\delta_{\ell+1,k}} \underline{\underline{\Theta_{\ell+1}}}
\big )^{\text{T}}
\odot
\underline{a_{\ell,k}}^{\text{T}}
\odot
(1 - \underline{a_{\ell,k}})^{\text{T}}
\underline{a_{\ell-1,k}}
$$

# Sources
* http://www.cs.cornell.edu/courses/cs5740/2016sp/resources/backprop.pdf
* https://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2018/06/ML_NeuralNetworks_SS18.pdf
* https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf
* https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/14%3A_Differentiation_of_Functions_of_Several_Variables/14.5%3A_The_Chain_Rule_for_Multivariable_Functions
* https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_d%C3%A9rivation_des_fonctions_compos%C3%A9es