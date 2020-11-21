---
layout: post
title:  "Mostly used activation functions"
date:   2019-11-17
categories: [activation function]
---
This is a simple record of the most used activation functions and their derivatives.

## Sigmoid

$$
g(z)={1 \over 1 + e^{-z}}
$$

**Derivative**

$${d \over dz} g(z) = g(z)(1-g(z))$$

**Proof**

$$
\frac{d}{dz}(g(z))=\frac{d}{dz}(\frac{1}{1+e^{-z}})=\frac{e^{-z}}{(1+e^{-z})^2}=\frac{1}{1+e^{-z}}\frac{-1+1+e^{-z}}{1+e^{-z}}=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})
$$

## Tangent hyperbolic

$$
g(z)=\tanh(z)
$$

**Derivative**

$$
{d \over dz} g(z) = 1-g^2(z)
$$

**Proof**

$$
\frac{d}{dz}g(z)=\frac{d}{dz}(\tanh(z))=\frac{d}{dz}(\frac{\sinh(z)}{\cosh(z)})=\frac{\cosh^2(z)-\sinh^2(z)}{\cosh^2(z)}=1-\tanh^2(z)
$$

## ReLU

$$
g(z)=\max(0,z)
$$

**Derivative**

$$
{d \over dz} g(z)=
\begin{cases}
    0   &\text{if z < 0} \\
    1   &\text{if z > 0} \\
    undefined   &\text{if z = 0}
\end{cases}
$$

## Softplus or SmoothReLU

$$
g(z) = \log(1 + e^z)
$$

**Derivative**

Derivative of the softplus function is the logistic function.

$$
{d \over dz} g(z)=
\frac{e^z}{1+e^z}=
{1 \over 1 + e^{-z}}
$$

## Leaky ReLU

$$
g(z)=\max(0.01z,z)
$$

**Derivative**

$$
{d \over dz} g(z)=
\begin{cases}
    0.01   &\text{if z < 0} \\
    1   &\text{if z > 0} \\
    undefined   &\text{if z = 0}
\end{cases}
$$

# Sources
* https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/