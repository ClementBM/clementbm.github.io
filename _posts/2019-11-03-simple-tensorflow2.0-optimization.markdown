---
layout: post
title:  "Simple tensorflow 2.0 optimization"
excerpt: "Understanding function optimization with tensorflow 2.0"
date:   2019-11-03
categories: [tensorflow2.0, tf.keras.optimizers]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Why tensorflow 2.0 ?
Tensorflow 2.0 was recently released ! It provides, better usability with `tf.keras` as the recommended high-level API and `Eager execution` by default. It also improves clarity by removing duplicated functionalities and putting forward an intuitive syntax accross APIs. Save models is simpler, in fact there's just one way of saving a model.
It improves flexibility by providing different level of customization. For instance `variable`, `checkpoint` and `layers` have now inheritable interfaces.

Time to dig in one simple example !

# One simple equation

Let's define one simple equation
$$ x^2 + 6x + 9 = 0 $$.

This quadratic equation can be reduced to
$$ (x + 3)^2 = 0 $$
therefore we can easily find the exact solution when $$ x = -3 $$.

![Simple quadratic function](/assets/2019-11-03/quadratic-function-plot.png)

# Numerical resolution with tensorflow 2.0
For this example, we'll use tensorflow 2.0 and one of the optimizer in `tf.keras.optimizers` to find a numerical approximation of the previous equation.
First, let's import the basic packages
{% highlight python %}
import numpy as np
import tensorflow as tf
{% endhighlight %}

Then, we define the variable x with `tf.Variable`, see the [doc](https://www.tensorflow.org/api_docs/python/tf/Variable#__init__). We arbitrarily initialize the variable `x` to 0.
{% highlight python %}
x = tf.Variable(initial_value=0, name='x', trainable=True, dtype=tf.float32)
{% endhighlight %}


We specify the coefficients with `tf.constant`, see the [doc](https://www.tensorflow.org/api_docs/python/tf/constant)
{% highlight python %}
coefficients = tf.constant(value=[1,6,9], shape=(3,1), dtype=tf.float32)
{% endhighlight %}

As the `minimize()` function requires the cost function as a method, we set it as follow
{% highlight python %}
def compute_loss():
    return coefficients[0][0]*x**2 + coefficients[1][0]*x + coefficients[2][0]
{% endhighlight %}

Then we choose the `stochastic gradient descent and momentum` optimizer among those provided by `tf.keras`, see the [doc](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD#__init__)
{% highlight python %}
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
{% endhighlight %}

And finally we call the `minimize()` method that will simply compute gradient using `tf.GradientTape` and call `apply_gradients()`, see the [doc](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD#minimize)
{% highlight python %}
for i in range(1000):
  optimizer.minimize(loss=compute_loss, var_list=[x])
# print approximate solution
print(x.numpy())
{% endhighlight %}

# The whole piece of code
{% highlight python %}
# Import basic packages
import numpy as np
import tensorflow as tf

# define variable
x = tf.Variable(initial_value=0, name='x', trainable=True, dtype=tf.float32)

# define coefficients
coefficients = tf.constant(value=[1,6,9], shape=(3,1), dtype=tf.float32)

# cost function
def compute_loss():
  return coefficients[0][0]*x**2 + coefficients[1][0]*x + coefficients[2][0]

# get SGD optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

## Run 1000 times
for i in range(1000):
  optimizer.minimize(loss=compute_loss, var_list=[x])

# Print approximation
print(x.numpy())
#=> <tf.Variable 'UnreadVariable' shape=() dtype=int64, numpy=1000>
#=> -2.9999943
{% endhighlight %}

# Sources
* https://www.tensorflow.org/
* https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0