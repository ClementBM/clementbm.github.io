---
layout: post
title:  "Play with MNIST and tensorflow 2.0"
excerpt: "Use of a convolutional neural network on MNIST with tensorflow 2.0"
date:   2020-02-12
categories: [tensorflow2.0, tf.keras.optimizers]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

In this post we'll try tensorflow 2.0 custom model and custom loop on the famous MNIST dataset.
We'll perform a multiclass classification, with a simple convolutional neural network.

## Load useful packages
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
```

## Load the dataset
> The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.
> It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

Sourced from [here](http://yann.lecun.com/exdb/mnist/)

```python
# Load
mnist_dataset = tf.keras.datasets.mnist.load_data()
# Unpack
(x_train, y_train), (x_test, y_test) = mnist_dataset

# Train dataset shapes
print('Train X shape ', x_train.shape)
print('Train Y shape ', y_train.shape)

# Test dataset shapes
print('Test X shape ', x_test.shape)
print('Test Y shape ', y_test.shape)
```

{% highlight python %}
#=> <tf.Variable 'UnreadVariable' shape=() dtype=int64, numpy=1000>
#=> -2.9999943
{% endhighlight %}

### Explore
```python
valueCount = len(np.unique(x_train))
minValue = np.min(x_train)
maxValue = np.max(x_train)
std = np.std(x_train)
mean = np.mean(x_train)
print("minValue:", minValue, ", maxValue:", maxValue, ", std:", std, ", mean:", mean)
```

**Distribution** in the training and test sets

```python
H, edges = np.histogram(y_train)
```

```python
plt.hist(y_train)
plt.gca().set(title='Training set', ylabel='Frequency')
plt.show()

plt.hist(y_test)
plt.gca().set(title='Test set', ylabel='Frequency')
plt.show()
```
We can see that probability distribution for the training set and the test set are pretty close. Then we can perform the test phase without skewed classes.

### Show image
```python
showImage = x_train[0]
showImage = showImage.reshape(28,28)
plt.imshow(showImage, cmap=plt.get_cmap('gray_r'))
plt.show()
```

### Gather and prepare
```python
# Get the data
def prepare_mnist_dataset(mnist_dataset):
  """
  Format MNIST dataset
  http://yann.lecun.com/exdb/mnist/
  """
  (x_train, y_train), (x_test, y_test) = mnist_dataset
  # Reduce the samples from integers 0-255 to floating-point numbers 0.0-1.0
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  # Prepare for convolutional layer
  mTrain = x_train.shape[0]
  mTest = x_test.shape[0]
  x_train = x_train.reshape(mTrain, 28, 28, 1)
  x_test = x_test.reshape(mTest, 28, 28, 1)
  return (x_train, y_train), (x_test, y_test)
```
**Prepare for conv layer**: Add one dimension, because the first convolutional layer except 4D tensor [batch, in_height, in_width, in_channels]. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d for more details.

```python
# Shuffle the data
def shuffle_batch_dataset(train_dataset, take_count, shuffle_count, batch_count):
  """
  Use tf.data to batch and shuffle the datasets
  """
  train_ds = train_dataset.take(take_count).shuffle(shuffle_count, seed=42).batch(batch_count)
  return train_ds
```

For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.

## Define the model

```python
# Define custom model
class SimpleConvModel(Model):
  def __init__(self):
    super(SimpleConvModel, self).__init__()

    self.convLayer = Conv2D(32, (3,3), input_shape=(28, 28, 1), activation="relu")
    self.maxPoolingLayer = MaxPooling2D(2, 2)
    self.flatten = Flatten()
    self.denseLayer1 = Dense(128, activation="relu")
    self.denseLayer2 = Dense(10, activation="softmax")
    self.convolutionalLayerOutput = tf.constant(0)

  def call(self, inputs):
    self.convolutionalLayerOutput = self.convLayer(inputs)
    x = self.maxPoolingLayer(self.convolutionalLayerOutput)
    x = self.flatten(x)
    x = self.denseLayer1(x)
    return self.denseLayer2(x)
  
  def convolutionalLayerOutput():
    return self.convolutionalLayerOutput
```

`Conv2D(32, (3,3), input_shape=(28, 28, 1), activation="relu")`

2D convolutional layer for spatial convolution over an image\
Conv2D has only one channel has the input is grey colour\
Number of convolutions: 32, size of the convolution 3x3 grid
Activation function: ReLU

`MaxPooling2D(2, 2)`

MaxPooling layer compress the image,
while maintaining the content of the features that were highlighted 
by the convolution. \
By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.

### Loss
`tf.keras.losses.SparseCategoricalCrossentropy()`

Computes the crossentropy loss between the labels and predictions.\
https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

```python
trainLoss = tf.keras.losses.SparseCategoricalCrossentropy()
# @tf.function
def compute_loss(labels, logits):
  return trainLoss(labels, logits)
```

## Action

### Create a model instance
```python
# Create a model instance
model = SimpleConvModel()
model.build(input_shape=(None, 28, 28, 1))
model.summary()
```

### Get the data prepared
```python
# Prepare data
(x_train, y_train), (x_test, y_test) = prepare_mnist_dataset(mnist_dataset)

# Train
# Get a `TensorSliceDataset` object from `ndarray`s
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_batch = shuffle_batch_dataset(train_dataset,
                        take_count = 60000,
                        shuffle_count = 10000,
                        batch_count = 64)

# Test
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset_batch = test_dataset.batch(100)
```

### Optimizer
`tf.keras.optimizers.Adam()`

Define loss function, optimizer, and metrics to measure model loss and accuracy\
ADAM optimizer for (Adaptive Moment Estimation)\
ADAM computes first and seconde order gradients
```python
optimizerFunction = tf.keras.optimizers.Adam()
```

## Loop

```python
EPOCHS = 10

train_losses = []
train_accurarcies = []

test_losses = []
test_accurarcies = []

for epoch in range(EPOCHS):
  trainLossAggregate = tf.keras.metrics.Mean(name="train_loss")
  testLossAggregate = tf.keras.metrics.Mean(name="test_loss")
  trainAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
  testAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

  for train_images, train_labels in train_dataset_batch:
    with tf.GradientTape() as tape:
      # forward propagation
      predictions = model(train_images)
      # calculate loss
      loss = compute_loss(train_labels, predictions)
      
    # calculate gradients from model definition and loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # update model from gradients, apply_gradients(grads_and_vars, name=None)
    # grads_and_vars: List of (gradient, variable) pairs.
    optimizerFunction.apply_gradients(zip(gradients, model.trainable_variables))

    trainLossAggregate(loss)
    trainAccuracy(train_labels, predictions)

  for test_images, test_labels in test_dataset_batch:
    predictions = model(test_images)

    loss = compute_loss(test_labels, predictions)

    testLossAggregate(loss)
    testAccuracy(test_labels, predictions)

  train_losses.append(trainLossAggregate.result().numpy())
  train_accurarcies.append(trainAccuracy.result().numpy()*100)

  test_losses.append(testLossAggregate.result().numpy())
  test_accurarcies.append(testAccuracy.result().numpy()*100)

  print('epoch', epoch,
        'train loss', train_losses[-1],
        'train accuracy', train_accurarcies[-1],
        'test loss', test_losses[-1],
        'test accuracy', test_accurarcies[-1])
```

### Save
```python
"""
export saved model
"""
tf.saved_model.save(model, 'mnist/1')
```

### Plot losses and accuracies
```python
# Train/Test losses
plt.plot(train_losses)
plt.plot(test_losses)
plt.show()
```

```python
# Train/Test accuracies
plt.plot(train_accurarcies)
plt.plot(test_accurarcies)
plt.show()
```

### Show me the convolutions
```python
VALUE_INDEX = 10

singleTestValue = x_test[VALUE_INDEX].reshape(1, 28, 28, 1)
singleTestPrediction = model(singleTestValue)

print("label", y_test[VALUE_INDEX])
print("prediction", tf.math.argmax(singleTestPrediction, axis=1).numpy())

d = x_test[VALUE_INDEX].reshape(28,28)
plt.imshow(d, cmap=plt.get_cmap('gray_r'))
plt.show()

print(model.convolutionalLayerOutput.shape)

# print all convolutions
for convolutionIndex in range(0,model.convolutionalLayerOutput.shape[-1] - 1):
  convolutionLayer = model.convolutionalLayerOutput[0, :, :, convolutionIndex]
  plt.imshow(convolutionLayer, cmap=plt.get_cmap('gray_r'))
  plt.show()
```

## Autograph
https://towardsdatascience.com/tensorflow-2-0-tf-function-and-autograph-af2b974cf4f7
@tf.function

## Benchmark
http://yann.lecun.com/exdb/mnist/

## Confusion matrix
https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

# Sources
Build the tf.keras model using the Keras model subclassing API
* https://www.tensorflow.org/guide/keras#model_subclassing
* https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
* https://www.tensorflow.org/