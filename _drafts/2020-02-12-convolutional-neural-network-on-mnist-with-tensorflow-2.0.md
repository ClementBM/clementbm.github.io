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
> The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image. <br>
> It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

## Load useful packages
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
```

## Load the dataset

```python
# Load
mnist_dataset = tf.keras.datasets.mnist.load_data()
# Unpack
(x_train, y_train), (x_test, y_test) = mnist_dataset

# Train dataset shapes
print('Train X shape ', x_train.shape)
print('Train Y shape ', y_train.shape)

#=> Train X shape  (60000, 28, 28)
#=> Train Y shape  (60000,)

# Test dataset shapes
print('Test X shape ', x_test.shape)
print('Test Y shape ', y_test.shape)

#=> Test X shape  (10000, 28, 28)
#=> Test Y shape  (10000,)
```

### Explore images
```python
valueCount = len(np.unique(x_train))
minValue = np.min(x_train)
maxValue = np.max(x_train)

print("pixel unique values:", valueCount,
      "\nminValue:", minValue,
      "\nmaxValue:", maxValue)

#=> pixel unique values: 256 
#=> minValue: 0 
#=> maxValue: 255 
```

### Explore labels
```python
labelCount = len(np.unique(y_train))
minValue = np.min(y_train)
maxValue = np.max(y_train)

print("unique labels:", labelCount,
      "\nminValue:", minValue,
      "\nmaxValue:", maxValue)

#=> unique labels: 10
#=> minValue: 0
#=> maxValue: 9
```

### Inspect density probability distribution over the training/test sets


```python
y_train_df = pd.DataFrame({'label': y_train})
y_test_df = pd.DataFrame({'label': y_test})

train = y_train_df.groupby(["label"], as_index=False)["label"].size() * 100 / mTrain
test = y_test_df.groupby(["label"], as_index=False)["label"].size()* 100 / mTest

traintest = pd.DataFrame({'train': train, 'test': test})
print(traintest)

#=>            train   test
#=> label                  
#=> 0       9.871667   9.80
#=> 1      11.236667  11.35
#=> 2       9.930000  10.32
#=> 3      10.218333  10.10
#=> 4       9.736667   9.82
#=> 5       9.035000   8.92
#=> 6       9.863333   9.58
#=> 7      10.441667  10.28
#=> 8       9.751667   9.74
#=> 9       9.915000  10.09
```

```python
labels = np.arange(0,10)  # the x locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (14,10))
ax.bar(labels - width / 2, traintest['train'], width, label='train')
ax.bar(labels + width / 2, traintest['test'], width, label='test')

ax.set_ylabel('%')
ax.set_xlabel('Label')
ax.set_xticks(labels)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_title('Probability density on training and test set')

plt.show()
```

![Training set and test set density probability](/assets/2020-02-12/density-probabilty.png)

We can see that probability distribution for the training set and the test set are pretty close. Then we can perform the test phase without skewed classes.

### Show image
```python
def show_image(image_number):
  showImage = x_train[image_number]
  showImage = showImage.reshape(28,28)
  plt.imshow(showImage, cmap=plt.get_cmap('gray_r'))
  plt.show()
```

```python
show_image(0)
```
![Number 5](/assets/2020-02-12/number-5.png)

### Gather and prepare

Reduce the samples from integers 0-255 to floating-point numbers 0.0-1.0

Prepare for conv layer
Add one dimension, because the first convolutional layer except 4D tensor [batch, in_height, in_width, in_channels]
See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d for more details

```python
def prepare_mnist_dataset(mnist_dataset):
  """
  Format MNIST dataset
  http://yann.lecun.com/exdb/mnist/
  """
  (x_train, y_train), (x_test, y_test) = mnist_dataset
  # Reduce the samples
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  # Prepare for conv layer
  mTrain = x_train.shape[0]
  mTest = x_test.shape[0]
  x_train = x_train.reshape(mTrain, 28, 28, 1)
  x_test = x_test.reshape(mTest, 28, 28, 1)
  return (x_train, y_train), (x_test, y_test)
```
**Prepare for conv layer**: Add one dimension, because the first convolutional layer except 4D tensor [batch, in_height, in_width, in_channels]. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d for more details.

For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.

```python
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
class SimpleConvModel(Model):
  """
  Define custom model
  """
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

### Optimizer
```python
optimizerFunction = tf.keras.optimizers.Adam()
```
`tf.keras.optimizers.Adam()`

Define loss function, optimizer, and metrics to measure model loss and accuracy\
ADAM optimizer for (Adaptive Moment Estimation)\
ADAM computes first and seconde order gradients

### Loss
```python
trainLoss = tf.keras.losses.SparseCategoricalCrossentropy()
# @tf.function
def compute_loss(labels, logits):
  return trainLoss(labels, logits)
```

`tf.keras.losses.SparseCategoricalCrossentropy()`

Computes the crossentropy loss between the labels and predictions.\
https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

### Accuracy
```python
# @tf.function
def compute_accuracy(labels, logits):
  predictions = tf.math.argmax(logits, axis=1)
  return tf.math.reduce_mean(tf.cast(tf.math.equal(predictions, labels), tf.float32))
```
`tf.math.argmax`

Returns the index with the largest value across axes of a tensor.

`tf.math.reduce_mean`

Computes the mean of elements across dimensions of a tensor.

`tf.cast`

Casts a tensor to a new type.

`tf.math.equal`

Returns the truth value of (x == y) element-wise.

## Action

### Create a model instance
```python
# Create a model instance
model = SimpleConvModel()
model.build(input_shape=(None, 28, 28, 1))
model.summary()
```

```python
#=> Model: "simple_conv_model"
#=> _________________________________________________________________
#=> Layer (type)                 Output Shape              Param #   
#=> =================================================================
#=> conv2d (Conv2D)              multiple                  320       
#=> _________________________________________________________________
#=> max_pooling2d (MaxPooling2D) multiple                  0         
#=> _________________________________________________________________
#=> flatten (Flatten)            multiple                  0         
#=> _________________________________________________________________
#=> dense (Dense)                multiple                  692352    
#=> _________________________________________________________________
#=> dense_1 (Dense)              multiple                  1290      
#=> =================================================================
#=> Total params: 693,962
#=> Trainable params: 693,962
#=> Non-trainable params: 0
#=> _________________________________________________________________
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
                        shuffle_count = 60000,
                        batch_count = 64)

# Test
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset_batch = test_dataset.batch(100)
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
    # update model from gradients
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
  
  signature_dict = {'model': tf.function(model, input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32, name='prev_img')])}
  saved_model_dir = 'mnist/epoch/{0}'.format(epoch)
  tf.saved_model.save(model, saved_model_dir, signature_dict)
```

Aggregation loss function

`tf.keras.metrics.Mean`\
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Mean

Accuracy function

`tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")`

Calculates how often predictions matches integer labels.\
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy

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
* [MNIST Database](http://yann.lecun.com/exdb/mnist/)