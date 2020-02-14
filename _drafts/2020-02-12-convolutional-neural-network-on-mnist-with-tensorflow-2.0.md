---
layout: post
title:  "Play with MNIST and tensorflow 2.0"
excerpt: "Use of a convolutional neural network on MNIST with tensorflow 2.0"
date:   2020-02-12
categories: [tensorflow2.0, tf.keras.optimizers]
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

In this post we use tensorflow 2.0 custom model and custom loop on the famous MNIST dataset.
We perform a multiclass classification with a simple convolutional neural network.

Here is a brief presentation from the offical website
> The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image. <br>
> It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

MNIST dataset however only contains 10 classes and it’s images are in the grayscale (1-channel). 
Color images have three 3 channels, one for Red, Green, Blue

## Load useful packages
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
def datasets_distribution(trainingset, testset):
  """
  Get distributions of the training and test set in one dataframe
  :param trainingset: 
  :param testset:
  :return: dataframe containing distribution of the training and test set in percent
  """
  trainingset_df = pd.DataFrame({'label': trainingset})
  testset_df = pd.DataFrame({'label': testset})

  train = trainingset_df.groupby(["label"], as_index=False)["label"].size() * 100 / mTrain
  test = testset_df.groupby(["label"], as_index=False)["label"].size()* 100 / mTest

  traintest = pd.DataFrame({'train': train, 'test': test})
  return traintest
```

```python
traintest = datasets_distribution(y_train, y_test)
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

We then want to plot an histogram of the distribution to see how balanced are the labels among samples, and among the training and test set.

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

### Show me an image
```python
def show_image(dataset, image_index):
  """
  Args:
    dataset (Tensor): MNIST dataset of dimension (None, 28, 28, 1)
    image_index (integer): 
  """
  showImage = dataset[image_index]
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
  Custom convolutional model
  """
  def __init__(self):
    super(SimpleConvModel, self).__init__()

    # Define sequential layers
    self.convolution = Conv2D(32, (3,3), input_shape=(28, 28, 1), activation="relu")
    self.max_pooling = MaxPooling2D(2, 2)
    self.flatten = Flatten()
    self.dense = Dense(128, activation="relu")
    self.softmax = Dense(10, activation="softmax")

    # Save convolutional layer output as property
    self.convolutional_output = tf.constant(0)
    # Input signature for tf.saved_model.save()
    self.input_signature = tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32, name='prev_img')

  def call(self, inputs):
    self.convolutional_output = self.convolution(inputs)
    x = self.max_pooling(self.convolutional_output)
    x = self.flatten(x)
    x = self.dense(x)
    return self.softmax(x)
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

### Save
```python
def save(model, epoch):
  """
  export saved model
  """
  signature_dict = {'model': tf.function(model, input_signature = [model.input_signature])}
  saved_model_dir = 'mnist/epoch/{0}'.format(epoch)
  tf.saved_model.save(model, saved_model_dir, signature_dict)
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
  
  save(model, epoch)

```

```python
#=> epoch 0 train loss 0.20436248 train accuracy 94.03333067893982 test loss 0.07500681 test accuracy 97.68999814987183
#=> INFO:tensorflow:Assets written to: mnist/epoch/0/assets
#=> epoch 1 train loss 0.06440167 train accuracy 98.0983316898346 test loss 0.05907348 test accuracy 98.03000092506409
#=> INFO:tensorflow:Assets written to: mnist/epoch/1/assets
#=> epoch 2 train loss 0.04284174 train accuracy 98.73666763305664 test loss 0.052099597 test accuracy 98.32000136375427
#=> INFO:tensorflow:Assets written to: mnist/epoch/2/assets
#=> epoch 3 train loss 0.031010678 train accuracy 99.05333518981934 test loss 0.040207263 test accuracy 98.69999885559082
#=> INFO:tensorflow:Assets written to: mnist/epoch/3/assets
#=> epoch 4 train loss 0.022623235 train accuracy 99.3066668510437 test loss 0.045790948 test accuracy 98.51999878883362
#=> INFO:tensorflow:Assets written to: mnist/epoch/4/assets
#=> epoch 5 train loss 0.016978111 train accuracy 99.49833154678345 test loss 0.03954892 test accuracy 98.71000051498413
#=> INFO:tensorflow:Assets written to: mnist/epoch/5/assets
#=> epoch 6 train loss 0.01201544 train accuracy 99.66166615486145 test loss 0.043141313 test accuracy 98.64000082015991
#=> INFO:tensorflow:Assets written to: mnist/epoch/6/assets
#=> epoch 7 train loss 0.009675883 train accuracy 99.71500039100647 test loss 0.038950354 test accuracy 98.72000217437744
#=> INFO:tensorflow:Assets written to: mnist/epoch/7/assets
#=> epoch 8 train loss 0.008462262 train accuracy 99.74166750907898 test loss 0.043463036 test accuracy 98.65999817848206
#=> INFO:tensorflow:Assets written to: mnist/epoch/8/assets
#=> epoch 9 train loss 0.0055311522 train accuracy 99.85166788101196 test loss 0.049176537 test accuracy 98.66999983787537
#=> INFO:tensorflow:Assets written to: mnist/epoch/9/assets
```

Aggregation loss function

`tf.keras.metrics.Mean`\
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Mean

Accuracy function

`tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")`

Calculates how often predictions matches integer labels.\
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy


### Plot losses and accuracies
```python
# Train/Test losses
plt.plot(train_losses)
plt.plot(test_losses)
plt.show()
```
![Losses](/assets/2020-02-12/losses.jpg)

```python
# Train/Test accuracies
plt.plot(train_accurarcies)
plt.plot(test_accurarcies)
plt.show()
```
![Accuracies](/assets/2020-02-12/accuracies.jpg)

### Confusion matrix, accuracies, precisions and recalls

The accuracy is the percentage of predictions that are correct

$$
accuracy =
$$

The precision is

$$
precision = {
  tp
  \over
  tp + fp
}
$$

The recall is

$$
recall = {
  tp
  \over
  tp + fn
}
$$

```python
def print_confusion(model, images, labels):
  test_predictions = model(images.reshape(mTest, 28, 28, 1))
  confusion = confusion_matrix(labels, np.argmax(test_predictions,axis=1))
  confusion = confusion.astype('float64')

  recalls = np.diagonal(confusion) / np.sum(confusion, axis=0)
  precisions = np.diagonal(confusion) / np.sum(confusion, axis=1)
  accuracy = compute_accuracy(labels, test_predictions)
  precisions = np.append(precisions, accuracy.numpy())

  accuracies = np.diagonal(confusion) / np.sum(confusion, axis=1)

  confusion = np.round(confusion * 10000 / confusion.sum(axis=1)[:, np.newaxis])
  np.fill_diagonal(confusion, accuracies)

  confusion = np.vstack((confusion, recalls))
  confusion = np.column_stack((confusion, precisions))
  
  fig = plt.figure(figsize = (14,10))
  heatmap = sns.heatmap(confusion, annot=True, cbar=False, fmt='g', vmin=0, vmax=300)
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.plot()
```

### Show me what the networks see
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

### Error analysis


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