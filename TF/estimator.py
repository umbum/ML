import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)

def img_to_dataset():
    pass

def input_fn_wrap():
    pass



def main():
  # Load datasets.
  train_datas = mnist.train.images
  train_labels = mnist.train.labels.astype(np.int32)    # cast 해야 함.

  test_datas = mnist.test.images
  test_labels = mnist.test.labels.astype(np.int32)

  # Specify that all features have real-value data. 그리고 input shape도 지정한다.
  feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=10,    # output. 분류해야 하는 class 수
                                          model_dir="./mnist_model")

  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(train_datas)},
      y=np.array(train_labels),
      num_epochs=2,
      shuffle=True)

  ### Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_datas)},
      y=np.array(test_labels),
      num_epochs=1,
      shuffle=False)

  ### Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  ### Prediction
  '''
  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))
  '''

if __name__ == "__main__":
    main()
