import os
import glob

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data



class Estimator:
    def __init__(self, fpaths, shape, n_output, model_dir):
        self.fpaths = fpaths
        self.shape = shape
        self.train_labels = []
        self.train_epochs = 1
        # Specify that all features have real-value data.
        # 그리고 feature name과 input shape도 지정한다.
        # input_fn이 반환하는 features는 feature name : data 형태의 dict여야 한다.
        feature_columns = [tf.feature_column.numeric_column("x", shape=self.shape)]
        # Build 3 layer DNN with 10, 20, 10 units respectively.
        self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[512, 256, 128],
                                                n_classes=n_output,    # 분류해야 하는 class 수
                                                model_dir=model_dir)
        self.mnist = None
    
        
    def _image_processing(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        # tf.image.decode_image를 사용하면 호환성이 좋지만 shape=<unknown>이라 resize에서 에러 발생
        image_resized = tf.image.resize_images(image_decoded, self.shape)
        return {"x": image_resized}, label
    
    # predict일 경우 label이 없어도 에러나지 않음.
    def dataset_input_fn(self):
        fpaths = tf.constant(self.fpaths)
        labels = tf.constant(self.train_labels)
    
        dataset = tf.data.Dataset.from_tensor_slices((fpaths, labels))
        
        dataset = dataset.map(self._image_processing)
        ### Note! batch 실행 안해주면 [N, W, H, C]가 아니라 [W, H, C]가 넘어가서 에러 발생.
        dataset = dataset.repeat(self.train_epochs)
        dataset = dataset.batch(1)
        
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    
    def predict(self):
        ### Prediction
        self.train_labels = np.empty(len(self.fpaths))
        self.train_epochs = 1
        predictions = self.classifier.predict(input_fn=self.dataset_input_fn)
        return [p["classes"] for p in predictions]
    
    def train(self, steps=100, epochs=1):
        fnames_ext = [os.path.split(f)[1] for f in self.fpaths]
        fnames = [int(os.path.splitext(f)[0]) for f in fnames_ext]
        self.train_labels = fnames    # 파일 이름이 곧 label인 경우.
        self.train_epochs = epochs
        self.classifier.train(input_fn=self.dataset_input_fn, steps=steps)
    
    def accuracy(self):
        fnames_ext = [os.path.split(f)[1] for f in self.fpaths]
        fnames = [int(os.path.splitext(f)[0]) for f in fnames_ext]
        self.train_labels = fnames
        return self.classifier.evaluate(input_fn=self.dataset_input_fn)["accuracy"]
    
    
    def mnist_train(self, steps=100, epochs=1):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
            
        train_datas = self.mnist.train.images
        train_labels = self.mnist.train.labels.astype(np.int32)    # cast 해야 함.
        
        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(train_datas)},
            y=np.array(train_labels),
            num_epochs=epochs,
            shuffle=True)
        ### Train model.
        self.classifier.train(input_fn=train_input_fn, steps=steps)
    
    def mnist_accuracy(self):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
            
        test_datas = self.mnist.test.images
        test_labels = self.mnist.test.labels.astype(np.int32)        
        
        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(test_datas)},
            y=np.array(test_labels),
            num_epochs=1,
            shuffle=False)
        ### Evaluate accuracy.
        return self.classifier.evaluate(input_fn=test_input_fn)["accuracy"]
        


if __name__ == "__main__":
    e = Estimator(
        fpaths = glob.glob("./imgs/*"), 
        shape = (28,28), 
        n_output = 13, 
        model_dir="./ckpt_estimator")
        
    # e.mnist_train(500, 2)
    # print("\nMNIST Test Accuracy: {0:f}\n".format(e.mnist_accuracy()))
    
    e.train(steps=100, epochs=10)
    print("Prediction : ", e.predict())
    