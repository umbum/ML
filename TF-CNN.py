import tensorflow as tf
import random
#import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 100  
epochs = 2
#mnist.train.num_examples는 55000이므로 1 epoch 당 550번 반복
#총 반복 수는 15 * 550...인데 속도 때문에 2 * 550으로 변경.

class CNN:
    __slots__ = ('sess', 'name', 'training', 'X', 'Y', 'logits', 'cost', 'optimizer', 'accuracy')
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training =  tf.placeholder(tf.bool)
            
            '''mnist.train.images[0]의 형상은 (784, )이므로 밑에가 [784, None]이 되어야 할 것 같지만 
            X에 들어갈 때 한 번 더 묶여서 2차원 배열(1,784)로 변경된다는 점에 주의.'''
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1]) #N, H, W, C
            self.Y = tf.placeholder(tf.float32, [None, 10])

            #filter는 3x3x1이고 FN은 32. 그러나 channel은 어차피 입력과 같으므로 입력하지 않는다.
            #W1 정의, tf.nn.conv2d, tf.nn.relu를 한꺼번에 layers로 처리한다.
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)         
            
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)
                                     
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding = "SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)
            
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
            self.logits = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

        '''원래 Softmax의 출력을 MSE 또는 CEE에 넣고 결과를 최소로 하는 방향으로 최적화 해 가는건데.
        여기서 reduce_mean으로 또 평균을 내는 이유는 배치 때문이다.
        tf.nn.softmax_cross_entropy_with_logits는 A 1-D Tensor of length batch_size를 리턴하기 때문.
        오차함수를 계산하려면 비교 대상 data와 정답 label이 필요하다.
tf.nn.softmax_cross_entropy_with_logits은 입력 data를 logits으로, 정답 label을 labels로 받는다.'''
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        '''
        logit과 Y는 반복할 때 마다 변경되는 값이지만 여기서 테스트하기 위해 한 번 수행.
        correct_prediction은 True/False 1-D Tensor
        캐스팅 한 다음 다 더해서 개수만큼 나누기 = 평균이니까 reduce_mean.
        '''
        correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.Y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def prediction(self, x_test, training=False):
        '''
        train에서 optimizer 호출하면 한방에 처리되기 때문에 굳이 있을 필요는 없지만
        logits의 값을 알고 싶은 경우(신경망이 어떻게 추론했는지 출력 뉴런 값을 알고 싶은 경우)에는 prediction을 따로 호출.
        '''
        return self.sess.run(self.logits, feed_dict={X: x_test, self.training:training})
    
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict = {X: x_test, Y: y_test, self.training: training})
        
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y:y_data, self.training: training})
        
def _main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #Ensemble
    models = []
    num_models = 2
    for m in range(num_models):
        models.append(CNN(sess, "model" + str(m)))
    
    for epoch in range(epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)
        total_batch = int(total_batch/10) #속도 때문에 추가.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)    
            for m_idx, m in enumerate(models):
                c, _ = m.train(batch_xs, batch_ys)
                avg_cost_list[m_idx] += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        
    _check(models)

def _check(models):
    test_size = len(mnist.test.labels[:1000])
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)
    
    for m_idx, m in enumerate(models):
        print(m_idx, "Accuracy   :", m.get_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        p = m.predict(mnist.test.images[:1000])
        predictions += p 
        
    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(mnist.test.labels, axis=1))
    #캐스팅 한 다음 다 더해서 개수만큼 나누기 = 평균이니까 reduce_mean.
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print("Ensemble accuracy : ", sess.run(ensemble_accuracy))

if __name__ == '__main__':
    _main()
