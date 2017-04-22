import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 100  
epochs = 2
#mnist.train.num_examples는 55000이므로 1 epoch 당 550번 반복
#총 반복 수는 15 * 550...인데 속도 때문에 2 * 550으로 변경.

class Model:
    __slots__ = ('sess', 'name')
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = 

'''mnist.train.images[0]의 형상은 (784, )이므로 밑에가 [784, None]이 되어야 할 것 같지만 
X에 제공될 때 한 번 더 묶어서 2차원 배열(1,784)로 변경된다는 점에 주의.'''
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) #N, H, W, C
Y = tf.placeholder(tf.float32, [None, 10])

#filter는 3x3x1이고.. FN은 32.
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
''' padding="SAME"은 stride가 1, 1일 때 입력 형상이 그대로 출력 형상이 되는 padding을 말한다.
conv layer에서는 stride가 1, 1이고, padding="SAME"이므로
입력 형상이 그대로 출력 형상이 된다. (?, 28, 28, 32)
pooling layer에서는 stride가 2, 2이고,padding="SAME"이므로
출력 형상이 입력 형상의 반이 된다. (?, 14, 14, 32)
(stride의 첫 번째와 마지막은 그냥 1로 고정이라고 생각하면 된다)'''
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                   strides=[1, 2, 2, 1], padding="SAME")

'''L1에서는 입력이 1channel이라 filter가 1장만 있었지만
L1을 거쳐 channel이 32가 되었다.
channel 1개 당 filter 1장이 필요하니까 32장의 filter가 필요하고
이런 filter 박스는 몇 개 있어도 상관 없지만 여기서는 64개.'''
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
strides=[1, 2, 2, 1], padding="SAME")

#FC layer를 위해 flatting
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
initializer = tf.contrib.layers.xavier_initializer()) 

b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

#원래 Softmax의 출력을 MSE 또는 CEE에 넣고 결과를 최소로 하는 방향으로 최적화 해 가는건데.
#여기서 reduce_mean으로 또 평균을 내는 이유는 배치 때문이다.
#tf.nn.softmax_cross_entropy_with_logits는 A 1-D Tensor of length batch_size를 리턴하기 때문.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
'''오차함수를 계산하려면 비교 대상 data와 정답 label이 필요하다.
tf.nn.softmax_cross_entropy_with_logits은 입력 data를 logits으로, 정답 label을 labels로 받는다.'''
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


total_batch = int(mnist.train.num_examples / batch_size)
#속도 때문에 아래 줄을 추가.
total_batch = int(total_batch/10)

for epoch in range(epochs):
    avg_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch 
        #total_batch만큼 반복하니까. c만 더하고 나중에 total_batch로 한번만 나눠줘도 된다.
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


#logit과 Y는 반복할 때 마다 변경되는 값이지만 여기서 테스트하기 위해 한 번 수행.
#correct_prediction은 True/False 1-D Tensor
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
#캐스팅 한 다음 다 더해서 개수만큼 나누기 = 평균이니까 reduce_mean.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy   :", sess.run(accuracy, feed_dict={X:mnist.test.images[:1000], Y: mnist.test.labels[:1000]}))

print("=====TEST=====")
r = random.randint(0, mnist.test.num_examples -1)
print("Label      : ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], axis=1)))
print("Prediction : ", sess.run(tf.argmax(logits, axis=1), feed_dict={X: mnist.test.images[r:r + 1]}))
