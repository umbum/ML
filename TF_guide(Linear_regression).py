import tensorflow as tf
import numpy

tf.set_random_seed(777)

x_data = [[73., 80., 85.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]
'''y는 CNN의 정답레이블 t에 대응된다고 생각하면 된다.
이 예제에서는 MSE를 이용하게 되는데, 
CNN에서의 MSE는 신경망의 출력 y에 정답 레이블 t를 빼서 오차를 구하게 된다.
여기서는 손실 함수의 개념이 아직 적용되지 않았고
classification이 아니라 regression이므로, y 데이터가 저런 모양이다.'''

#X는 5,3 W는 3,1이므로, X는 5개의 2차원 x batch라고 생각하면 된다.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y)) #MSE
# 위에서 설명한 대로 hypothesis(출력)에서 Y를 뺀다.
# hypothesis = Y일 때 cost가 최소이므로,
# W와 b는 hypothesis = Y로 만드는 방향으로 학습한다.
###   hypothesis는 Y값에 가까워 진다는게 핵심.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
'''
DLfromScratch에서는 반복 마다 계산된 grads를 optimizer.update(params, grads)로 집어 넣어
params를 갱신하도록 했다. 
여기서는 optimizer에 cost만 집어 넣고 sess.run(optimizer) 시 알아서 갱신된다.
따라서 train만 호출하면 train-cost-hypothesis-w, b로 연결된다.
'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y: y_data})
    if step % 100 == 0:
        cost_val, hy_val = sess.run(
            [cost, hypothesis], feed_dict={X:x_data, Y: y_data})
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

#Ask to machine
myScore = sess.run(hypothesis, feed_dict={X:[[100, 90, 101]]})
print(myScore)