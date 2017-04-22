import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

#dataset을 한 줄씩 가져오기 위해
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')

#일단은 text 형태로 가져온다.
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#record_defaults 인자는 csv 내부의 data를 어떤 식으로 반환할지를 지정한다.
record_defaults=[[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


'''
x_data = [[73., 80., 85.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]
y는 CNN의 정답레이블 t에 대응된다고 생각하면 된다.
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

coord = tf.train.Coordinator()#threads의 종료를 조정하기 위해
#string_input_producer에서 자동으로 추가된 QueueRunner를 start시켜 파일을 읽어온다
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    sess.run(train, feed_dict={X: x_batch, Y: y_batch})
    if step % 100 == 0:
        cost_val, hy_val = sess.run(
            [cost, hypothesis], feed_dict={X:x_batch, Y: y_batch})
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
print("approximation target\n", y_batch)        

#request that the threads stop and wait until the threads terminate
coord.request_stop()
coord.join(threads)

#Ask to machine
myScore = sess.run(hypothesis, feed_dict={X:[[100, 90, 101]]})
print(myScore)

