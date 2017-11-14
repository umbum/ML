import tensorflow as tf

### tensorboard 사용 가능.
## input은 28*28이어야 한다.

class CNN:
    __slots__ = ('sess', 'name', 'lr', 'training', 'X', 'Y', 'logits', 'cost', 'optimizer', 'accuracy', 'merged_summary')
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.lr = tf.placeholder(tf.float32)    # train에서만 사용한다.
            self.training =  tf.placeholder(tf.bool)
            summarys = []
            
            ''' mnist.train.images[0]의 형상은 (784, )이므로 밑에가 [784, None]이 되어야 할 것 같지만 
            X에 들어갈 때 한 번 더 묶여서 2차원 배열(1,784)로 변경된다는 점에 주의.'''
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1]) #N, H, W, C
            self.Y = tf.placeholder(tf.float32, [None, 10])

            #filter는 3x3x1이고 FN은 32. 그러나 channel은 어차피 입력과 같으므로 입력하지 않는다.
            #W1 정의, tf.nn.conv2d, tf.nn.relu를 한꺼번에 layers로 처리한다.
            with tf.name_scope('L1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
                
            with tf.name_scope('L2'):
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)
            
            with tf.name_scope('L3'):
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding = "SAME", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding="SAME", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)
            
            with tf.name_scope('L4_dense'):
                flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
            
            with tf.name_scope('L5_dense'):
                self.logits = tf.layers.dense(inputs=dropout4, units=10)
            summary_logits = tf.summary.histogram('output value', self.logits)
            #summary.append(tf.summary.histogram('output value', logits))
            
            ''' 원래 Softmax의 출력을 MSE 또는 CEE에 넣고 결과를 최소로 하는 방향으로 최적화 해 가는건데.
            여기서 reduce_mean으로 또 평균을 내는 이유는 배치 때문이다.
            tf.nn.softmax_cross_entropy_with_logits는 A 1-D Tensor of length batch_size를 리턴하기 때문.
            오차함수를 계산하려면 비교 대상 data와 정답 label이 필요하다.
            tf.nn.softmax_cross_entropy_with_logits은 입력 data를 logits으로, 정답 label을 labels로 받는다.'''
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        
            ''' epoch 마다 feed_dict = {test}를 넘겨서 정확한 accuracy를 계산하도록 한다.
            tensorboard에 찍히는 accuracy는 train data를 대상으로 계산한 accuracy임에 주의.
            correct_prediction은 True/False 1-D Tensor.
            캐스팅 한 다음 다 더해서 개수만큼 나누면 되는데 이게 곧 평균이니까 reduce_mean.'''
            correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            summary_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            #summary.append(tf.summary.scalar('accuracy', accuracy))
            
            self.merged_summary = tf.summary.merge([summary_logits, summary_accuracy])

    
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict = {self.X: x_test, self.Y: y_test, self.training: training})
        
    def train(self, x_data, y_data, learning_rate, training=True, summary=True):
        fetches = [self.cost, self.optimizer]
        if summary:
            fetches.append(self.merged_summary)
        
        return self.sess.run(fetches, feed_dict={self.X: x_data, self.Y:y_data, self.training:training, self.lr:learning_rate})
    
    def prediction(self, x_test, training=False):
        ''' logits의 값을 알고 싶은 경우(신경망이 어떻게 추론했는지 출력 뉴런 값을 알고 싶은 경우)에는 prediction을 따로 호출. '''
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training:training})
        