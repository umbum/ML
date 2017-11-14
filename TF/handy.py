import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from TF_CNN import CNN


tf.set_random_seed(777)
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


class TFCNN_handy:
    def __init__(self, ensemble=2, restore_step=0):
        self.sess = tf.Session()

        ### Build Ensemble Graph ###
        self.ensemble = ensemble
        self.models = []
        for m in range(self.ensemble):
            self.models.append(CNN(self.sess, "model" + str(m)))

        ### restore variables ###
        self.saver = tf.train.Saver(max_to_keep=15)
        self.restore_step = restore_step
        self.restore_variables()
        
        
    def restore_variables(self):
        if self.restore_step:
            restore_path = "./Variables/mnist-en{}-{}".format(self.ensemble, self.restore_step)
            self.saver.restore(self.sess, restore_path)
            print("[*] Variables are restored ", restore_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("[*] Variables are initialized")
        
    def train_wrap(self, lr, epochs, batch_size, tflogs=True, save=True):
        if tflogs:
            today = datetime.now().strftime('%y%m%d-%H%M')
            writer = tf.summary.FileWriter('./tflogs/log{}-{}'.format(today, self.restore_step), graph=self.sess.graph)
            #writer.add_graph(self.sess.graph)
        
        step = 0
        for epoch in range(epochs):
            avg_cost_list = np.zeros(len(self.models))
            total_batch = int(mnist.train.num_examples / batch_size)
            #total_batch = int(total_batch/100)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)    
                step += 1
                for m_idx, m in enumerate(self.models):
                    ### run Neural Net
                    if tflogs:
                        c, _, s = m.train(batch_xs, batch_ys, lr)
                        writer.add_summary(s, global_step=step)
                    else:
                        c, _ = m.train(batch_xs, batch_ys, lr, summary=False)
                    avg_cost_list[m_idx] += c / total_batch
                if (step % 100) == 0:
               	    print('step:', '%04d' % (step+self.restore_step), 'cost =', c)
            print('Epoch:', '%04d' % (epoch + 1), 'avg_cost =', avg_cost_list)
            self.test_accuracy(mnist.test.labels[:1000], mnist.test.images[:1000])
            if save:
                self.saver.save(self.sess, "./Variables/mnist-en{}-{}".format(self.ensemble, step+self.restore_step))
        
    
    def test_accuracy(self, test_label, test_images):
        # for m_idx, m in enumerate(self.models):
            # print("model", m_idx, " Test Dataset Accuracy   :", m.get_accuracy(test_images, test_label))
        
        ensemble_correct_prediction = tf.equal(self.prediction_wrap(test_label, test_images), tf.argmax(test_label, axis=1))
        # 캐스팅 한 다음 다 더해서 개수만큼 나누면 되는데 이게 곧 평균이니까 reduce_mean
        ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        print("Ensemble accuracy : ", self.sess.run(ensemble_accuracy))


    def prediction_wrap(self, label, images):
        predictions = np.zeros(len(label) * 10).reshape(len(label), 10)
        
        for m_idx, m in enumerate(self.models):
            # ensemble이니까, prediction에서 두 결과를 더했을 때 가장 높은 결과를 갖는 라벨을 반환해야 의미가 있다.
            predictions += m.prediction(images)    # numpy broadcast
        
        return self.sess.run(tf.argmax(predictions, axis=1))
        


if __name__ == '__main__':
    cnn = TFCNN_handy(ensemble=2, restore_step=5500)
    # mnist.train.num_examples는 55000이므로 55000/100 => 1 epoch 당 550번 반복
    # 원래 epochs=15정도 줘야.
    cnn.train_wrap(lr=0.001, epochs=5, batch_size=100, tflogs=True, save=True)
    # print("idx : ", cnn.prediction_wrap(mnist.test.labels[:10], mnist.test.images[:10]))
    
