import tensorflow as tf
import random
from datetime import date
#import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from TF_CNN import CNN


today = date.today().strftime('%y%m%d')
tf.set_random_seed(777)

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 100  
# mnist.train.num_examples는 55000이므로 55000/100 => 1 epoch 당 550번 반복
epochs = 2
# 원래 epochs = 15라 15 * 550이지만 속도 때문에 2 * 550으로 변경.


def _main():
    sess = tf.Session()
    
    ### Ensemble ###
    models = []
    num_models = 2
    for m in range(num_models):
        models.append(CNN(sess, "model" + str(m), learning_rate))
    
    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter('./tflogs/log-'+today, graph=sess.graph)
    #writer.add_graph(sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    step = 0
    for epoch in range(epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)
        #total_batch = int(total_batch/10) #속도 때문에 추가. total_batch = 55
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)    
            for m_idx, m in enumerate(models):
                ### run Neural Net
                c, _, s = m.train(batch_xs, batch_ys)
                writer.add_summary(s, global_step=step)
                step += 1
                avg_cost_list[m_idx] += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        saver.save(sess, "./Variables/mnist.ckpt", global_step=(epoch+1)*total_batch)
        
    #_check(sess, models)

def _check(sess, models):
    test_size = len(mnist.test.labels[:1000])
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)
    
    for m_idx, m in enumerate(models):
        print(m_idx, "Accuracy   :", m.get_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        p = m.prediction(mnist.test.images[:1000])
        predictions += p 
        
    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(mnist.test.labels[:1000], axis=1))
    #캐스팅 한 다음 다 더해서 개수만큼 나누기 = 평균이니까 reduce_mean.
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print("Ensemble accuracy : ", sess.run(ensemble_accuracy))

if __name__ == '__main__':
    _main()
