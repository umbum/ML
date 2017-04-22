#coding: utf-8
import tensorflow as tf

const1 = tf.constant(3.0, tf.float32)
const2 = tf.constant(4.0)
const_add = tf.add(const1, const2)

print("node1:", const1, "node2:", const2)
print("node3:", const_add)
#단순히 print로 출력하면 tensorflow 객체가 출력되기 때문에
#내부에 있는 값을 보고 싶다면 sess.run()에 넣어야 한다.
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run((const1, const2)))
print("sess.run(node3): ", sess.run(const_add))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

#matmul은 rank > 2에서만 동작하기 때문에 한번 더 묶었다.
va = tf.Variable([[1, 2, 3]])
vb = tf.Variable([[1, 2, 3]])
result = tf.matmul(va, vb, transpose_b=True)
sess = tf.Session()
#Variable을 사용하기 전에는 꼭 아래 메서드를 호출해줘야 한다.
sess.run(tf.global_variables_initializer())
print(sess.run((va, vb)))
print(sess.run(result))