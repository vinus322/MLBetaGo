import tensorflow as tf


#H(x) = Wx+b
#X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight') #[1] : 랭크가 1
b = tf.Variable(tf.random_normal([1]), name='bias') #[1] : 랭크가 1

# Our hypothesis XW+b
hypothesis = x_train * W + b

#cost(W, b) = 1/m * ∑(H(xi) - yi)^2
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 ==0 :
        print(step, sess.run(cost), sess.run(W), sess.run(b))


#학습이 진행될수록 W는 1로 b는 0에 가까워진다. 