import tensorflow as tf

#X and Y data
#x_train = [1, 2, 3]
#y_train = [1, 2, 3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight') #[1] : 랭크가 1
b = tf.Variable(tf.random_normal([1]), name='bias') #[1] : 랭크가 1

# Our hypothesis XW+b
hypothesis = x * W + b

#cost(W, b) = 1/m * ∑(H(xi) - yi)^2
cost = tf.reduce_mean(tf.square(hypothesis - y))

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())


for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], 
        feed_dict= {x : [1, 2, 3], y:[1, 2, 3]})

    if step % 20 == 0 :
        print(step, cost_val, W_val, b_val)