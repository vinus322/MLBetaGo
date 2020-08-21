import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot = True)

print("훈련 이미지:", mnist.train.images.shape)
print("훈련 라벨:", mnist.train.labels.shape)

print("테스트 이미지:", mnist.test.images.shape)
print("테스트 라벨:", mnist.test.labels.shape)

print("검증 이미지:", mnist.validation.images.shape)
print("검증 라벨:", mnist.validation.labels.shape)

mnist_idx = 1

print('[label]')
print('one-hot vector label = ', mnist.train.labels[mnist_idx])
print('number label = ', np.argmax(mnist.train.labels[mnist_idx]))
print('\n')



# region k-NN algorithm 

# x_data_train : training data 전부의 784개의 픽셀
# y_data_train : training data 전부의 숫자 라벨
# x_data_test : test data 한개 784개의 픽셀
x_data_train = tf.placeholder(tf.float32, [None, 784])
y_data_train = tf.placeholder(tf.float32,[None, 10])
x_data_test = tf.placeholder(tf.float32, [784])
paramK = tf.placeholder(tf.int32)

# distance : K-Nearest Neighbor - 오차 거리를 구한다(유클리드 거리)
distance = tf.reduce_sum(tf.abs(tf.add(x_data_train, tf.negative(x_data_test))), reduction_indices=1)

# nearest k points
_, top_k_indices = tf.nn.top_k(tf.negative(distance), k=paramK) #가장 가까운 이미지 K개의 인덱스를 가져온다. 
top_k_label = tf.gather(y_data_train, top_k_indices)  #Index에 해당하는 숫자 라벨를 가져온다.

sum_up_predictions = tf.reduce_sum(top_k_label, axis=0) #공통되는 숫자 라벨의 개수를 더한다. 
prediction = tf.argmax(sum_up_predictions) #가장 가까운 숫자값을 구한다.

# endregion




print("--------- TEST -----------")
# 학습을 위해서 기준 데이터들의 개수를 Batch size로 잡음
# 1000개에 대해 테스트 해본다
train_images, train_labels = mnist.train.next_batch(3000)
test_images, test_labels = mnist.test.next_batch(1000)


# Tensorflow 세션 실행
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


# 정확도를 초기화
accuracy = 0
confusion_matrix = [[0 for i in range(10)] for j in range(10)]


for i in range(len(test_images)):

    # k-NN 알고리즘 수행 예측된 숫자를 구한다.
    pre_label = sess.run(prediction, feed_dict={x_data_train: train_images, y_data_train: train_labels, x_data_test: test_images[i, :], paramK :  4})
    real_label = np.argmax(test_labels[i])

    #nn_index의 라벨값과 실제 라벨값을 비교한다.
    print("테스트 횟수 : ", i)
    print("실제값 : ", real_label)
    print("예측값 : ", pre_label)

    confusion_matrix[real_label][pre_label] +=1

    # 예측도 파악
    # KNN은 비교 데이터에서 가장 가까운 것을 찾는 것이므로 매번 확률을 갱신해야한다(가중치를 찾는게 아니다)
    # 가장 가까운 것이 무엇이 될지 모름
    if pre_label == real_label :
        accuracy += 1./len(test_images)
    #else :
        # 잘못 예측한 이미지 보기
        # plt.figure(figsize = (5, 5))
        # image = np.reshape(test_images[i], [28, 28])
        # plt.imshow(image, cmap = 'Greys')
        # plt.show()



print("--------- RESULT -----------")

print("예측 정확도 : ", round(accuracy*100, 2)," %")
print("The resulting confusion matrix")
# confusion matrix 생성
for i in range(10):
    print(confusion_matrix[i])