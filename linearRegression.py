import tensorflow as tf

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)


def loss_function(a, b):
    predic_y = train_x * a + b
    return tf.keras.losses.mse(train_y, predic_y)  # (예측1 - 실제1)^2 + (예측2 - 실제2)^2 ...


opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(2400):
    # opt.minimize(손실함수, var_list=[a, b])
    opt.minimize(lambda: loss_function(a, b), var_list=[a, b])
    print(a.numpy(), b.numpy())
