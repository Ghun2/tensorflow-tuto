import tensorflow as tf

# height = [170, 180, 175, 160]
# shoes = [260, 270, 265, 255]

height = 170.0
shoes = 260.0

# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)


def loss_function(): # 손실 함수
    predicted = height * a + b
    return tf.square(260 - predicted)      # (실제값 - 예측값)^2


opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    # opt.minimize(손실함수, var_list=[a, b])
    opt.minimize(loss_function, var_list=[a, b])
    print(a.numpy(), b.numpy())