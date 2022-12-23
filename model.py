import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

def inceptionv2(image_size=80):
    inp = tf.keras.Input(shape=(1, image_size, image_size)) # (80, 80, 1)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation='relu')(inp) # (74, 74, 64)
    x = tf.keras.layers.BatchNormalization()(x) # (74, 74, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # (37, 37, 64)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(6,6), activation='relu')(x) # (32, 32, 128)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # (16, 16, 128)

    # INCEPTION LAYERS 1
    tower_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1,1), padding='same', activation='relu')(x) # (16, 16, 96)
    tower_1 = tf.keras.layers.BatchNormalization()(tower_1)
    tower_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(tower_1) # (16, 16, 128)
    tower_1 = tf.keras.layers.BatchNormalization()(tower_1)
    tower_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(x) # (16, 16, 16)
    tower_2 = tf.keras.layers.BatchNormalization()(tower_2)
    tower_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu')(tower_2) # (16, 16, 16)
    tower_2 = tf.keras.layers.BatchNormalization()(tower_2)
    tower_3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x) # (16, 16, 128)
    tower_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(tower_3) # (16, 16, 32)
    tower_3 = tf.keras.layers.BatchNormalization()(tower_3)
    tower_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(x) # (16, 16, 64)
    tower_4 = tf.keras.layers.BatchNormalization()(tower_4)
    tower_5 = tf.keras.layers.Concatenate(axis=1)([tower_1, tower_2, tower_3, tower_4]) # (16, 16, 240)

    # INCEPTION LAYERS 2
    tower_6 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(tower_5) # (16, 16, 128)
    tower_6 = tf.keras.layers.BatchNormalization()(tower_6)
    tower_6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu')(tower_6) # (16, 16, 192)
    tower_6 = tf.keras.layers.BatchNormalization()(tower_6)
    tower_7 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(tower_5) # (16, 16, 32)
    tower_7 = tf.keras.layers.BatchNormalization()(tower_7)
    tower_7 = tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), padding='same', activation='relu')(tower_7) # (16, 16, 96)
    tower_7 = tf.keras.layers.BatchNormalization()(tower_7)
    tower_8 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(tower_5) # (16, 16, 96)
    tower_8 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_8) # (16, 16, 64)
    tower_8 = tf.keras.layers.BatchNormalization()(tower_8)
    tower_9 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(tower_5) # (16, 16, 128)
    tower_9 = tf.keras.layers.BatchNormalization()(tower_9)
    tower_10 = tf.keras.layers.Concatenate(axis=1)([tower_6, tower_7, tower_8, tower_9]) # (16, 16, 480)
    
    tower_11 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(tower_10) # (8, 8, 480)
    
    # INCEPTION LAYERS 3
    tower_12 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1,1), padding='same', activation='relu')(tower_11) # (8, 8, 96)
    tower_12 = tf.keras.layers.BatchNormalization()(tower_12)
    tower_12 = tf.keras.layers.Conv2D(filters=208, kernel_size=(3,3), padding='same', activation='relu')(tower_12) # (8, 8, 208)
    tower_12 = tf.keras.layers.BatchNormalization()(tower_12)
    tower_13 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(tower_11) # (8, 8, 16)
    tower_13 = tf.keras.layers.BatchNormalization()(tower_13)
    tower_13 = tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='relu')(tower_13) # (8, 8, 48)
    tower_13 = tf.keras.layers.BatchNormalization()(tower_13)
    tower_14 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(tower_11) # (8, 8, 480)
    tower_14 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_14) # (8, 8, 64)
    tower_14 = tf.keras.layers.BatchNormalization()(tower_14)
    tower_15 = tf.keras.layers.Conv2D(filters=192, kernel_size=(1,1), padding='same', activation='relu')(tower_11) # (8, 8, 192)
    tower_15 = tf.keras.layers.BatchNormalization()(tower_15)
    tower_16 = tf.keras.layers.Concatenate(axis=1)([tower_12, tower_12, tower_14, tower_15]) # (8, 8, 512)

    # INCEPTION LAYERS 4
    tower_17 = tf.keras.layers.Conv2D(filters=112, kernel_size=(1,1), padding='same', activation='relu')(tower_16) # (8, 8, 112)
    tower_17 = tf.keras.layers.Conv2D(filters=224, kernel_size=(3,3), padding='same', activation='relu')(tower_17) # (8, 8, 224)
    tower_18 = tf.keras.layers.Conv2D(filters=24, kernel_size=(1,1), padding='same', activation='relu')(tower_16) # (8, 8, 24)
    tower_18 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu')(tower_18) # (8, 8, 64)
    tower_19 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(tower_16) # (8, 8, 512)
    tower_19 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_19) # (8, 8, 64)
    tower_20 = tf.keras.layers.Conv2D(filters=160, kernel_size=(1,1), padding='same', activation='relu')(tower_16) # (8, 8, 160)
    tower_21 = tf.keras.layers.Concatenate(axis=1)([tower_17, tower_18, tower_19, tower_20]) # (8, 8, 512)

    # INCEPTION LAYERS 5
    tower_22 = tf.keras.layers.Conv2D(filters=144, kernel_size=(1,1), padding='same', activation='relu')(tower_21) # (8, 8, 144)
    tower_22 = tf.keras.layers.Conv2D(filters=288, kernel_size=(3,3), padding='same', activation='relu')(tower_22) # (8, 8, 288)
    tower_23 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(tower_21) # (8, 8, 32)
    tower_23 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu')(tower_23) # (8, 8, 64)
    tower_24 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(tower_21) # (8, 8, 512)
    tower_24 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_24) # (8, 8, 64)
    tower_25 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(tower_21) # (8, 8, 128)
    tower_26 = tf.keras.layers.Concatenate(axis=1)([tower_22, tower_23, tower_24, tower_25]) # (8, 8, 544)
    
    pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))(tower_26) # (4, 4, 544)
    
    # INCEPTION LAYERS 6
    tower_27 = tf.keras.layers.Conv2D(filters=144, kernel_size=(1,1), padding='same', activation='relu')(pool) # (4, 4, 144)
    tower_27 = tf.keras.layers.Conv2D(filters=288, kernel_size=(3,3), padding='same', activation='relu')(tower_27) # (4, 4, 288)
    tower_28 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(pool) # (4, 4, 32)
    tower_28 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu')(tower_28) # (4, 4, 64)
    tower_29 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(pool) # (4, 4, 544)
    tower_29 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu')(tower_29) # (4, 4, 64)
    tower_30 = tf.keras.layers.Conv2D(filters=112, kernel_size=(1,1), padding='same', activation='relu')(pool) # (4, 4, 112)
    tower_31 = tf.keras.layers.Concatenate(axis=1)([tower_27, tower_28, tower_29, tower_30]) # (4, 4, 528)


    output_1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(tower_31) # (2, 2, 528)
    
    output_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', activation='relu')(output_1) # (2, 2, 256)
    output_1 = tf.keras.layers.Flatten()(output_1) # (1024)
    output_1 = tf.keras.layers.Dense(units=1024, activation='relu')(output_1) # (1024)
    output_1 = tf.keras.layers.Dropout(rate=0.5)(output_1) # (1024)
    output_1 = tf.keras.layers.Dense(units=1024, activation='relu')(output_1) # (1024)
    output_1 = tf.keras.layers.Dropout(rate=0.5)(output_1) # (1024)
    
    y1 = tf.keras.layers.Dense(units=5, activation='softmax',name='y1')(output_1) # (5)
    
    model = tf.keras.Model(inputs=inp, outputs=y1)
    
    return model



if __name__ == '__main__':
    
    model = inceptionv2(80)
