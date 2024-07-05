#Carga de dados
import pre_processamento_imag as dados

#Carga do tensorflow e keras
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam

#Modelo
modelo = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(dados.alt_imag, dados.larg_imag, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(dados.num_classes, activation='softmax')
])

modelo.compile(optimizer=Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
