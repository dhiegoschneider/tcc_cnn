#Carga do tensorflow
import tensorflow as tf

#Configurações
alt_imag = 128
larg_imag = 128
batch_size = int(32)
num_classes = 2
epochs = int(3)

#Diretórios
teste = "dados_teste"
treinamento = "dados_treinamento"
validacao: str = "dados_validação"

#Pré-processamento
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory=treinamento,
    target_size=(alt_imag, larg_imag),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    directory=validacao,
    target_size=(alt_imag, larg_imag),
    batch_size=batch_size,
    class_mode='categorical')
