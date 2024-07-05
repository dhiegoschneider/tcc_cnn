import tensorflow as tf
import pre_processamento_imag as process
import arquitetura_cnn as cnn
import matplotlib.pyplot as plt

# Debug prints
print(f'Number of training samples: {process.train_generator.samples}')
print(f'Number of validation samples: {process.validation_generator.samples}')
print(f'Batch size: {process.batch_size}')
print(f'Epochs: {process.epochs}')

for epoch in range(process.epochs):
    history = cnn.modelo.fit(
        process.train_generator,
        steps_per_epoch=process.train_generator.samples // process.batch_size,
        epochs=1,
        validation_data=process.validation_generator,
        validation_steps=process.validation_generator.samples // process.batch_size
    )

    logs = history.history

    if logs:
        for key, value in logs.items():
            print(f'{key}: {value}')
    else:
        print('No logs available')

# Métricas
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Número de épocas
epochs = range(1, len(train_loss) + 1)

#Gráfico Perda
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#Gráfico Acurácia
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
