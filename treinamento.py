import arquitetura_cnn as cnn
import pre_processamento_imag as process

history = cnn.modelo.fit(process.train_generator,
                         steps_per_epoch=process.train_generator.samples // process.batch_size,
                         epochs=process.epochs,
                         validation_data=process.validation_generator,
                         validation_steps=process.validation_generator.samples / process.batch_size)
