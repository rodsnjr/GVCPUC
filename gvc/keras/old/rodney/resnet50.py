import keras

train_path = "C:/Imagens Treinamento/"
test_path = "C:/Imagens Treinamento/Test/"

def load_test_image():
    img_path = 'C:/Imagens Treinamento/Doors/030151114301md.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def image_generators():
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        classes=['Doors', 'Indoors'],
        target_size=(32, 32),
        class_mode='sparse'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        test_path,
        classes=['Doors', 'Indoors'],
        target_size=(32, 32),
        class_mode='sparse'
    )

    return train_generator, validation_generator

def top_layer():
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    
    input_shape = (32, 32, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    return model

def checkpoints():
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    
    import os
    my_dir = os.path.dirname(__file__)
    filepath= my_dir + "/checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping()
    callbacks_list = [checkpoint, early_stop]

    return callbacks_list


def resnet50():
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(include_top=False, input_shape=(224,224,3), weights='imagenet')
    return model

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def show_output(model):
    x = load_test_image()
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3))

# model = top_layer()
base_model = resnet50()

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

"""
train_generator, validation_generator = image_generators()
callbacks = checkpoints()

model.fit_generator(
        train_generator,
        2000,
        verbose=1,
        callbacks=callbacks,
        epochs=10,
        validation_steps=800,
        validation_data=validation_generator)
    
evaluation = model.evaluate_generator(validation_generator, 32)
print("Evaluation - {}".format(evaluation))
"""