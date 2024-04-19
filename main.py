from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
import imghdr
import cv2
import os

images_dir = 'images'

def main():
    for image_class in os.listdir(images_dir):
        for image in os.listdir(os.path.join(images_dir, image_class)):
            image_path = os.path.join(images_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in ['jpeg','jpg', 'bmp', 'png']:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))

    images = tf.keras.utils.image_dataset_from_directory(images_dir)

    images = images.map(lambda x,y: (x/255, y))

    train_size = int(len(images) * .4)
    val_size = int(len(images) * .3)
    test_size = int(len(images) * .3)

    train = images.take(train_size)
    val = images.skip(train_size).take(val_size)
    test = images.skip(train_size + val_size).take(test_size)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print('Precision', pre.result())
    print('Recall', re.result())
    print('Accuracy', acc.result())

    model.save(os.path.join('models', 'classifier.keras'))


if __name__ == '__main__':
    main()
