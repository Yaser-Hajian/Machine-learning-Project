import os
import cv2
import shutil
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse


classes = ['dena', 'pars', 'peju206', 'pride', 'samand', 'tiba']
model = load_model('bestCar.h5')

def predict(img_dir, base_dir='/content/data/train'):
    if img_dir.split('.')[-1] == img_dir:
        try:
            img_dirs = glob(f'{img_dir}/*')
            imgs = []
            fnames = []
            for img_path in img_dirs:
                fname = os.path.basename(img_path)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                imgs.append(img)
                fnames.append(fname)
            
            imgs = np.stack(imgs, axis=0)
            outs = model.predict(imgs)
            print('Predictions\n')
            print(10*'-')
            for idx, fname in enumerate(fnames):
                print(fname)
                out = outs[idx]
                pred = np.argmax(out, axis=1)
                name = classes[pred[i]]
                shutil.copy(f'{img_dir}/fname', f'{base_dir}/{name}/{fname}')
                for i, pct, in enumerate(out):
                    print('{}:{:2f}'.format(classes[i], pct*100))

            print('Updating model\n')

            def create_datagen(augment=False):
                train_datagen = ImageDataGenerator(
                                            rescale=1./255,
                                            horizontal_flip=True,
                                            rotation_range=20,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            fill_mode='nearest',
                                            )

                valid_datagen = ImageDataGenerator(
                                                rescale=1./255,
                                                )
                if augment:
                    return train_datagen, valid_datagen

                else:
                    return valid_datagen, valid_datagen

            train_datagen, valid_datagen = create_datagen(augment=True)
            train_gen = train_datagen.flow_from_directory(
                                              '/content/data/train',
                                              target_size=(128, 128),
                                              batch_size=50,
                                              shuffle=True,
                                              class_mode='categorical',
                                              seed=1
                                              )

            valid_gen = valid_datagen.flow_from_directory(
                                                      '/content/data/valid',
                                                      target_size=(128, 128),
                                                      batch_size=50,
                                                      shuffle=False,
                                                      class_mode='categorical',
                                                      seed=1
                                                      )
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [
                ModelCheckpoint('updatedCar.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ]
            model.fit(train_gen, steps_per_epoch=42, epochs=50, validation_data=valid_gen, validation_steps=9, callbacks=callbacks)
        
        except:
            ('folder content is not valid!')
    else:
        try:
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0)
            out = model.predict(img)
            print('Predictions')
            print(10*'-')
            for i, pct, in enumerate(out[0]):
                print('{}{}:{:.2f}%'.format(classes[i], (7-len(classes[i]))*' ', pct*100))
        except:
            ('The file is not supported image file!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='image file path or folder of image directory')
    args = parser.parse_args()

    predict(img_dir=args.img_dir)
"""
python car_prediction.py --img_dir
"""