import time
import numpy as np
import pandas as pd
from PIL import Image


def change_image_name(df, column):
    return [i + '.jpeg' for i in df[column]]


def convert_images_to_arrays_train(file_path, df):
    lst_imgs = [l for l in df['train_image_name']]
    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])


def save_to_array(arr_name, arr_object):
    return np.save(arr_name, arr_object)


if __name__ == '__main__':
    start_time = time.time()
    labels = pd.read_csv("trainLabels_master_256_v2.csv")
    print("Writing Train Array")
    X_train = convert_images_to_arrays_train('train-resized-256/', labels)
    print(X_train.shape)
    print("Saving Train Array")
    save_to_array('X_train.npy', X_train)
    print("--- %s seconds ---" % (time.time() - start_time))
