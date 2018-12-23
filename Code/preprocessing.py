import os
import sys
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np
import time
import pandas as pd
from PIL import Image
from skimage.transform import rotate

def create_directory(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):

    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0

    for item in dirs:
        img = io.imread(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)

def find_black_images(file_path, df):
    
    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]

def rotate_images(file_path, degrees_of_rotation, lst_imgs):
    
    for l in lst_imgs:
        img = io.imread(file_path + str(l) + '.jpeg')
        img = rotate(img, degrees_of_rotation)
        io.imsave(file_path + str(l) + '_' + str(degrees_of_rotation) + '.jpeg', img)


def mirror_images(file_path, mirror_direction, lst_imgs):

    for l in lst_imgs:
        img = cv2.imread(file_path + str(l) + '.jpeg')
        img = cv2.flip(img, 1)
        cv2.imwrite(file_path + str(l) + '_mir' + '.jpeg', img)


def get_lst_images(file_path):

    return [i for i in os.listdir(file_path) if i != '.DS_Store']

def change_image_name(df, column):
    
    return [i + '.jpeg' for i in df[column]]


def convert_images_to_arrays_train(file_path, df):
    
    lst_imgs = [l for l in df['train_image_name']]

    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])


def save_to_array(arr_name, arr_object):
    
    return np.save(arr_name, arr_object)


if __name__ == '__main__':
    
    crop_and_resize_images(path='train/', new_path='train-resized-256/', cropx=1800, cropy=1800, img_size=256)
    
    start_time = time.time()
    trainLabels = pd.read_csv('labels.csv')

    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan

    trainLabels['black'] = find_black_images('train-resized-256/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)
    trainLabels = pd.read_csv('labels.csv')

    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan

    trainLabels['black'] = find_black_images('train-resized-256/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)

    trainLabels = pd.read_csv("trainLabels_master.csv")

    lst_imgs = get_lst_images('train-resized-256/')

    new_trainLabels = pd.DataFrame({'image': lst_imgs})
    new_trainLabels['image2'] = new_trainLabels.image

    # Remove the suffix from the image names.
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Strip and add .jpeg back into file name
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

    # trainLabels = trainLabels[0:10]
    new_trainLabels.columns = ['train_image_name', 'image']

    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis=1, inplace=True)
    # print(trainLabels.head(100))
    trainLabels = trainLabels.dropna()
    print(trainLabels.shape)

    print("Writing CSV")
    trainLabels.to_csv('trainLabels_master_256_v2.csv', index=False, header=True)

    labels = pd.read_csv("trainLabels_master_256_v2.csv")

    print("Writing Train Array")
    X_train = convert_images_to_arrays_train('train-resized-256/', labels)

    print(X_train.shape)
    
    print("Saving Train Array")
    save_to_array('X_train.npy', X_train)


    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))
