import time
import numpy as np
import pandas as pd
from PIL import Image


def find_black_images(file_path, df):
    
    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]


if __name__ == '__main__':
    start_time = time.time()
    trainLabels = pd.read_csv('labels.csv')

    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan

    trainLabels['black'] = find_black_images('train-resized-256/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))