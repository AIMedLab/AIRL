from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd

raw_data_dir = 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
time_domain_list = [str(i) for i in list(range(1930, 2014, 1))]
m_image_name_list = os.listdir(raw_data_dir + '/M')
f_image_name_list = os.listdir(raw_data_dir + '/F')
m_data_by_year = defaultdict(list)
f_data_by_year = defaultdict(list)


def create_processed_folder():
    for i in time_domain_list:
        if not os.path.exists(i):
            os.makedirs(i + '/' + 'M')
            os.makedirs(i + '/' + 'F')


def convert_img_2_numpy(img):
    img = Image.open(img)
    img = img.resize((32, 32), Image.ANTIALIAS)
    img = np.array(img, np.float32)
    img = img / 255
    img = (img - 0.5) / 0.5
    return img


def process_data():
    for img_name in tqdm(m_image_name_list):
        year = img_name.split('_')[0]
        if int(year) > 1929:
            img = convert_img_2_numpy(raw_data_dir + '/M/' + img_name)
            img_name_new = img_name[5:-4]
            with open(year + '/M/' + img_name_new + '.npy', 'wb') as f:
                np.save(f, img)
            m_data_by_year[year].append(img_name_new)
    for img_name in tqdm(f_image_name_list):
        year = img_name.split('_')[0]
        if int(year) > 1929:
            img = convert_img_2_numpy(raw_data_dir + '/F/' + img_name)
            img_name_new = img_name[5:-4]
            with open(year + '/F/' + img_name_new + '.npy', 'wb') as f:
                np.save(f, img)
            f_data_by_year[year].append(img_name_new)
    return m_data_by_year, f_data_by_year


def split_data():
    metadata = []
    for year in time_domain_list:
        img_list = m_data_by_year[year]
        img_train, img_test = train_test_split(img_list, test_size=0.1, random_state=42)
        img_train, img_val = train_test_split(img_train, test_size=0.1, random_state=42)
        for img in img_train:
            metadata.append([img, year, 'M', 'train'])
        for img in img_val:
            metadata.append([img, year, 'M', 'val'])
        for img in img_test:
            metadata.append([img, year, 'M', 'test'])

        img_list = f_data_by_year[year]
        img_train, img_test = train_test_split(img_list, test_size=0.1, random_state=42)
        img_train, img_val = train_test_split(img_train, test_size=0.1, random_state=42)
        for img in img_train:
            metadata.append([img, year, 'F', 'train'])
        for img in img_val:
            metadata.append([img, year, 'F', 'val'])
        for img in img_test:
            metadata.append([img, year, 'F', 'test'])

    metadata = pd.DataFrame(metadata, columns=['img', 'year', 'label', 'dataset'])
    metadata.to_csv('metadata.csv', index=False)


create_processed_folder()
process_data()
split_data()
