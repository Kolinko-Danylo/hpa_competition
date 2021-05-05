import io
import os
import requests
import pathlib
import gzip
import imageio
import pandas as pd
from tqdm import tqdm
import urllib.request
import cv2

def tif_gzip_to_png(tif_path):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. for working in local work station
    '''
    png_path = pathlib.Path(tif_path.replace('.tif.gz', '.png'))
    tf = gzip.open(tif_path).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(png_path, img)


def download_and_convert_tifgzip_to_png(img_base):

    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    '''
    save_dir = os.path.join('/common/danylokolinko/publichpa', 'test')
    colors = ['blue', 'red', 'green', 'yellow']
    try:
        for color in colors:
            url = f'{img_base}_{color}.tif.gz'
            save_path = os.path.join(save_dir, f'{os.path.basename(img_base)}_{color}.png')

            r = requests.get(url)
            while r.status_code != 200:
                print(r.status_code)
                r = requests.get(url)

            f = io.BytesIO(r.content)
            tf = gzip.open(f).read()
            img = imageio.imread(tf, 'tiff')
            img = cv2.resize(img, (1024, 1024))
            imageio.imwrite(save_path, img)

            # download_and_convert_tifgzip_to_png(img_url, save_path)
            # print(f'Downloaded {img_url} as {save_path}')
    except:
        print(f'failed to download: {img_base}')



# All label names in the public HPA and their corresponding index.
all_locations = dict({
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
})


def add_label_idx(df, all_locations):
    '''Function to convert label name to index
    '''
    df["Label_idx"] = None
    for i, row in df.iterrows():
        labels = row.Label.split(',')
        idx = []
        for l in labels:
            if l in all_locations.keys():
                idx.append(str(all_locations[l]))
        if len(idx) > 0:
            df.loc[i, "Label_idx"] = "|".join(idx)

        print(df.loc[i, "Label"], df.loc[i, "Label_idx"])
    return df

def down(row):
    save_dir = os.path.join('/common/danylokolinko/publichpa', 'test')
    colors = ['blue', 'red', 'green', 'yellow']

    try:
        img = row
        # print(img)
        for color in colors:
            img_url = f'{img}_{color}.jpg'

            lcl_img = f'{os.path.basename(img)}_{color}.jpg'
            save_path = os.path.join(save_dir, lcl_img)
            # if os.path.exists((save_path)):
            #     continue

            img_bytes = requests.get(img_url)
            while img_bytes.status_code != 200:
                img_bytes = requests.get(img_url)
            img_bytes = img_bytes.content
            # img_name = img_url.split('/')[3]
            # img_name = f'{img_name}.jpg'
            with open(save_path, 'wb') as img_file:
                img_file.write(img_bytes)
                print(f'{img_url} was downloaded...')
            # print(img_url)
            # print(save_path)
            # print(img_url)

            # print('here')


            # img_data = imageio.core.urlopen(img_url).read()
            # img_data = requests.get(img_url).content
            # print(img_data)

            # image = imageio.imread(img_data, '.jpg')

            # imageio.imwrite(save_path, image)
            # with open(save_path, 'wb') as handler:
            #     handler.write(img_data)

            #             download_and_convert_tifgzip_to_png(img_url, save_path)
            # print(f'Downloaded {img_url} as {save_path}')
    except ValueError:
        print(f'failed to download: {img_url}')
        # raise()



public_hpa_df = pd.read_csv('~/hpa/public_hpa/kaggle_2021.tsv')
# Remove all images overlapping with Training set
public_hpa_df = public_hpa_df[public_hpa_df.in_trainset == False]

# Remove all images with only labels that are not in this competition
public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

colors = ['blue', 'red', 'green', 'yellow']
celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30', 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)]
# len(public_hpa_df), len(public_hpa_df_17)

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# for i, row in public_hpa_df_17[0:5].iterrows():
#     try:
#         img = row.Image
#         for color in colors:
#             img_url = f'{img}_{color}.tif.gz'
#             save_path = os.path.join(save_dir, f'{os.path.basename(img)}_{color}.png')
#             download_and_convert_tifgzip_to_png(img_url, save_path)
#             print(f'Downloaded {img_url} as {save_path}')
#     except:
#         print(f'failed to download: {img}')
save_dir = os.path.join('/common/danylokolinko/publichpa/HPA-Challenge-2021-trainset-extra/')

lst = os.listdir(save_dir)

import numpy as np
def resize(subpath):
    save_dir = os.path.join('/common/danylokolinko/publichpa/HPA-Challenge-2021-trainset-extra/')

    full_path = os.path.join(save_dir, subpath)
    cur_img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    if cur_img.max() > 255:
        cur_img = cur_img // 256
    #     print(cur_img.min())

    cur_img = cur_img.astype(np.uint8)
    image_size = 1024
    cur_img = cv2.resize(cur_img, (image_size, image_size))
    cv2.imwrite(full_path, cur_img)
# ppp = pd.read_csv('~/hpa/public_hpa/sample.csv')
# ids = public_hpa_df_17.Image.values
# for id in tqdm(ids):
#     download_and_convert_tifgzip_to_png(id)
with ProcessPoolExecutor(20) as executor:
    results = list(tqdm(executor.map(resize, lst), total=len(lst)))





