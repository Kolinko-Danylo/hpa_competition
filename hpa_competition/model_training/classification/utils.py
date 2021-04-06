# convert classification mask image to run length encoding
MAX_GREEN = 64  # filter out dark green cells
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import cv2
import torch
import os
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

def load_RGBY_image(root_path, image_id, train_or_test='train', image_size=None, yellow_channel=False):
    red = read_img(root_path, image_id, "red", train_or_test, image_size)
    green = read_img(root_path, image_id, "green", train_or_test, image_size)
    blue = read_img(root_path, image_id, "blue", train_or_test, image_size)
    if not yellow_channel:
        return (np.array([red, green, blue]))
    yellow = read_img(root_path, image_id, "yellow", train_or_test, image_size)

    return (np.array([red, yellow, green, blue]))

def read_img(root_path, image_id, color, train_or_test='train', image_size=None):
    filename = os.path.join(root_path, train_or_test, f'{image_id}_{color}.png')

    # filename = f'{root_path}/{train_or_test}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    # img_max = 255 if (img.max() <= 255) else img.max()
    # img = (img / img_max).astype('uint8')
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img


def get_cam(model, ori_image, scale):
    image = copy.deepcopy(ori_image)
    # flipped_image = image.flip(-1)

    # images = torch.stack([image, flipped_image])

    _, features = model(image, with_cam=True)

    cams = F.relu(features)
    # cams = cams[0] + cams[1].flip(-1)

    return cams


def build_image_names(image_id: str, dir_path: str) -> list:
    # mt is the mitchondria
    mt = os.path.join(dir_path, f'{image_id}_red.png')

    # er is the endoplasmic reticulum
    er = os.path.join(dir_path, f'{image_id}_yellow.png')

    # nu is the nuclei
    nu = os.path.join(dir_path, f'{image_id}_blue.png')

    return [mt], [er], [nu], [[mt], [er], [nu]]


def get_rles_from_mask(image_id, class_id):
    mask = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
    if class_id != '18':
        green_img = read_img(image_id, 'green')
    rle_list = []
    mask_ids = np.unique(mask)
    for val in mask_ids:
        if val == 0:
            continue
        binary_mask = np.where(mask == val, 1, 0).astype(bool)
        if class_id != '18':
            masked_img = green_img * binary_mask
            # print(val, green_img.max(),masked_img.max())
            if masked_img.max() < MAX_GREEN:
                continue
        rle = coco_rle_encode(binary_mask)
        rle_list.append(rle)
    return rle_list, mask.shape[0], mask.shape[1]


def coco_rle_encode(mask):
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


# mmdet custom dataset generator
def mk_mmdet_custom_data(image_id, class_id):
    rles, height, width = get_rles_from_mask(image_id, class_id)
    if len(rles) == 0:
        return {
            'filename': image_id + '.jpg',
            'width': width,
            'height': height,
            'ann': {}
        }
    rles = mutils.frPyObjects(rles, height, width)
    bboxes = mutils.toBbox(rles)
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return {
        'filename': image_id + '.jpg',
        'width': width,
        'height': height,
        'ann':
            {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.zeros(len(bboxes)),  # dummy data.(will be replaced later)
                'masks': rles
            }
    }


# print utility from public notebook
def print_masked_img(path, image_id, mask):
    img = load_RGBY_image(path, image_id, train_or_test='test', image_size=mask.size()[-1]).transpose([1, 2, 0])
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 20, 1)
    plt.imshow(img)
    plt.axis('off')

    for i in range(19):
        plt.subplot(1, 20, 1 + i)
        plt.imshow(img)
        plt.imshow(mask[i], alpha=0.6)
        plt.axis('off')
    plt.show()


# image loader, using rgb only here


# make annotation helper called multi processes
def mk_ann(idx):
    image_id = df.iloc[idx].ID
    class_id = df.iloc[idx].Label
    anno = mk_mmdet_custom_data(image_id, class_id)
    img = load_RGBY_image(image_id, train_or_test)
    cv2.imwrite(f'{img_dir}/{image_id}.jpg', img)
    return anno, idx, image_id

import os
import pandas as pd

def get_df(path, train=True):
    df = pd.read_csv(os.path.join(path, 'cell_df.csv'))
    if not train:
        return df

    #TODO: how to use small samples

    df = df.loc[df.size1 >= 224]
    df = df.loc[df.size2 >= 224]

    labels = [str(i) for i in range(19)]
    for x in labels:
        df[x] = df['image_labels'].apply(lambda r: int(x in r.split('|')))

    # df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    # df = df.reset_index(drop=True)
    dfs = df[['image_id' ] + labels].drop_duplicates()

    nfold = 5
    seed = None

    y = dfs[labels].values
    X = dfs[['image_id']].values
    # print(X)

    dfs['fold'] = np.nan

    mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
    for i, (_, test_index) in enumerate(mskf.split(X, y)):
        dfs.iloc[test_index, -1] = i

    dfs['fold'] = dfs['fold'].astype('int')
    dfs['is_valid'] = False
    dfs.loc[dfs['fold'] == 0, 'is_valid'] = True

    train_ = dfs.loc[~dfs.is_valid, 'image_id']
    val_ = dfs.loc[dfs.is_valid, 'image_id']


    # ln = int(0.8 * df.shape[0])
    # while df.iloc[ln].image_id == df.iloc[ln - 1].image_id:
    #     ln -= 1

    train_df = df.loc[df.image_id.isin(train_)].reset_index(drop=True)
    val_df = df.loc[df.image_id.isin(val_)].reset_index(drop=True)

    # return train_df.sample(frac=0.1, random_state=42).reset_index(drop=True), val_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    return train_df, val_df





def get_df_cam(path, train=True):
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    labels = [str(i) for i in range(19)]
    for x in labels:
        df[x] = df['Label'].apply(lambda r: int(x in r.split('|')))

    nfold = 5
    seed = None
    y = df[labels].values
    X = df[['ID']].values
    df['fold'] = np.nan

    mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
    for i, (_, test_index) in enumerate(mskf.split(X, y)):
        df.iloc[test_index, -1] = i

    df['fold'] = df['fold'].astype('int')
    df['is_valid'] = False
    df.loc[df['fold'] == 0, 'is_valid'] = True


    return df


