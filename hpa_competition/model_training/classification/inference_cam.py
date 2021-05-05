import os
import sys
import  time
sys.path.append('/home/danylokolinko/hpa')
import copy
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio
from tqdm import tqdm
import math

from hpa_competition.model_training.common.augmentations import get_transforms

from hpa_competition.PuzzleCAM.core.networks import *
from hpa_competition.PuzzleCAM.core.datasets import *

from hpa_competition.PuzzleCAM.tools.general.io_utils import *
from hpa_competition.PuzzleCAM.tools.general.time_utils import *
from hpa_competition.PuzzleCAM.tools.general.json_utils import *

from hpa_competition.PuzzleCAM.tools.ai.log_utils import *
from hpa_competition.PuzzleCAM.tools.ai.demo_utils import *
from hpa_competition.PuzzleCAM.tools.ai.optim_utils import *
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import *
from hpa_competition.PuzzleCAM.tools.ai.evaluate_utils import *

from hpa_competition.PuzzleCAM.tools.ai.augment_utils import *
from hpa_competition.PuzzleCAM.tools.ai.randaugment import *
from hpa_competition.model_training.common.datasets import HPADatasetCAMTest, HPADatasetCAM

from hpa_competition.model_training.classification.utils import  get_df_cam, build_image_names, read_img, get_cam, load_RGBY_image, print_masked_img, encode_binary_mask
import torch

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from concurrent.futures import ThreadPoolExecutor

from hpa_competition.model_training.classification.utils import load_RGBY_image
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import one_hot_embedding


# with open(os.path.join('config', 'cam_avenga.yaml')) as config_file:
#     config = yaml.full_load(config_file)
#
# # print(config['model']['model_path'])
#
#     def cust(id):
#
#         def load_shit(id):
#             cache_path = '/common/danylokolinko/hpa_cache/'
#             nuc_load = lambda image_id: os.path.join(cache_path, 'nuc', image_id)
#             cell_load = lambda image_id: os.path.join(cache_path, 'cell', image_id)
#             nuc_segmentations = np.load(f'{nuc_load(id)}.npy')
#             cell_segmentations = np.load(f'{cell_load(id)}.npy')
#             pth = os.path.join(cache_path, f'cams/efficientnet-b4-2021-04-21-00-18-21')
#             CAMs = torch.load(os.path.join(pth, f'{id}.npy'), map_location=torch.device('cpu')).type(
#                 torch.FloatTensor)
#             return nuc_segmentations, cell_segmentations, CAMs
#
#         nuclei_mask, cell_mask, CAMs = load_shit(id)
#         cell_mask = torch.from_numpy(cell_mask.astype(np.int32))
#         cell_masks = cell_mask #.cuda()
#
#         image_rles = []
#
#         CAMS = resize_for_tensors(CAMs.unsqueeze(0), cell_masks.shape)[0]
#         cur_sz = cell_masks.amax(dim=(0, 1)).item()
#         cur_cams = CAMS.unsqueeze(0) #1, 19, h, w
#         cur_cell_masks = cell_masks.unsqueeze(0).expand(cur_sz, -1, -1, -1)
#         cur_cell_masks = (cur_cell_masks == torch.arange(1, cur_sz+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
#         area = cur_cell_masks.sum(dim=(2, 3))
#         # print(cur_cell_masks.shape)
#
#         summa = (cur_cams * (cur_cell_masks)).sum(dim=(2, 3))
#
#         # y_pred = self.sigm(summa)
#
#         y_pred = nn.Sigmoid()(torch.div(summa, area))
#         y_pred = y_pred.cpu().numpy()
#         # print(y_pred)
#         # print(y_pred.shape)
#         cur_cell_masks = cur_cell_masks.squeeze(1)
#         for i in range(cur_sz):
#             image_rles.append(encode_binary_mask(cur_cell_masks[i].cpu().numpy()).decode("utf-8"))
#
#         res = [y_pred, (image_rles)]
#         return res



class Inferrer:
    def __init__(self, config, dl, df, trainset):
        self.sigm__ = lambda x: 1 / (1 + math.exp(-x))

        self.config = config
        self.df = df
        self.dl = dl
        self.trainset = trainset
        self.train_str = 'train' if trainset else 'test'
        self.sigm = nn.Sigmoid()
        self.model_path = self.config['model']['model_path']
        self.sigmoid = lambda arr: 1 / (1 + np.exp(-arr))
        self.relu = lambda arr: np.maximum(0, arr)
        self.use_amp = self.config['use_amp']
        self.img_dir = os.path.join(config[self.train_str]['path'], self.train_str)
        # self.mask_dir = self.config[self.train_str]['mask_path']
        self.exp_name = self.model_path.split('/')[-1].split('.')[0]
        self.num_classes = self.config['model']['classes']
        print(self.exp_name)

        self.cache_path = '/common/danylokolinko/hpa_cache/'
        self.pth = os.path.join(self.cache_path, f'cams/{self.exp_name}')

        self.best_val_ap = None

        self.model = Classifier(config['model'])
        self.model.cuda()
        self.model.eval()
        self.nuc_load = lambda image_id: os.path.join(self.cache_path, 'nuc', image_id)
        self.cell_load = lambda image_id: os.path.join(self.cache_path, 'cell', image_id)



        use_gpu = '0'

        the_number_of_gpu = len(use_gpu.split(','))
        if the_number_of_gpu > 1:
           self. model = nn.DataParallel(self.model)

        load_model(self.model, self.model_path, parallel=the_number_of_gpu > 1)

        self.segmentator = cellsegmentator.CellSegmentator(
            config['segm_model']['nuclei_path'],
            config['segm_model']['cell_path'],
            scale_factor=0.25,
            device='cuda',
            padding=True,
            multi_channel_model=True
        )

    def save_predictions(self, save_cams=True, save_masks=False, viz=False, infer=False):
        # max_logits = [0 for i in range(self.config['model']['classes'])]

        # dir_path = '/common/danylokolinko/hpa'
        # train = False
        # img_dir = os.path.join(dir_path, ('train' if train else 'test'))

        # glob_preds = []
        with torch.no_grad():
            for step, (image, empty_label, image_id, empty_cell, empty_nuclei) in (enumerate(tqdm(self.dl))):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    image = image.cuda()
                    CAMs, preds = get_cam(self.model, image, 1)
                    CAMs = CAMs.cpu()

                    if infer:
                        pass
                    if not os.path.exists(self.pth):
                        os.makedirs(self.pth)
                    if save_cams:
                        for j in range(len(image_id)):
                            torch.save(CAMs[j], os.path.join(self.pth, f'{image_id[j]}.npy'))
                    if save_masks:
                        images = build_image_names(image_id, self.img_dir)[-1]
                        nuc_segmentations = self.segmentator.pred_nuclei(images[2])
                        cell_segmentations = self.segmentator.pred_cells(images)
                        for j in range(len(image_id)):
                            nuc_seg, cell_seg = label_cell(nuc_segmentations[j], cell_segmentations[j])
                            np.save(self.nuc_load(image_id[j]), nuc_seg)
                            np.save(self.cell_load(image_id[j]), cell_seg)
                    if viz:
                        hrcams = nn.Sigmoid()(resize_for_tensors(CAMs.type(torch.FloatTensor), (512, 512)))
                        # print(torch.mean(hrcams, dim=(2, 3)).max())
                    # print(hrcams.shape)
                        for i, id in enumerate(image_id):
                            print_masked_img(self.config[self.train_str]['path'], self.train_str, id, hrcams[i])





                # params_lst = [(nuc_segmentations[i], cell_segmentations[i], CAMs[i]) for i in
                #               range(len(cell_segmentations))]
                # with Pool(16) as executor:
                #     results = executor.map(cust, params_lst)
                #
                # glob_preds += list(results)

    def from_preds2string(self, preds):
        image_preds = preds[0]
        image_rles = preds[1]
        strt = " ".join(
            [f'{class_idx} {image_preds[step][class_idx]} {rle}' for step, rle in enumerate(image_rles) for
             class_idx in range(self.num_classes)])
        return strt

    # def from_preds2string(preds):
    #     image_preds = preds[0]
    #     image_rles = preds[1]
    #     strt = " ".join(
    #         [f'{class_idx} {sigm(image_preds[step][class_idx])} {rle}' for step, rle in enumerate(image_rles) for
    #          class_idx in range(19)])
    #     return strt


    def infer(self, masktype):
        with torch.no_grad():

            self.masktype=masktype
            ids = self.df.ID.tolist()

            with ThreadPoolExecutor(6) as executor:
                results = list(tqdm(executor.map(self.cust, ids), total=len(ids)))
            # with ThreadPoolExecutor(8) as executor:
            # results = []
            # for  i in (tqdm(range(len(ids)))):
            #     results.append(self.cust(ids[i]))
                # results = list(tqdm(executor.map(self.cust, ids), total=len(ids)))
            # results = []
            # for step, id in enumerate(tqdm(ids)):
            #     res = self.cust(id)
            #     results.append(res)

            glob_preds = (results)
            # from_preds2stringpred = list(map(self.from_preds2string, glob_preds))
            #
            # df_submit = self.df.copy()
            # df_submit['PredictionString'] = np.array(from_preds2stringpred)

            return glob_preds

    def cust(self, id):
        nuclei_mask, cell_mask, CAMs = self.load_shit(id)
        cur_cams = resize_for_tensors(CAMs.unsqueeze(0), nuclei_mask.shape)[0]
        frac = [i / 100 for i in range(5, 65, 5)]

        image_preds = [[] for j in frac]
        image_rles = []
        image_mean = cur_cams.squeeze(0).mean(dim=(1, 2))
        # hrcams = nn.Sigmoid()(torch.mean(cur_cams, dim=(1, 2)))
        # print(torch.mean(hrcams, dim=(1, 2)).max())

        # print(nn.Sigmoid()(cur_cams).mean(dim=(1, 2)))

        # 0 is background
        general_mask = cell_mask if self.masktype=='cell' else nuclei_mask
        for cell_idx in range(1, cell_mask.max() + 1):
            image_rles.append(encode_binary_mask(cell_mask == cell_idx).decode("utf-8"))
            current_cell_lst = []
            for class_id in range(self.num_classes):
                result = cur_cams[class_id] * (general_mask == cell_idx)
                num_pixels = np.count_nonzero(result)
                logits = result.sum().item() / num_pixels if num_pixels else 0
                #                 prob = logits/thresh
                # max_logits[class_id] = max(logits, max_logits[class_id])
                current_cell_lst.append((logits))
            # print(self.sigmoid(np.array(current_cell_lst)))
            x = torch.FloatTensor(current_cell_lst)
            # print(self.sigm(x).tolist())
            # print(final)
            for i, lst in enumerate(image_preds):
                lst.append(self.sigm(frac[i]*image_mean + (1-frac[i])*x).numpy())

        res = [(image_preds), (image_rles)]
        return res

    # def cust(self, id):
    #     # cur_cams = CAMS[i].unsqueeze(0)
    #     # cur_cell_masks = cell_masks[i]
    #
    #
    #
    #     nuclei_mask, cell_mask, CAMs = self.load_shit(id)
    #
    #     cell_masks = torch.from_numpy(cell_mask.astype(np.int32))
    #     nuclei_masks = torch.from_numpy(nuclei_mask.astype(np.int32))
    #
    #     uniqs, counts = torch.unique(cell_masks, return_counts=True)
    #     uniqs = uniqs[1:]
    #     non_z = counts[1:].unsqueeze(-1)
    #     cur_sz = non_z.size(0)
    #
    #     image_rles = []
    #     for uni in uniqs.tolist():
    #         image_rles.append(encode_binary_mask(cell_mask == uni).decode('utf-8'))
    #
    #     CAMS = resize_for_tensors(CAMs.unsqueeze(0), cell_masks.shape)[0]
    #     # cur_sz = cell_masks.amax(dim=(0, 1)).item()
    #     cur_cams = CAMS.unsqueeze(0) #1, 19, h, w
    #
    #
    #     general_mask = cell_masks if self.masktype=='cell' else nuclei_masks
    #
    #     cur_cell_masks = general_mask.unsqueeze(0).expand(cur_sz, -1, -1, -1)
    #     cur_cell_masks = (cur_cell_masks == torch.arange(1, cur_sz+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    #
    #     # print(cur_cell_masks.device)
    #
    #     # area = cur_cell_masks.sum(dim=(2, 3))
    #     # print(cur_cell_masks.shape)
    #
    #     summa = (cur_cams * (cur_cell_masks)).sum(dim=(2, 3))
    #
    #     # y_pred = self.sigm(summa)
    #
    #     y_pred = nn.Sigmoid()(torch.div(summa, non_z))
    #     y_pred = y_pred.cpu().numpy()
    #     # print(y_pred)
    #     # print(y_pred.shape)
    #
    #
    #     res = [y_pred, (image_rles)]
    #     return res

    def load_shit(self, id):
        nuc_segmentations = np.load(f'{self.nuc_load(id)}.npy')
        cell_segmentations = np.load(f'{self.cell_load(id)}.npy')


        CAMs = torch.load(os.path.join(self.pth, f'{id}.npy')).type(torch.FloatTensor)
        # print(CAMs.device)
        return nuc_segmentations, cell_segmentations, CAMs
#spiorfk



if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'config', 'cam_avenga.yaml')) as config_file:
        config = yaml.full_load(config_file)

    # model = Classifier(config['model']['arch'], config['model']['pretreined'],
    #                num_classes=config['model']['classes'], mode=config['args']['mode'])
    # trainer = CAMTrainer(config, None, None)

    # train_transform = get_transforms(config['train']['transform'])
    test_transform = get_transforms(config['test']['transform'])

    testdf = pd.read_csv(os.path.join(config['test']['path'], 'sample_submission.csv'))

    test_dataset = HPADatasetCAMTest(config, testdf, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'],
                             shuffle=False,
                             drop_last=False)

    # train_df = get_df_cam(path=config['train']['path'])
    #
    # train_dataset = HPADatasetCAM(config['train']['path'], train_df, transform=test_transform, yellow=True)
    #
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'],
    #                           shuffle=True,
    #                           drop_last=True)




    inferrer = Inferrer(config, test_loader, testdf, False)
    # inferrer.save_predictions(True, False, False, False)
    preds = inferrer.infer(masktype='cell')
    # df_submit.to_csv('~/hpa/submn.csv')
