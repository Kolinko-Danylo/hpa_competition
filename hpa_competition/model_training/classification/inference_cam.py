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
import albumentations as A
res_dict = {t: A.Resize(t, t, interpolation=cv2.INTER_NEAREST_EXACT) for t in [2048, 3072, 1728]}




class Inferrer:
    def __init__(self, config, dl, df, trainset, use_gpu):
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

        self.nuc_load = lambda image_id: os.path.join(self.cache_path, 'nuc', image_id)
        self.cell_load = lambda image_id: os.path.join(self.cache_path, 'cell', image_id)

        # use_gpu = '0'
        # use_gpu = True
        device_str = 'cuda' if use_gpu else 'cpu'
        self.dev = torch.device(device_str)
        if use_gpu:
            self.model.cuda()
        self.model.eval()
        st_dt = torch.load(self.model_path, map_location=torch.device(device_str))['model']
        self.model.load_state_dict(st_dt)
        self.segmentator = cellsegmentator.CellSegmentator(
            config['segm_model']['nuclei_path'],
            config['segm_model']['cell_path'],
            device=device_str,
            multi_channel_model=True
        )

    def save_predictions(self, save_cams=True, save_masks=False, viz=False, infer=False, retmasks=False):
        image_rles = []
        tot_inf = []
        masklst = []
        with torch.no_grad():
            for step, (image, unnorm_image, image_id) in (enumerate(tqdm(self.dl))):
                # print(unnorm_image.shape)
                if not os.path.exists(self.pth):
                    os.makedirs(self.pth)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # unnorm_image = unnorm_image.to(self.dev)
                    image = image.to(self.dev)

                    # preds1, CAMs1, masks1 = get_cam(self.model, image, ttaflag=True, scale=1)

                    preds, CAMs, masks = get_cam(self.model, image, ttaflag=True, scale=1)
                    preds1, CAMs1, masks1 = get_cam(self.model, image, ttaflag=False, scale=1)
                    masks = masks.cpu().type(torch.FloatTensor)
                    masks1 = masks1.cpu().type(torch.FloatTensor)



                    # if retmasks:
                    #     masklst.append(masks)


                    # preds1, CAMs1, masks1 = get_cam(self.model, image, 1, flag=True)#TODO: few scales. flips. other tta; use mask
                    # CAMs1 = CAMs1.cpu()

                    CAMs = CAMs.cpu()
                    print(CAMs.size())
                    print(masks.shape)
                    # return preds, CAMs, masks, unnorm_image

                    # unnorm_image /= 255

                    # unnorm_image = unnorm_image.cpu()

                    cells = []

                    rgb_batch = [unnorm_image[..., [0, 3, 2]][i].numpy().astype(float) / 255 for i in
                          range(unnorm_image.size(0))]
                    blue_batch = [unnorm_image[..., 2][i].numpy().astype(float) / 255 for i in range(unnorm_image.size(0))]
                    nuc_segmentations = self.segmentator.pred_nuclei(blue_batch)
                    cell_segmentations = self.segmentator.pred_cells(rgb_batch, precombined=True)
                    for nuc_seg, cell_seg in zip(nuc_segmentations, cell_segmentations):
                        _, cell = label_cell(nuc_seg, cell_seg)
                        cells.append(cell)

                    if infer:
                        resized_cells = []

                        for i in range(len(image_id)):
                            cur_cell = cells[i]
                            og_size = self.df[self.df.ID == image_id[i]]
                            og_wh = (og_size.ImageWidth.values[0], og_size.ImageHeight.values[0])
                            resized = res_dict[og_wh[0]](image=np.random.randn(*og_wh), mask=cur_cell)['mask']
                            resized_cells.append(resized)
                        for i in range(len(image_id)):


                            # loaded = np.load(os.path.join(b, lds))['arr_0']
                            uniqs = np.unique(resized_cells[i]).tolist()
                            cur_rles = []
                            assert uniqs[0] == 0
                            for u in uniqs[1:]:
                                cur_rles.append(encode_binary_mask((resized_cells[i] == u)).decode("utf-8"))
                            image_rles.append(cur_rles)

                        for i in range(len(image_id)):
                            tot_inf.append(self.cust1(cells[i], CAMs[i]))
                    # for j in range(len(image_id)):
                    #     np.save(f'/common/danylokolinko/hpa_mask_semantic/nuc/{image_id[j]}', cv2.resize(nuc_segmentations[j], (1024, 1024)))


                    if save_cams:
                        for j in range(len(image_id)):
                            torch.save(CAMs[j], os.path.join(self.pth, f'{image_id[j]}.npy'))

                    if save_masks:
                        for j in range(len(image_id)):
                            np.save(self.cell_load(image_id[j]), cells[j])
                    if viz:
                        hrcams = nn.Sigmoid()(resize_for_tensors(CAMs.type(torch.FloatTensor), (512, 512)))
                        hrcams1 = nn.Sigmoid()(resize_for_tensors(CAMs1.type(torch.FloatTensor), (512, 512)))


                        for i, id in enumerate(image_id):

                            print_masked_img(self.config[self.train_str]['path'], self.train_str, id, hrcams[i], cell_mask=cells[i], cell_pred=nn.Sigmoid()(masks[i]))
                            print_masked_img(self.config[self.train_str]['path'], self.train_str, id, hrcams1[i],  cell_mask=cells[i], cell_pred=nn.Sigmoid()(masks1[i]))
                    if retmasks:
                        return masklst

        if infer:
            return [tot_inf, image_rles]


    def from_preds2string(self, preds):
        image_preds = preds[0]
        image_rles = preds[1]
        strt = " ".join(
            [f'{class_idx} {image_preds[step][class_idx]} {rle}' for step, rle in enumerate(image_rles) for
             class_idx in range(self.num_classes)])
        return strt

    def infer(self, masktype):
        with torch.no_grad():
            self.masktype=masktype
            ids = self.df.ID.tolist()
            with ThreadPoolExecutor(6) as executor:
                results = list(tqdm(executor.map(self.cust, ids), total=len(ids)))
            return results

    # def cust(self, id):
    #     nuclei_mask, cell_mask, CAMs = self.load_shit(id)

        # cur_cams = resize_for_tensors(CAMs.unsqueeze(0), nuclei_mask.shape)[0]
        # frac = [i / 100 for i in range(0, 65, 5)]

        # image_preds = [[] for j in frac]
        # image_rles = []
        # image_mean = cur_cams.squeeze(0).mean(dim=(1, 2))
        # # hrcams = nn.Sigmoid()(torch.mean(cur_cams, dim=(1, 2)))
        # # print(torch.mean(hrcams, dim=(1, 2)).max())

#         # print(nn.Sigmoid()(cur_cams).mean(dim=(1, 2)))

        # # 0 is background
        # general_mask = cell_mask if self.masktype=='cell' else nuclei_mask
        # for cell_idx in range(1, cell_mask.max() + 1):
        #     image_rles.append(encode_binary_mask(cell_mask == cell_idx).decode("utf-8"))
        #     current_cell_lst = []
        #     for class_id in range(self.num_classes):
        #         result = cur_cams[class_id] * (general_mask == cell_idx)
        #         num_pixels = np.count_nonzero(result)
        #         logits = result.sum().item() / num_pixels if num_pixels else 0
        #         #                 prob = logits/thresh
        #         # max_logits[class_id] = max(logits, max_logits[class_id])
        #         current_cell_lst.append((logits))
        #     # print(self.sigmoid(np.array(current_cell_lst)))
        #     x = torch.FloatTensor(current_cell_lst)
        #     # print(self.sigm(x).tolist())
        #     # print(final)
        #     for i, lst in enumerate(image_preds):
        #         lst.append(self.sigm(frac[i]*image_mean + (1-frac[i])*x).numpy())

        # res = [(image_preds), (image_rles)]
        # return res

    def load_shit(self, id):
        nuc_segmentations = np.load(f'{self.nuc_load(id)}.npy')
        cell_segmentations = np.load(f'{self.cell_load(id)}.npy')


        CAMs = torch.load(os.path.join(self.pth, f'{id}.npy'))
        # print(CAMs.device)
        return nuc_segmentations, cell_segmentations, CAMs.type(torch.FloatTensor)

    def cust1(self, cell_mask, CAMs):
        # nuclei_mask, cell_mask, CAMs = self.load_shit(id)
        CAMs = CAMs.type(torch.FloatTensor)
        cur_cams = resize_for_tensors(CAMs.unsqueeze(0), cell_mask.shape)[0]
        frac = [i / 100 for i in range(0, 65, 5)]

        image_preds = [[] for j in frac]
        image_rles = []
        image_mean = cur_cams.squeeze(0).mean(dim=(1, 2))
        # hrcams = nn.Sigmoid()(torch.mean(cur_cams, dim=(1, 2)))
        # print(torch.mean(hrcams, dim=(1, 2)).max())

        # print(nn.Sigmoid()(cur_cams).mean(dim=(1, 2)))

        # 0 is background
        general_mask = cell_mask
        # cell_mask_resized = cv2.resize((cell_mask), og_wh)
        cell_mask_resized = cell_mask

        for cell_idx in range(1, cell_mask.max() + 1):
            # cell_mask_resized1 = cv2.resize((cell_mask_resized), og_wh)

            # image_rles.append(encode_binary_mask((cell_mask_resized == cell_idx)).decode("utf-8"))
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
                lst.append(self.sigm(frac[i] * image_mean + (1 - frac[i]) * x).numpy())

        res = [(image_preds)]
        return res


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'config', 'cam_avenga.yaml')) as config_file:
        config = yaml.full_load(config_file)

    test_transform = get_transforms(config['test']['transform'])

    testdf = pd.read_csv(os.path.join(config['test']['path'], 'sample_submission.csv'))

    test_dataset = HPADatasetCAMTest(config, testdf, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'],
                             shuffle=False,
                             drop_last=False)


    inferrer = Inferrer(config, test_loader, testdf, False)
    inferrer.save_predictions(infer=True)
































# images = build_image_names(image_id, self.img_dir)[-1]
# nuc_segmentations = self.segmentator.pred_nuclei(images[2])
# cell_segmentations = self.segmentator.pred_cells(images)
# for j in range(len(image_id)):
#     nuc_seg, cell_seg = label_cell(nuc_segmentations[j], cell_segmentations[j])
#     np.save(self.nuc_load(image_id[j]), nuc_seg)
