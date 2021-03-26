

import yaml
import torch
import numpy as np
import os
import random

from hpa_competition.model_training.common.datasets import  HPADatasetTest
from hpa_competition.model_training.common.augmentations import get_transforms
from utils import get_df
from hpa_competition.model_training.common.models import get_network
import pandas as pd
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


with open(os.path.join(os.path.dirname(__file__), 'config', 'test.yaml')) as config_file:
    config = yaml.full_load(config_file)

# path = Path('../input/hpa-cell-tiles-test-with-enc-dataset')


# train_transform = get_transforms(config['train']['transform'])
test_transform = get_transforms(config['val']['transform'])

test_df = get_df(path=config['val']['path'], train=False)

test_ds = HPADatasetTest(config['val']['path'], test_df, transform=test_transform)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=12)

# trainer = Trainer(config, train_dl, val_dl)
# trainer.train()

model = get_network(config['model'])

log_path = os.path.join(
    config['log_path'],
    config['task'],
    config['experiment_name']
)
file_prefix = 'model_best'

device = config['devices'][0]
PATH = os.path.join(log_path, '{}.h5'.format(file_prefix))
model.load_state_dict(torch.load(PATH)['model'])
model.to(device)
model.eval()


y_pred = np.empty(shape=(0, config['model']['classes']))

for data in test_dl:
    X, y = data.to(device)
    current_y_pred = model(X)
    y_pred = np.vstack((y_pred, current_y_pred.detach().cpu()))




# with open('preds.pickle', 'wb') as handle:
#     pickle.dump(y_pred, handle)

#TODO: submit every class

cls_prds = np.argmax(y_pred, axis=-1)


test_df['cls'] = cls_prds


test_df['pred'] = test_df[['cls', 'enc']].apply(lambda r: str(r[0]) + ' 1 ' + r[1], axis=1)



subm = test_df.groupby(['image_id'])['pred'].apply(lambda x: ' '.join(x)).reset_index()
print(subm.head())
# subm = subm.loc[3:]
# subm.head()
#
sample_submission = pd.read_csv('/datasets/kolinko/hpa/sample_submission.csv')

sub = pd.merge(
    sample_submission,
    subm,
    how="left",
    left_on='ID',
    right_on='image_id',
)
#
def isNaN(num):
    return num != num

for i, row in sub.iterrows():
    if isNaN(row['pred']): continue
    sub.PredictionString.loc[i] = row['pred']

sub = sub[sample_submission.columns]
print(sub.head())


# sub.to_csv('submission.csv', index=False)
