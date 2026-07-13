import os
import numpy as np
import pandas as pd
import torch
import argparse
from model import SFCN
from utils import Dataset3d

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(description='Extract feature maps from a trained BMI model.')
parser.add_argument('--data_dir', type=str, dest='data_dir', required=True,
                    help='Path to directory containing per-dataset CSV metadata files.')
parser.add_argument('--mask_path', type=str, dest='mask_path', required=True,
                    help='Path to the brain mask NIfTI file.')
parser.add_argument('--img_dir', type=str, dest='img_dir', required=True,
                    help='Path to directory containing preprocessed NIfTI images.')
parser.add_argument('--datasets', type=str, dest='datasets', default='ukbb,hcp,hcp-aging',
                    help='Comma-separated list of dataset names (default: ukbb,hcp,hcp-aging).')
parser.add_argument('--ckpt_dir', type=str, dest='ckpt_dir', required=True,
                    help='Directory containing model checkpoints.')
parser.add_argument('--epoch', type=int, dest='epoch', default=10,
                    help='Epoch checkpoint to load.')
parser.add_argument('--out_dir', type=str, dest='out_dir', default='./tbl/',
                    help='Directory to save feature map output tables.')

args = parser.parse_args()
EPOCH = args.epoch
dataset_all = [d.strip() for d in args.datasets.split(',')]

name_label = 'bmi'
os.makedirs(args.out_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SFCN(output_dim=50)
model = torch.nn.DataParallel(model)
model.to(device)

saved_model_name = os.path.join(args.ckpt_dir, 'checkpoint_epoch%03d.pth.tar' % EPOCH)
checkpoint = torch.load(saved_model_name)
model.load_state_dict(checkpoint['state_dict'])
print(model)

## register forward hook to capture classifier feature maps
feature_map = []


def hook_feat_map(mod, inp, out):
    feature_map.append(out)


for layer in model.module.classifier:
    layer.register_forward_hook(hook_feat_map)

df_all = []
fm_all = []
for dataset in dataset_all:
    the_df = pd.read_csv(os.path.join(args.data_dir, dataset + '.csv'))
    the_df['age_at_image'] = (the_df['days_since_baseline'] / 365) + the_df['age_at_baseline']
    the_df['file'] = os.path.join(args.img_dir, dataset,
                                  'temp_preprocessing_') + the_df['image_id'].astype('str') + '.nii.gz'
    # filter by image quality and BMI range
    the_index = (the_df['r_correlation'] > 0.4) & \
                (the_df['bmi'] > 10) & (the_df['bmi'] < 60)
    the_df = the_df.loc[the_index, :]
    the_df['dataset'] = dataset
    the_df = the_df.loc[:, ['subject_id', 'image_id', 'dataset', name_label, 'gender', 'age_at_image', 'file']]
    the_df.dropna(inplace=True)
    df_all.append(the_df)

    the_dataset = Dataset3d(the_df, mask_path=args.mask_path, test=True)
    the_dataloader = torch.utils.data.DataLoader(the_dataset, batch_size=1)
    model.eval()
    with torch.no_grad():
        for step, (batch_val_x, batch_val_y) in enumerate(the_dataloader):
            print(step, the_df['subject_id'].iloc[step])
            feature_map = []
            model(batch_val_x)
            the_map = feature_map[0].squeeze(dim=0).cpu().numpy().astype('float')
            fm_all.append(the_map.flatten())

# save feature maps only
df_fm = pd.DataFrame(fm_all)
df_fm.to_csv(os.path.join(args.out_dir, 'fm_64.csv'), index=False)

# save feature maps with subject metadata
df_out = pd.concat([pd.concat(df_all, axis=0, ignore_index=True), df_fm], axis=1)
df_out.drop(columns=['file'], inplace=True)
df_out.to_csv(os.path.join(args.out_dir, 'tbl_fm_64.csv'), index=False)
