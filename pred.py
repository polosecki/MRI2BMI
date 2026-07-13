import os
import numpy as np
import pandas as pd
import torch
from model import SFCN
import argparse
from utils import Dataset3d
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)


def mae(y_true, y_pred):
    return abs(y_true - y_pred).mean()


def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean() ** 0.5


parser = argparse.ArgumentParser(description='Run inference with a trained BMI prediction model.')
parser.add_argument('--data_dir', type=str, dest='data_dir', required=True,
                    help='Path to directory containing per-dataset CSV metadata files.')
parser.add_argument('--mask_path', type=str, dest='mask_path', required=True,
                    help='Path to the brain mask NIfTI file.')
parser.add_argument('--img_dir', type=str, dest='img_dir', required=True,
                    help='Path to directory containing preprocessed NIfTI images.')
parser.add_argument('--datasets', type=str, dest='datasets', default='ukbb,hcp,hcp-aging',
                    help='Comma-separated list of dataset names (default: ukbb,hcp,hcp-aging).')
parser.add_argument('--num_epoch', type=int, dest='num_epoch', default=10,
                    help='Epoch checkpoint to load for inference.')
parser.add_argument('--ckpt_dir', type=str, dest='ckpt_dir', default='./weights/',
                    help='Directory containing model checkpoints.')
parser.add_argument('--out_dir', type=str, dest='out_dir', default='./pred/',
                    help='Directory to save prediction outputs.')

args = parser.parse_args()
NUM_EPOCH = args.num_epoch
dataset_all = [d.strip() for d in args.datasets.split(',')]

name_label = 'bmi'
path_out = args.out_dir
os.makedirs(path_out, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

model = SFCN(output_dim=50)
model = torch.nn.DataParallel(model)
model.to(device)
saved_model_name = os.path.join(args.ckpt_dir, 'checkpoint_epoch%03d.pth.tar' % NUM_EPOCH)
checkpoint = torch.load(saved_model_name)
model.load_state_dict(checkpoint['state_dict'])

# train-vali-test split (must match split used during training)
ratio_partition = [0.7, 0.2, 0.1]
seed_partition = 0

gt_all = np.zeros(0)
pred_all = []
df_out_all = []
for dataset_name in dataset_all:
    the_df = pd.read_csv(os.path.join(args.data_dir, dataset_name + '.csv'))
    the_df['age_at_image'] = (the_df['days_since_baseline'] / 365) + the_df['age_at_baseline']
    the_df.drop_duplicates('subject_id', inplace=True)
    the_df['file'] = os.path.join(args.img_dir, dataset_name,
                                  'temp_preprocessing_') + the_df['image_id'].astype('str') + '.nii.gz'
    # filter by image quality and BMI range
    the_index = (the_df['r_correlation'] > 0.4) & \
                (the_df['bmi'] > 10) & (the_df['bmi'] < 60)
    the_df = the_df.loc[the_index, :]
    the_df = the_df.loc[:, ['image_id', name_label, 'subject_id', 'file']]
    the_df.dropna(inplace=True)
    # train-vali-test split
    the_df_train, the_df_vt = train_test_split(
        the_df, train_size=ratio_partition[0], shuffle=True, random_state=seed_partition)
    the_df_vali, the_df_test = train_test_split(
        the_df_vt,
        train_size=ratio_partition[1] / (ratio_partition[1] + ratio_partition[2]),
        random_state=seed_partition)
    # inference on test set
    test_dataset = Dataset3d(the_df_test, mask_path=args.mask_path, test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    gt = the_df_test[name_label].to_numpy()
    gt_all = np.concatenate((gt_all, gt))
    pred = []
    model.eval()
    with torch.no_grad():
        for step, (batch_val_x, batch_val_y) in enumerate(test_loader):
            prediction = model(batch_val_x)
            age_bin_centers = batch_val_y[1][0]
            prediction_reshaped = prediction[0].reshape([1, -1]).reshape(-1)
            prediction_age = torch.exp(prediction_reshaped) @ age_bin_centers
            pred.append(prediction_age.tolist())
            pred_all.append(prediction_age.tolist())
    pred = np.array(pred).flatten()
    np.save(os.path.join(path_out, 'gt_' + dataset_name), gt)
    np.save(os.path.join(path_out, 'pred_' + dataset_name), pred)
    test_pear = np.corrcoef(gt, pred)[0, 1]
    test_mae = mae(gt, pred)
    test_rmse = rmse(gt, pred)
    print(dataset_name)
    print('Pearson=%.3f' % test_pear)
    print('MAE=%.3f' % test_mae)
    print('RMSE=%.3f' % test_rmse)
    the_df_out = the_df_test.loc[:, ['subject_id', name_label]].copy()
    the_df_out['pred'] = pred
    the_df_out['dataset'] = dataset_name
    df_out_all.append(the_df_out)

df_out = pd.concat(df_out_all)
df_out.to_csv(os.path.join(path_out, 'tbl_pred_epoch%d.csv' % NUM_EPOCH), index=False)

pred_all = np.array(pred_all).flatten()
test_pear = np.corrcoef(gt_all, pred_all)[0, 1]
test_mae = mae(gt_all, pred_all)
test_rmse = rmse(gt_all, pred_all)
print('Overall performance:')
print('Pearson=%.3f' % test_pear)
print('MAE=%.3f' % test_mae)
print('RMSE=%.3f' % test_rmse)
