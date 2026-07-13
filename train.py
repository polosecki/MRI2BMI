import os
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import SFCN
import loss as dpl
from utils import save_checkpoint, Dataset3d
import torch
import argparse

parser = argparse.ArgumentParser(description='Train CNN to predict BMI from 3D MRI.')
parser.add_argument('--data_dir', type=str, dest='data_dir', required=True,
                    help='Path to directory containing per-dataset CSV metadata files.')
parser.add_argument('--mask_path', type=str, dest='mask_path', required=True,
                    help='Path to the brain mask NIfTI file (e.g. MNI152_T1_1mm_brain_mask_dil.nii.gz).')
parser.add_argument('--img_dir', type=str, dest='img_dir', required=True,
                    help='Path to directory containing preprocessed NIfTI images.')
parser.add_argument('--datasets', type=str, dest='datasets', default='ukbb,hcp,hcp-aging',
                    help='Comma-separated list of dataset names (default: ukbb,hcp,hcp-aging).')
parser.add_argument('--sample_size', type=int, dest='sample_size', default=-1,
                    help='Subsample training set to this size (-1 = use all).')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=8)
parser.add_argument('--start_epoch', type=int, dest='start_epoch', default=0)
parser.add_argument('--num_epoch', type=int, dest='num_epoch', default=1,
                    help='Number of epochs to train.')
parser.add_argument('--ckpt_dir', type=str, dest='ckpt_dir', default='./weights/',
                    help='Directory to save model checkpoints.')

args = parser.parse_args()
SAMPLE_SIZE = args.sample_size
BATCH_SIZE = args.batch_size
START_EPOCH = args.start_epoch
END_EPOCH = args.start_epoch + args.num_epoch
LEARNING_RATE = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

path_ckpt = args.ckpt_dir
os.makedirs(path_ckpt, exist_ok=True)

dataset_all = [d.strip() for d in args.datasets.split(',')]

## training sample size of each dataset (for oversampling small datasets)
num_sample = 2000

# train-vali-test split
ratio_partition = [0.7, 0.2, 0.1]
seed_partition = 0

df_train = []
df_vali = []
df_test = []
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
    the_df = the_df.loc[:, ['image_id', 'bmi', 'subject_id', 'file']]
    the_df.dropna(inplace=True)
    # train-vali-test split
    the_df_train, the_df_vt = train_test_split(
        the_df, train_size=ratio_partition[0], shuffle=True, random_state=seed_partition)
    the_df_vali, the_df_test = train_test_split(
        the_df_vt,
        train_size=ratio_partition[1] / (ratio_partition[1] + ratio_partition[2]),
        random_state=seed_partition)
    print(dataset_name, len(the_df), len(the_df_train), len(the_df_vali), len(the_df_test))
    # oversample small datasets to balance with ukbb
    if dataset_name == 'ukbb':
        sample_rate = 12000 / len(the_df_train)
    else:
        sample_rate = num_sample / len(the_df_train)
    the_df_train = the_df_train.sample(frac=sample_rate, replace=sample_rate > 1)
    df_train.append(the_df_train)
    df_vali.append(the_df_vali)
    df_test.append(the_df_test)

df_train = pd.concat(df_train, ignore_index=True)
df_vali = pd.concat(df_vali, ignore_index=True)
df_test = pd.concat(df_test, ignore_index=True)
# shuffle
df_train = df_train.sample(frac=1)
df_train.reset_index(inplace=True, drop=True)
df_vali.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)

if SAMPLE_SIZE != -1:
    df_train = df_train.sample(n=SAMPLE_SIZE, random_state=0)

train_dataset = Dataset3d(df_train, mask_path=args.mask_path, test=False)
vali_dataset = Dataset3d(df_vali, mask_path=args.mask_path, test=True)
test_dataset = Dataset3d(df_test, mask_path=args.mask_path, test=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=1)
model = SFCN(output_dim=50)
model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

log_out = open('log.txt', 'a')
if START_EPOCH == 1:
    log_out.write('epoch\tmse_train\tmse_vali\tmae_vali\tpear_vali\n')

saved_model_name = os.path.join(path_ckpt, 'checkpoint_epoch%03d.pth.tar' % (START_EPOCH - 1))
if exists(saved_model_name):
    print('RESUMING TRAINING')
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])

loss_func = dpl.my_KLDivLoss

for epoch in range(START_EPOCH, END_EPOCH):
    if epoch % 30 == 0 and epoch != 0:
        LEARNING_RATE = LEARNING_RATE * 0.3
        print('lr:', LEARNING_RATE)
    ## train
    model.train()
    loss_train = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        prediction = model(batch_x)
        loss = loss_func(prediction[0].reshape([len(batch_y[0]), -1]), batch_y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        if step % 10 == 0:
            print('epoch [{}], Loss: {:.2f}'.format(epoch, loss.item()))
    loss_train = loss_train / len(train_loader)
    ## validate
    gt_vali = []
    pred_vali = []
    mse_vali = 0
    mae_vali = 0
    model.eval()
    with torch.no_grad():
        for step, (batch_val_x, batch_val_y) in enumerate(vali_loader):
            prediction = model(batch_val_x)
            age_bin_centers = batch_val_y[1][0]
            prediction_reshaped = prediction[0].reshape([1, -1]).reshape(-1)
            prediction_age = torch.exp(prediction_reshaped) @ age_bin_centers
            y_reshaped = batch_val_y[0].reshape(-1)
            y_age = y_reshaped @ age_bin_centers
            mae_vali += torch.abs(prediction_age - y_age).sum()
            mse_vali += ((prediction_age - y_age) ** 2).sum()
            gt_vali.append(y_age.item())
            pred_vali.append(prediction_age.item())
    mae_vali = mae_vali / len(vali_loader)
    mse_vali = mse_vali / len(vali_loader)
    gt_vali = np.array(gt_vali)
    pred_vali = np.array(pred_vali)
    pear_vali = np.corrcoef(gt_vali, pred_vali)[0, 1]
    ## save and print
    log_out.write('%3d\t%.3f\t%.3f\t%.3f\t%.3f\n' % (epoch, loss_train, mse_vali, mae_vali, pear_vali))
    log_out.flush()
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'mae_vali': mae_vali,
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(path_ckpt, 'checkpoint_epoch%03d.pth.tar' % epoch))
    print('epoch [{}], mse: {:.3f}, mae: {:.3f}, pear: {:.3f}'.format(epoch, mse_vali, mae_vali, pear_vali))

log_out.close()
