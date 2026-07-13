import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auroc, binary_auprc
from sklearn.model_selection import train_test_split
from model import BrainToDiseaseNet
from utils import Dataset3d
import argparse

parser = argparse.ArgumentParser(description='Train CNN to predict diseases from 3D MRI end-to-end.')
parser.add_argument('--data_dir', type=str, dest='data_dir', required=True,
                    help='Path to directory containing train/test CSV metadata files.')
parser.add_argument('--mask_path', type=str, dest='mask_path', required=True,
                    help='Path to the brain mask NIfTI file (e.g. MNI152_T1_1mm_brain_mask_dil.nii.gz).')
parser.add_argument('--img_dir', type=str, dest='img_dir', required=True,
                    help='Path to directory containing preprocessed NIfTI images.')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=8)
parser.add_argument('--start_epoch', type=int, dest='start_epoch', default=1)
parser.add_argument('--num_epoch', type=int, dest='num_epoch', default=1)
parser.add_argument('--fold', type=int, dest='fold', default=0,
                    help='Cross-validation fold index.')
parser.add_argument('--ckpt_dir', type=str, dest='ckpt_dir', default=None,
                    help='Directory to save checkpoints (default: ./ckpt_fold<fold>/).')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
START_EPOCH = args.start_epoch
END_EPOCH = args.start_epoch + args.num_epoch
FOLD = args.fold
path_ckpt = args.ckpt_dir or './ckpt_fold%d/' % FOLD
os.makedirs(path_ckpt, exist_ok=True)

INIT_LEARNING_RATE = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

list_disease = [
    'atrial_fibrillation_or_flutter', 'coronary_artery_disease', 'diabetes_type_2',
    'hypercholesterolemia', 'hypertension', 'myocardial_infarction',
    'heart_failure', 'coronary_artery_disease_hard', 'chronic_obstructive_pulmonary_disease',
]

################ data #############
seed_partition = 0

# CSV files are pre-filtered for image quality and BMI outliers
df_tv = pd.read_csv(os.path.join(args.data_dir, 'tbl_clinical_train%d.csv' % FOLD))
df_tv = df_tv[['subject_id', 'image_id', 'bmi', 'gender', 'age_at_image'] + list_disease]
df_tv['file'] = os.path.join(args.img_dir, 'temp_preprocessing_') + \
    df_tv['image_id'].astype('str') + '.nii.gz'

df_train, df_vali = train_test_split(df_tv, train_size=0.8 / 0.9, shuffle=True,
                                     random_state=seed_partition)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_vali = df_vali.reset_index(drop=True)

train_dataset = Dataset3d(df_train, mask_path=args.mask_path, test=False)
vali_dataset = Dataset3d(df_vali, mask_path=args.mask_path, test=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=BATCH_SIZE)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size=1)
#####################################

################# model #############
model = BrainToDiseaseNet(output_dim=len(list_disease))
model = torch.nn.DataParallel(model)
model.to(device)

log_out = open('log_fold%d.txt' % FOLD, 'a')
if START_EPOCH == 1:
    header = 'epoch\tloss_train\t' + '\t'.join(
        ['auroc%d\tauprc%d' % (i + 1, i + 1) for i in range(len(list_disease))])
    log_out.write(header + '\n')

saved_model_name = os.path.join(path_ckpt, 'checkpoint_epoch%03d.pth.tar' % (START_EPOCH - 1))
if os.path.exists(saved_model_name):
    print('RESUMING TRAINING')
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

loss_func = torch.nn.BCEWithLogitsLoss()

for epoch in range(START_EPOCH, END_EPOCH):
    LEARNING_RATE = INIT_LEARNING_RATE * 0.3 ** (epoch // 30)
    print('epoch=%d, lr=%.5f' % (epoch, LEARNING_RATE))
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    ## train
    model.train()
    loss_train = 0
    for step, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = loss_func(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        if step % 10 == 0:
            print(f'epoch {epoch}, loss = {loss.item():.2f}')
    loss_train = loss_train / len(train_dataloader)
    log_out.write('%3d\t%.3f' % (epoch, loss_train))
    print('epoch%3d, loss_train=%.3f' % (epoch, loss_train))
    ## validate
    model.eval()
    gt_vali = []
    pred_vali = []
    with torch.no_grad():
        for step, (batch_val_x, batch_val_y) in enumerate(vali_dataloader):
            batch_val_y = batch_val_y.to(device)
            pred = F.sigmoid(model(batch_val_x))
            gt_vali += batch_val_y.tolist()
            pred_vali += pred.tolist()
    gt_vali = torch.tensor(gt_vali)
    pred_vali = torch.tensor(pred_vali)
    for i in range(pred_vali.shape[1]):
        auroc = binary_auroc(pred_vali[:, i], gt_vali[:, i]).item()
        auprc = binary_auprc(pred_vali[:, i], gt_vali[:, i]).item()
        log_out.write('\t%.3f\t%.3f' % (auroc, auprc))
        print('auroc%d=%.3f, auprc%d=%.3f' % (i + 1, auroc, i + 1, auprc))
    log_out.write('\n')
    log_out.flush()
    ## save checkpoint
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'loss_train': loss_train,
        'optimizer': optimizer.state_dict(),
    }, os.path.join(path_ckpt, 'checkpoint_epoch%03d.pth.tar' % epoch))

log_out.close()
