import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auroc, binary_auprc
from model import BrainToDiseaseNet
from utils import Dataset3d
import argparse

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(description='Run inference with a trained disease prediction model.')
parser.add_argument('--data_dir', type=str, dest='data_dir', required=True,
                    help='Path to directory containing test CSV metadata files.')
parser.add_argument('--mask_path', type=str, dest='mask_path', required=True,
                    help='Path to the brain mask NIfTI file.')
parser.add_argument('--img_dir', type=str, dest='img_dir', required=True,
                    help='Path to directory containing preprocessed NIfTI images.')
parser.add_argument('--epoch', type=int, dest='epoch', default=1,
                    help='Epoch checkpoint to load for inference.')
parser.add_argument('--fold', type=int, dest='fold', default=0,
                    help='Cross-validation fold index.')
parser.add_argument('--ckpt_dir', type=str, dest='ckpt_dir', default=None,
                    help='Directory containing model checkpoints (default: ./ckpt_fold<fold>/).')

args = parser.parse_args()
EPOCH = args.epoch
FOLD = args.fold
path_ckpt = args.ckpt_dir or './ckpt_fold%d/' % FOLD

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
# CSV files are pre-filtered for image quality and BMI outliers
df_test = pd.read_csv(os.path.join(args.data_dir, 'tbl_clinical_test%d.csv' % FOLD))
df_test = df_test[['subject_id', 'image_id', 'bmi', 'gender', 'age_at_image'] + list_disease]
df_test['file'] = os.path.join(args.img_dir, 'temp_preprocessing_') + \
    df_test['image_id'].astype('str') + '.nii.gz'

test_dataset = Dataset3d(df_test, mask_path=args.mask_path, test=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

################ model #############
model = BrainToDiseaseNet(output_dim=len(list_disease))
model = torch.nn.DataParallel(model)
model.to(device)

saved_model_name = os.path.join(path_ckpt, 'checkpoint_epoch%03d.pth.tar' % EPOCH)
checkpoint = torch.load(saved_model_name)
model.load_state_dict(checkpoint['state_dict'])

## inference
model.eval()
gt = []
pred = []
with torch.no_grad():
    for step, (batch_x, batch_y) in enumerate(test_dataloader):
        batch_y = batch_y.to(device)
        prediction = F.sigmoid(model(batch_x))
        gt += batch_y.tolist()
        pred += prediction.tolist()

gt = torch.tensor(gt)
pred = torch.tensor(pred)

log_out = open('log_pred_fold%d.txt' % FOLD, 'w')
header = 'epoch\t' + '\t'.join(
    ['auroc%d\tauprc%d' % (i + 1, i + 1) for i in range(len(list_disease))]) + \
    '\tauroc_avg\tauprc_avg\n'
log_out.write(header)
log_out.write('%d' % EPOCH)

auroc_sum = auprc_sum = 0
for i in range(pred.shape[1]):
    auroc = binary_auroc(pred[:, i], gt[:, i]).item()
    auprc = binary_auprc(pred[:, i], gt[:, i]).item()
    auroc_sum += auroc
    auprc_sum += auprc
    log_out.write('\t%.3f\t%.3f' % (auroc, auprc))
    print('auroc%d=%.3f, auprc%d=%.3f' % (i + 1, auroc, i + 1, auprc))

log_out.write('\t%.3f\t%.3f\n' % (auroc_sum / pred.shape[1], auprc_sum / pred.shape[1]))
print('auroc_avg=%.3f, auprc_avg=%.3f' % (auroc_sum / pred.shape[1], auprc_sum / pred.shape[1]))
log_out.close()
