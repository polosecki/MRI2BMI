import numpy as np
import torch
import torch.utils.data as Data
import nibabel as nib


def randomShift(img, shiftx, shifty, shiftz):
    if shiftx > 0:
        temp_x_index = slice(shiftx, None)
        img_x_index = slice(None, -shiftx)
    elif shiftx < 0:
        temp_x_index = slice(None, shiftx)
        img_x_index = slice(abs(shiftx), None)
    else:
        temp_x_index = slice(None, None)
        img_x_index = slice(None, None)
    if shifty > 0:
        temp_y_index = slice(shifty, None)
        img_y_index = slice(None, -shifty)
    elif shifty < 0:
        temp_y_index = slice(None, shifty)
        img_y_index = slice(abs(shifty), None)
    else:
        temp_y_index = slice(None, None)
        img_y_index = slice(None, None)
    if shiftz > 0:
        temp_z_index = slice(shiftz, None)
        img_z_index = slice(None, -shiftz)
    elif shiftz < 0:
        temp_z_index = slice(None, shiftz)
        img_z_index = slice(abs(shiftz), None)
    else:
        temp_z_index = slice(None, None)
        img_z_index = slice(None, None)
    temp = np.zeros_like(img)
    temp[temp_x_index, temp_y_index, temp_z_index] = img[img_x_index, img_y_index, img_z_index]
    return temp


def crop_center(data, out_sp):
    """
    Returns the centre crop of a 3D or 4D volume.

    Args:
        data: numpy array of shape (X, Y, Z) or (C, X, Y, Z)
        out_sp: desired output spatial size as (X, Y, Z)

    Example:
        data = np.random.rand(182, 218, 182)
        out = crop_center(data, (160, 192, 160))
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ValueError('Wrong dimension! dim=%d.' % nd)
    return data_crop


class Dataset3d(Data.Dataset):
    """PyTorch Dataset for 3D brain MRI volumes with multi-label disease targets."""

    def __init__(self, dataframe, mask_path, test=False):
        """
        Args:
            dataframe: pandas DataFrame with columns ['file'] plus one column per disease label
            mask_path: path to the brain mask NIfTI file
            test: if False, apply random shift augmentation during loading
        """
        self.dataframe = dataframe
        self.test = test
        self.mask = nib.load(mask_path).get_fdata().astype(bool)
        self.diseases = [
            'atrial_fibrillation_or_flutter', 'coronary_artery_disease', 'diabetes_type_2',
            'hypercholesterolemia', 'hypertension', 'myocardial_infarction',
            'heart_failure', 'coronary_artery_disease_hard', 'chronic_obstructive_pulmonary_disease',
        ]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        label = torch.tensor(row[self.diseases].values, dtype=torch.float32)
        image = nib.load(row['file']).get_fdata(dtype=np.float32)
        # apply brain mask and normalise
        image[~self.mask] = 0
        image = image / image.mean()
        if not self.test:
            random_shift = np.random.randint(-2, 2, size=3)
            image = randomShift(image, random_shift[0], random_shift[1], random_shift[2])
        image = crop_center(image, (160, 192, 160))
        # (160, 192, 160) -> (1, 160, 192, 160)
        feature = torch.from_numpy(np.asarray([image]))
        return feature, label
