import numpy as np
from scipy.stats import norm
import torch
import shutil
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


def num2vect(x, bin_range, bin_step, sigma):
    """
    v, bin_centers = num2vect(x, bin_range, bin_step, sigma)

    Convert a scalar or array of values to a soft probability vector over bins.

    Args:
        x: scalar or array of values to encode
        bin_range: (start, end) tuple defining the bin range
        bin_step: bin width; must evenly divide (end - start)
        sigma: standard deviation for the Gaussian soft label
                (0 for hard/index label, >0 for soft label)

    Returns:
        v: soft label vector (or index when sigma=0)
        bin_centers: centre value of each bin
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step).astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


class Dataset3d(Data.Dataset):
    """PyTorch Dataset for 3D brain MRI volumes with BMI soft labels."""

    def __init__(self, dataframe, mask_path, test=False, return_row=False):
        """
        Args:
            dataframe: pandas DataFrame with columns ['file', 'bmi', 'subject_id', 'image_id']
            mask_path: path to the brain mask NIfTI file
            test: if False, apply random shift augmentation during loading
            return_row: if True, also return the raw DataFrame row as a dict
        """
        self.dataframe = dataframe
        self.bin_range = [10, 60]
        self.bin_step = 1
        self.sigma = 1
        self.test = test
        self.return_row = return_row
        self.mask = nib.load(mask_path).get_fdata().astype(bool)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        y, bc = num2vect(row['bmi'], self.bin_range, self.bin_step, self.sigma)
        y = torch.tensor(y, dtype=torch.float32)
        bc = torch.tensor(bc, dtype=torch.float32)
        img = nib.load(row['file'])
        imgex = img.get_fdata(dtype=np.float32)
        # apply brain mask and normalise
        imgex[~self.mask] = 0
        imgex = imgex / imgex.mean()

        if not self.test:
            random_shift = np.random.randint(-2, 2, size=3)
            imgex = randomShift(imgex, random_shift[0], random_shift[1], random_shift[2])

        imgex = crop_center(imgex, (160, 192, 160))
        input_data = torch.from_numpy(np.asarray([imgex]))

        if self.return_row:
            return (
                input_data.to(self.device),
                (y.to(self.device), bc.to(self.device)),
                row.to_dict()
            )
        else:
            return (
                input_data.to(self.device),
                (y.to(self.device), bc.to(self.device)),
            )
