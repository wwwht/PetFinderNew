'''
Author: your name
Date: 2021-11-06 09:36:32
LastEditTime: 2021-11-07 19:41:07
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \PetFinderNew\data\MyDataset.py
'''
from torchvision.io import read_image
from torch.utils.data import Dataset
from .transformation.default_tranform import default_transforms

class MyDataset(Dataset):
    def __init__(self, df, transforms, image_size, mode):
        self._X = df['Id'].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        self._transform = transforms
        self._image_size = image_size
        self_mode = mode

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform[self.mode](image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image


    def __len__(self):
        return len(self._X)