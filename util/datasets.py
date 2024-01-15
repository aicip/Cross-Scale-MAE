import os
import warnings
from glob import glob
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
import rasterio as rio
from rasterio import logging
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Callable, Tuple, Dict
from torch import Tensor
import fiona
from fiona.errors import FionaValueError
from fiona.transform import transform_geom


log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


class BaseDataset(Dataset):
    """
    Abstract class.
    """

    ignore_index = -9999

    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """

        # Train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        # TODO: The following paper proposes using bilinear instead of bicubic for interpolation mode
        # https://arxiv.org/pdf/1511.08861.pdf

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(transforms.RandomHorizontalFlip())
            t.append(transforms.RandomVerticalFlip())
            t.append(
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.25, 1.0),
                    interpolation=interpol_mode,
                    antialias=True,
                ),  # 3 is bicubic
            )
            ###########################################
            return transforms.Compose(t)

        # Eval transform
        # TODO: These may need adjustment
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode, antialias=True
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class Dataset_fmow_rgb(BaseDataset):
    # ORIGINAL SatMAE:
    # mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    # std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    # UPDATED:
    mean = [0.43392888, 0.43578541, 0.40744025]
    std = [0.19828456, 0.19250111, 0.19454683]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        self.base_path = os.path.dirname(csv_path)
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # keep first 10 rows only
        self.data_info = self.data_info
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)
        print(f"Dataset contains {self.data_len} samples")

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        single_image_name = (
            single_image_name
            if os.path.isabs(single_image_name)
            else os.path.join(self.base_path, single_image_name)
        )
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class Dataset_coco(BaseDataset):
    # TODO: Note, this only loads images for pretraining for now, not the annotations
    mean = [0.47004986, 0.44683802, 0.40762289]
    std = [0.24388726, 0.23901215, 0.24204848]

    def __init__(self, root_path, transform):
        super().__init__(in_c=3)
        self.root_path = root_path
        self.transforms = transform
        self.imgs = glob(os.path.join(root_path, "**/*.jpg"), recursive=True)
        self.data_len = len(self.imgs)
        print(f"Dataset contains {self.data_len} samples")

    def __getitem__(self, index):
        impath = self.imgs[index]
        img_as_img = Image.open(impath).convert("RGB")

        img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor, 0)

    def __len__(self):
        return self.data_len


class Dataset_fmow_temporal(BaseDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose(
            [
                # transforms.Scale(224),
                transforms.RandomCrop(224),
            ]
        )
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image2_arr = np.asarray(self.data_info.iloc[:, 3])
        self.date2 = np.asarray(self.data_info.iloc[:, 4])
        self.site = np.asarray(self.data_info.iloc[:, 5])
        self.region = np.asarray(self.data_info.iloc[:, 6])
        self.sensor = np.asarray(self.data_info.iloc[:, 7])
        self.sensor2 = np.asarray(self.data_info.iloc[:, 8])
        self.original_ind = np.asarray(self.data_info.iloc[:, 9])
        self.original_ind2 = np.asarray(self.data_info.iloc[:, 10])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name1 = self.image_arr[index]
        single_image_name2 = self.image2_arr[index]
        # Open image
        img_as_img1 = Image.open(single_image_name1)  # .convert('RGB')
        img_as_img2 = Image.open(single_image_name2)  # .convert('RGB')
        # Transform the image
        img_as_tensor1 = self.transforms(img_as_img1)
        img_as_tensor2 = self.transforms(img_as_img2)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):
        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class Dataset_fmow_sentinel(BaseDataset):
    label_types = ["value", "one-hot"]
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(
        self,
        csv_path: str,
        transform: Any,
        years: Optional[List[int]] = [*range(2000, 2021)],
        categories: Optional[List[str]] = None,
        label_type: str = "value",
        masked_bands: Optional[List[int]] = None,
        dropped_bands: Optional[List[int]] = None,
    ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path).sort_values(
            ["category", "location_id", "timestamp"]
        )

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df["year"] = [
                int(timestamp.split("-")[0]) for timestamp in self.df["timestamp"]
            ]
            self.df = self.df[self.df["year"].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f"FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:",
                ", ".join(self.label_types),
            )
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(float)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection["image_path"])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection["category"])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [
                i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands
            ]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            "images": images,
            "labels": labels,
            "image_ids": selection["image_id"],
            "timestamps": selection["timestamp"],
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(
                SentinelNormalize(mean, std)
            )  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(
                    input_size, scale=(0.2, 1.0), interpolation=interpol_mode
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class Dataset_eurosat(BaseDataset):
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, "r") as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(float)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [
                i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands
            ]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> BaseDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == "fmow_rgb":
        mean = Dataset_fmow_rgb.mean
        std = Dataset_fmow_rgb.std
        transform = Dataset_fmow_rgb.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = Dataset_fmow_rgb(csv_path, transform)
    elif args.dataset_type == "fmow_temporal":
        dataset = Dataset_fmow_temporal(csv_path)
    elif args.dataset_type == "fmow_sentinel":
        mean = Dataset_fmow_sentinel.mean
        std = Dataset_fmow_sentinel.std
        transform = Dataset_fmow_sentinel.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = Dataset_fmow_sentinel(
            csv_path,
            transform,
            masked_bands=args.masked_bands,
            dropped_bands=args.dropped_bands,
        )
    elif args.dataset_type == "euro_sat":
        mean, std = Dataset_eurosat.mean, Dataset_eurosat.std
        transform = Dataset_eurosat.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = Dataset_eurosat(
            csv_path,
            transform,
            masked_bands=args.masked_bands,
            dropped_bands=args.dropped_bands,
        )
    elif args.dataset_type == "naip":
        from util.naip_loader import (
            NAIP_CLASS_NUM,
            NAIP_test_dataset,
            NAIP_train_dataset,
        )

        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    elif args.dataset_type == "coco":
        mean = Dataset_coco.mean
        std = Dataset_coco.std
        transform = Dataset_coco.build_transform(is_train, args.input_size, mean, std)
        dataset = Dataset_coco(csv_path, transform)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    print(f"Using dataset: {dataset}")
    return dataset
