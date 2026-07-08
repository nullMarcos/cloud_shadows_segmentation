from typing import Dict, List, Optional, Tuple

import joblib
import netCDF4 as nc
import numpy as np
import torch
from datasets.dataset_utils import (
    center_crop,
    compute_percentiles,
    random_crop,
    random_horizontal_flip,
    random_rotation,
    random_vertical_flip,
)
from torch.utils.data import DataLoader, Dataset


class HyperspectralDataset(Dataset):
    """
    Dataset class for hyperspectral data.

    Args:
        base_path (str): Base path for data files.
        model_name (str): Name of the model.
        obj_list (List[str]): List of object file names.
        norm_type (str): Type of normalization to apply.
        class_map (Dict[str, int]): Mapping of class names to integer labels.
        scaler (Optional[object]): Scaler object for normalization.
        apply_transforms (bool): Whether to apply data augmentation transforms. Default is False.
        win_size (int): Window size for data processing. Default is 32.
    """

    def __init__(
        self,
        base_path: str,
        obj_list: List[str],
        norm_type: str,
        class_map: Dict[str, int],
        scaler: Optional[object],
        dataset_type: str,
        apply_transforms: bool = False,
        win_size: int = 32,
        test: bool = False,
    ):
        self.base_path = base_path
        self.obj_list = obj_list
        self.class_map = class_map
        self.scaler = scaler
        self.norm_type = norm_type
        self.dataset_type = dataset_type
        self.apply_transforms = apply_transforms
        self.win_size = win_size
        self.test = test

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.obj_list)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[np.ndarray, torch.Tensor]: Normalized data and corresponding mask.
        """
        mask_path = self.obj_list[index]

        if self.dataset_type == "mair_cs":
            data_path = (
                mask_path.replace("_L2_", "_L1B_")
                .replace("_cloudmask_", "_CH4_")
                .replace(".nc", ".npy")
            )
            (start_x, end_x), (start_y, end_y) = (0, 300), (27, 199)
            data = np.load(f"{self.base_path}/images/{data_path}")
            mask = nc.Dataset(f"{self.base_path}/masks/{mask_path}")

            mask_full = torch.zeros(
                (end_x - start_x, end_y - start_y), dtype=torch.long
            )
            for key, value in self.class_map.items():
                mask_tmp = torch.from_numpy(
                    mask[key][start_x:end_x, start_y:end_y].astype(np.int64)
                )
                mask_full = torch.where(mask_tmp == 1, value, mask_full)

        elif self.dataset_type == "msat_cs2":
            (start_y, end_y) = (16, 2032)

            data_path = mask_path
            data = np.load(f"{self.base_path}/images/{data_path}")
            mask_file = (
                data_path.replace("_L1B_CH4_", "_L2_")
                .replace("v04001001", "v03002000")
                .replace(".npy", "_ns.npy")
            )
            mask_full = np.load(f"{self.base_path}/masks/{mask_file}")[
                1:, start_y:end_y
            ]

            mask_full = np.nan_to_num(mask_full, nan=0.0)
            mask_full[mask_full == 3.0] = 1.0

            if self.apply_transforms:
                data, mask_full = random_crop(data, mask_full, crop_size=224)
            elif not self.test:
                data, mask_full = center_crop(data, mask_full, crop_size=224)
            elif self.test:
                pass

        if self.apply_transforms:
            data, mask_full = random_horizontal_flip(data, mask_full)
            data, mask_full = random_vertical_flip(data, mask_full)
            data, mask_full = random_rotation(data, mask_full)

        data = self.normalize_data(data)
        return data, mask_full

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the input data based on the specified normalization type.

        Args:
            data (np.ndarray): Input data to normalize.

        Returns:
            np.ndarray: Normalized data.
        """
        if self.norm_type in ["minmax", "minmax_full"]:
            return (data - self.scaler.data_min_) / (
                self.scaler.data_max_ - self.scaler.data_min_
            )
        elif self.norm_type in ["std", "std_full"]:
            return (data - self.scaler.mean_) / (self.scaler.var_**0.5)
        elif self.norm_type == "p_norm":
            lower, higher = compute_percentiles(data, lower=1, higher=99)
            data = np.clip(data, a_min=lower, a_max=higher)
            return (data - self.scaler.mean_) / (self.scaler.var_**0.5)
        else:
            return data  # Return original data if no normalization is specified


def get_dataloader(
    base_path: str,
    norm_type: str,
    mask_types: List[str] = ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"],
    batch_size: int = 32,
    num_workers: int = 0,
    fold: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Build and return dataloaders for training and validation.

    Args:
        base_path (str): Base path for data files.
        norm_type (str): Type of normalization to apply.
        mask_types (List[str]): Types of masks to use. Default is ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"].
        model_name (str): Name of the model. Default is "logreg".
        batch_size (int): Batch size for dataloaders. Default is 32.
        num_workers (int): Number of worker processes for data loading. Default is 0.
        fold (int): Fold number for cross-validation. Default is 0.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: Train dataloader, validation dataloader, and full class map.
    """
    assert any(
        x in ["plumes", "cloud_shadow_mask", "cloud_mask", "dark_surface_mask"]
        for x in mask_types
    )

    obj_list = np.load(base_path + "/mask_list.npy")
    test_list = np.load(base_path + "/mask_list_test.npy")

    scaler = (
        None
        if norm_type in ["none", "l_norm_clip"]
        else joblib.load(f"{base_path}/scaler_{norm_type}.pkl")
    )
    fold_ids = joblib.load(f"{base_path}/fold_{fold}_ids.pkl")

    mask = np.ones(len(obj_list), dtype=bool)
    mask[fold_ids] = False  # setting validation indexes as False
    val_list, train_list = obj_list[mask], obj_list[~mask]

    # Tomar aprox. 100 GB del dataset (pesa 591G).
    train_list = train_list[: len(train_list) // 10]
    val_list = val_list[: len(val_list) // 10]
    test_list = test_list[: len(test_list) // 10]

    class_map = {
        type_: i + 1 for i, type_ in enumerate(mask_types)
    }  # label 0 reserved for normal objects

    dataset_type = base_path.split("/")[-1]

    train_dataset = HyperspectralDataset(
        base_path,
        train_list,
        norm_type,
        class_map,
        scaler,
        dataset_type,
        apply_transforms=False,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    val_dataset = HyperspectralDataset(
        base_path,
        val_list,
        norm_type,
        class_map,
        scaler,
        dataset_type,
        apply_transforms=False,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )
    test_dataset = HyperspectralDataset(
        base_path,
        test_list,
        norm_type,
        class_map,
        scaler,
        dataset_type,
        apply_transforms=False,
        test=True,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    full_class_map = {**class_map, "background": 0}
    return train_loader, val_loader, test_loader, full_class_map
