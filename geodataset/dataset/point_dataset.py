from pathlib import Path
from typing import List, Union

import albumentations
import open3d as o3d
import torch
from geodataset.dataset.base_dataset import BaseLabeledPointCloudCocoDataset
from torch_cluster import knn_graph
from torch_geometric.data import Data


class SegmentationLabeledPointCloudCocoDataset(BaseLabeledPointCloudCocoDataset):
    """
    A dataset class that loads COCO datasets and their associated tiles (point clouds).

    Can be used for semantic segmentation tasks, where the annotations are segmentations.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test'...).
    root_path: str or List[str] or pathlib.Path or List[pathlib.Path]
        The root directory of the dataset.
    transform: albumentations.core.composition.Compose
        A composition of transformations to apply to the tiles and their associated annotations
        (applied in __getitem__).
    """

    def __init__(
        self,
        fold: str,
        root_path: Union[str, List[str], Path, List[Path]],
        label_type: str,
        transform: albumentations.core.composition.Compose = None,
        sample_points: int = None,
        class_fraction: List[float] = None,
        with_rgb: bool = False,
        normalize_height_for_clustering: bool = False,
        k_neighbours:int=16
    ):
        super().__init__(fold=fold, root_path=root_path)

        self.label_type = label_type
        self.transform = transform
        self.sample_points = sample_points
        self.with_rgb = with_rgb
        self.normalize_height_for_clustering = normalize_height_for_clustering
        self.class_weights = torch.zeros(16)
        self.k_neighbours = k_neighbours
        if class_fraction:
            for k, v in class_fraction.items():
                self.class_weights[k] = 1 / v

            self.class_weights /= self.class_weights.sum()

        assert label_type in [
            "semantic",
            "instance",
        ], f"Invalid label type: {label_type}. Must be either 'semantic' or 'instance'."

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying the transform passed to the constructor of the class,
        if any. It also normalizes the tile data between 0 and 1.

        Parameters
        ----------
        idx: int
            The index of the tile to retrieve

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            The transformed tile (image) data, normalized between 0 and 1, and a dictionary containing the annotations
            and metadata of the tile. The dictionary has the following keys:

            - **masks** (list of numpy.ndarray): A list of segmentation masks for the annotations.
            - **labels** (numpy.ndarray): An array of category ids for the annotations (same length as 'masks').
            - **area** (list of float): A list of areas for the segmentation masks annotations (same length as 'masks').
            - **iscrowd** (numpy.ndarray): An array of zeros (same length as 'masks'). Currently not implemented.
            - **image_id** (numpy.ndarray): A single-value array containing the index of the tile.
        """
        tile_info = self.tiles[idx]
        min_x, max_x, min_y, max_y = tile_info["min_x"], tile_info["max_x"], tile_info["min_y"], tile_info["max_y"]

        pcd = o3d.t.io.read_point_cloud(tile_info["path"].as_posix())

        pos_np = pcd.point.positions.numpy()
        pos = torch.Tensor(pos_np)

        colors = torch.Tensor(pcd.point.colors.numpy())

        # Normalized between 0 and 1
        colors = colors/255.0

        # Required due to rounding errors
        min_x = min(min_x, pos_np[:, 0].min())
        max_x = max(max_x, pos_np[:, 0].max())
        min_y = min(min_y, pos_np[:, 1].min())
        max_y = max(max_y, pos_np[:, 1].max())

        min_arr = torch.Tensor([min_x, min_y, 0])
        max_minus_min = torch.Tensor([max_x - min_x, max_y - min_y, 1])

        pos = (pos - min_arr) / max_minus_min

        y = torch.Tensor(getattr(pcd.point, f"{self.label_type}_labels").numpy())

        if self.sample_points:
            indices = torch.randint(0, pos.shape[0], (self.sample_points,))
            pos = pos[indices]
            y = y[indices]
            colors = colors[indices]


        if self.normalize_height_for_clustering:
            normed_points = (pos - pos.mean(dim=0)) / (pos.max(dim=0).values - pos.min(dim=0).values)
            edge_index = knn_graph(normed_points, k=self.k_neighbours, batch=None, loop=True)
        else:
            edge_index = knn_graph(pos, k=self.k_neighbours, batch=None, loop=True)


        if self.with_rgb:
            data = Data(x=colors, pos=pos, edge_index=edge_index, y=y)
        else:
            data = Data(x=pos, pos=pos, edge_index=edge_index, y=y)
        if self.transform:
            data = self.transform(data)
        return data
