import sys
import warnings
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Union

import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from laspy import CopcReader
from shapely import box
from shapely.geometry import Point
from tqdm import tqdm

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_base import AOIBaseFromGeoFileInCRS
from geodataset.geodata.point_cloud_tile import (
    PointCloudTileMetadata,
    PointCloudTileMetadataCollection,
)
from geodataset.utils import strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import (
    PointCloudTileNameConvention,
    validate_and_convert_product_name,
)


class PointCloudTilerizer:
    """
    Class to tilerize point cloud data from a point cloud file and associated AOIs.

    Parameters
    ----------
    point_cloud_path : Union[str, Path]
        Path to the point cloud file.
    output_path : Union[str, Path]
        Path to the output folder where the tiles and annotations will be saved.
    tiles_metadata : Union[TileMetadataCollection, None], optional
        Metadata of the tiles, by default None
    aois_config : Union[AOIFromPackageConfig, None], optional
        Configuration for AOIs, by default None
    tile_side_length: float, optional
        Side length of the tile, by default None
    keep_dims : List[str], optional
        List of dimensions to keep, by default None
    downsample_voxel_size : float, optional
        Voxel size for downsampling, by default None
    verbose : bool, optional
        Verbose mode, by default False
    tile_overlap : float, optional
        Overlap between tiles, by default 1.0
    max_tile : int, optional
        Maximum number of tiles, by default 5000
    force : bool, optional
        Force tilerization, by default False
    """

    def __init__(
        self,
        point_cloud_path: Union[str, Path],
        output_path: Union[str, Path],
        tiles_metadata: Union[PointCloudTileMetadataCollection, None] = None,
        aois_config: Union[AOIFromPackageConfig, None] = None,
        tile_side_length: float = None,
        keep_dims: List[str] = None,
        downsample_voxel_size: float = None,
        verbose: bool = False,
        tile_overlap: float = None,
        max_tile: int = 5000,
        force: bool = False,
    ):
        assert not (
            tile_side_length and tiles_metadata
        ), "Only one of tile_side_length or tiles_metadata should be provided"
        if tile_overlap and tiles_metadata:
            print("Tile overlap will be ignored, as tile metadata is also provided")

        self.point_cloud_path = Path(point_cloud_path)
        self.product_name = validate_and_convert_product_name(
            strip_all_extensions_and_path(self.point_cloud_path)
        )
        self.tiles_metadata = tiles_metadata
        self.output_path = Path(output_path)
        self.aois_config = aois_config
        self.tile_side_length = tile_side_length
        self.tile_overlap = tile_overlap
        self.keep_dims = keep_dims if keep_dims is not None else "ALL"
        self.downsample_voxel_size = downsample_voxel_size
        self.verbose = verbose
        self.force = force

        if self.tiles_metadata is None:
            assert (
                self.tile_overlap is not None
            ), "Tile Overlap is required if tile metadata is not provided"

            self.populate_tiles_metadata()

        # Just a sanity check to make sure that the number of tiles is less than the max_tile
        assert (
            len(self.tiles_metadata) < max_tile
        ), f"Number of max possible tiles {len(self.tiles_metadata)} exceeds the maximum number of tiles {max_tile}"

        self.pc_tiles_folder_path = (
            self.output_path / f"pc_tiles_{self.downsample_voxel_size}"
            if self.downsample_voxel_size
            else self.output_path / "pc_tiles"
        )

        # Processing AOIs
        if self.aois_config is None:
            raise Exception(
                "Please provide an aoi_config. Currently don't support 'None'."
            )
        self.aoi_engine = AOIBaseFromGeoFileInCRS(aois_config)
        self.aois_in_both = self._check_aois_in_both_tiles_and_config()

        self.create_folder()

    def _check_aois_in_both_tiles_and_config(self):
        aois_in_both = set(self.tiles_metadata.aois) and set(
            self.aoi_engine.loaded_aois.keys()
        )

        if set(self.tiles_metadata.aois) != set(self.aoi_engine.loaded_aois.keys()):
            warnings.warn(
                f"AOIs in the AOI engine and the tiles metadata do not match, got"
                f" {set(self.tiles_metadata.aois)} from tiles_metadata and"
                f" {set(self.aoi_engine.loaded_aois.keys())} from aoi_config."
                f" Will only save tiles for the AOIs present in both: "
                f"{aois_in_both}."
            )

        return aois_in_both

    def populate_tiles_metadata(self):
        name_convention = PointCloudTileNameConvention()

        with CopcReader.open(self.point_cloud_path) as reader:
            min_x, min_y = reader.header.x_min, reader.header.y_min
            max_x, max_y = reader.header.x_max, reader.header.y_max

            tiles_metadata = []
            tile_id = 0
            for i, x in enumerate(
                np.arange(min_x, max_x, self.tile_side_length * self.tile_overlap)
            ):
                for j, y in enumerate(
                    np.arange(min_y, max_y, self.tile_side_length * self.tile_overlap)
                ):
                    x_bound = [x, x + self.tile_side_length]
                    y_bound = [y, y + self.tile_side_length]

                    tile_name = name_convention.create_name(
                        product_name=self.product_name, row=i, col=j
                    )

                    tile_md = PointCloudTileMetadata(
                        x_bound=x_bound,
                        y_bound=y_bound,
                        crs=reader.header.parse_crs(),
                        tile_name=tile_name,
                        tile_id=tile_id,
                    )

                    tiles_metadata.append(tile_md)
                    tile_id += 1

            self.tiles_metadata = PointCloudTileMetadataCollection(
                tiles_metadata, product_name=self.tiles_metadata.product_name
            )

        print(f"Number of tiles generated: {len(self.tiles_metadata)}")

        if self.verbose:
            print(f"Point cloud x range: {min_x} - {max_x}")
            print(f"Point cloud y range: {min_y} - {max_y}")

        if not self.force:
            cont = str(input("Do you want to continue with the tilerization? (y/n)"))
            if cont.lower() != "y":
                print("Aborting tilerization")
                sys.exit()

    def create_folder(self):
        self.pc_tiles_folder_path.mkdir(parents=True, exist_ok=True)
        for aoi in self.aois_in_both:
            (self.pc_tiles_folder_path / aoi).mkdir(parents=True, exist_ok=True)

    def query_tile(self, tile_md, reader):
        reader_crs = reader.header.parse_crs()

        tile_md_epsg = tile_md.crs.to_epsg()
        reader_crs_epsg = reader_crs.to_epsg()
        if tile_md_epsg != reader_crs_epsg:
            min_max_gdf = gpd.GeoDataFrame(
                geometry=[
                    box(tile_md.min_x, tile_md.min_y, tile_md.max_x, tile_md.max_y)
                ],
                crs=tile_md.crs,
            )

            min_max_gdf = min_max_gdf.to_crs(reader_crs)

            bounds = min_max_gdf.geometry[0].bounds
            mins = np.array([bounds[0], bounds[1]])
            maxs = np.array([bounds[2], bounds[3]])
        else:
            mins = np.array([tile_md.min_x, tile_md.min_y])
            maxs = np.array([tile_md.max_x, tile_md.max_y])

        b = laspy.copc.Bounds(mins=mins, maxs=maxs)
        data = reader.query(b)

        return data

    def _tilerize(self):
        # Queue for tiles to be processed
        processing_queue = Queue()

        # Add tiles to the processing queue
        tile_md_list = [tile_md for tile_md in self.tiles_metadata]
        for tile_md in tile_md_list:
            processing_queue.put(tile_md)

        # List to hold only successfully processed tiles
        self.new_tile_md_list = []  # Defined as a class attribute to access within threads

        # Number of threads dedicated to processing (each thread will also write after processing)
        num_processing_threads = 4

        # Start threads for processing and writing
        with tqdm(total=len(tile_md_list), desc="Processing tiles") as progress_bar:
            # Start threads for processing and writing
            processors = []
            for i in range(num_processing_threads):
                processor_thread = Thread(
                    target=self._process_and_write_tile,
                    args=(processing_queue, progress_bar),
                )
                processor_thread.start()
                processors.append(processor_thread)

            # Wait for all processing tasks to complete
            processing_queue.join()

        # Print summary
        print(
            f"Finished tilerizing. Number of tiles generated: {len(self.new_tile_md_list)}."
        )
        print(
            f"Number of tiles removed: {len(tile_md_list) - len(self.new_tile_md_list)}."
        )

        # Update tiles metadata
        self.tiles_metadata = PointCloudTileMetadataCollection(
            self.new_tile_md_list, product_name=self.tiles_metadata.product_name
        )

    def _process_and_write_tile(self, processing_queue, progress_bar):
        while not processing_queue.empty():
            tile_md = processing_queue.get()
            try:
                # Process the tile
                pcd = self._process_tile(tile_md)
                if pcd is not None:
                    # Write the processed tile data immediately
                    self._write_tile(pcd, tile_md)
                    # Append to new_tile_md_list if processing was successful
                    self.new_tile_md_list.append(tile_md)

                progress_bar.update(1)
            except Exception as e:
                print(f"Error processing or writing tile {tile_md}: {e}")
            finally:
                processing_queue.task_done()

    def _process_tile(self, tile_md):
        with CopcReader.open(self.point_cloud_path) as reader:
            data = self.query_tile(tile_md, reader)

            if len(data) == 0:
                return None

            map_to_tensor = self._laspy_to_o3d(data, keep_dims=self.keep_dims.copy())

            if self.downsample_voxel_size:
                map_to_tensor = self._downsample_tile(
                    map_to_tensor,
                    self.downsample_voxel_size,
                    keep_dims=self.keep_dims.copy(),
                )

            map_to_tensor = self._keep_unique_points(map_to_tensor)
            map_to_tensor = self._remove_points_outside_aoi(
                map_to_tensor, tile_md, reader.header.parse_crs()
            )

            return map_to_tensor

    def _write_tile(self, map_to_tensor, tile_md):
        try:
            downsampled_tile_path = (
                self.pc_tiles_folder_path / f"{tile_md.aoi}/{tile_md.tile_name}"
            )
            for k, v in map_to_tensor.items():
                if v.shape[1] == 1:
                    map_to_tensor[k] = v.reshape((-1, 1))

            pcd = o3d.t.geometry.PointCloud(map_to_tensor)
            o3d.t.io.write_point_cloud(str(downsampled_tile_path), pcd)
        except Exception as e:
            print(f"Error writing tile {tile_md}: {e}")

    def tilerize(
        self,
    ):
        self._tilerize()
        self.plot_aois()

    def plot_aois(self) -> None:
        fig, ax = self.tiles_metadata.plot()

        palette = {
            "blue": "#26547C",
            "red": "#EF476F",
            "yellow": "#FFD166",
            "green": "#06D6A0",
        }

        if "train" in self.aoi_engine.loaded_aois:
            self.aoi_engine.loaded_aois["train"].plot(
                ax=ax, color=palette["blue"], alpha=0.5
            )
        if "valid" in self.aoi_engine.loaded_aois:
            self.aoi_engine.loaded_aois["valid"].plot(
                ax=ax, color=palette["green"], alpha=0.5
            )
        if "test" in self.aoi_engine.loaded_aois:
            self.aoi_engine.loaded_aois["test"].plot(
                ax=ax, color=palette["red"], alpha=0.5
            )

        plt.savefig(self.output_path / f"{self.tiles_metadata.product_name}_aois.png")

    def _laspy_to_o3d(self, pc_file: Path, keep_dims: List[str]):
        float_type = np.float32
        dimensions = list(pc_file.point_format.dimension_names)

        if keep_dims == "ALL":
            keep_dims = dimensions

        assert all([dim in keep_dims for dim in ["X", "Y", "Z"]])
        assert all([dim in dimensions for dim in keep_dims])

        pc = np.ascontiguousarray(
            np.vstack([pc_file.x, pc_file.y, pc_file.z]).T.astype(float_type)
        )

        map_to_tensors = {}
        map_to_tensors["positions"] = pc.astype(float_type)

        keep_dims.remove("X")
        keep_dims.remove("Y")
        keep_dims.remove("Z")

        if "red" in keep_dims and "green" in keep_dims and "blue" in keep_dims:
            colors = np.ascontiguousarray(
                np.vstack([pc_file.red, pc_file.green, pc_file.blue]).T.astype(
                    float_type
                )
            )

            pc_colors = np.round(colors / 255)
            map_to_tensors["colors"] = pc_colors.astype(np.uint8)

            keep_dims.remove("red")
            keep_dims.remove("blue")
            keep_dims.remove("green")

        for dim in keep_dims:
            dim_value = np.ascontiguousarray(pc_file[dim]).astype(float_type)

            assert len(dim_value.shape) == 1
            dim_value = dim_value.reshape(-1, 1)
            map_to_tensors[dim] = dim_value

        return map_to_tensors

    def _remove_black_points(self, tile, tile_size):
        is_rgb = self.raster.data.shape[0] == 3
        is_rgba = self.raster.data.shape[0] == 4
        skip_ratio = self.ignore_black_white_alpha_tiles_threshold

        # Checking if the tile has more than a certain ratio of white, black, or alpha pixels.
        if is_rgb:
            if np.sum(tile.data == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
        elif is_rgba:
            if np.sum(tile.data[:-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[:-1] == 255) / (tile_size * tile_size * 3) > skip_ratio:
                return True
            if np.sum(tile.data[-1] == 0) / (tile_size * tile_size * 3) > skip_ratio:
                return True

        return False

    def _keep_unique_points(self, map_to_tensors):
        positions = map_to_tensors["positions"]
        unique_pc, ind = np.unique(positions, axis=0, return_index=True)

        new_map_to_tensors = {"positions": unique_pc}

        for key, value in map_to_tensors.items():
            if key != "positions":
                new_map_to_tensors[key] = value[ind]

        if self.verbose:
            n_removed = positions.shape[0] - unique_pc.shape[0]
            percentage_removed = (n_removed / positions.shape[0]) * 100
            print(f"Removed {n_removed} duplicate points ({percentage_removed:.2f}%)")
        return new_map_to_tensors

    def _remove_points_outside_aoi(self, map_to_tensor, tile_md, reader_crs):
        aoi_gdf = self.aoi_engine.loaded_aois[tile_md.aoi]
        aoi_gdf = aoi_gdf.to_crs(reader_crs)
        aoi_bounds = aoi_gdf.total_bounds

        positions = map_to_tensor["positions"]

        points_min_bound = np.min(positions, axis=0)
        points_max_bound = np.max(positions, axis=0)

        # Check if point cloud bounds are completely within the tile bounds
        if (
            points_min_bound[0] >= aoi_bounds[0]
            and points_max_bound[0] <= aoi_bounds[2]
            and points_min_bound[1] >= aoi_bounds[1]
            and points_max_bound[1] <= aoi_bounds[3]
        ):
            # If all points are within the AOI bounds, return the original point cloud
            return map_to_tensor

        geopoints = gpd.GeoDataFrame(positions)
        geopoints = gpd.GeoDataFrame(geopoints.apply(lambda x: Point(x), axis=1))
        geopoints.columns = ["points"]
        geopoints = geopoints.set_geometry("points")

        geopoints.crs = reader_crs

        points_in_aoi = gpd.sjoin(geopoints, aoi_gdf, predicate="within")

        positions_in_aoi = (
            points_in_aoi["points"].apply(lambda x: [x.x, x.y, x.z]).values
        )
        positions_in_aoi = np.array(positions_in_aoi.tolist())

        indices_in_aoi = points_in_aoi.index.values

        new_map_to_tensor = {"positions": positions_in_aoi}

        for k, v in map_to_tensor.items():
            if k != "positions":
                new_map_to_tensor[k] = v[indices_in_aoi]

        return new_map_to_tensor

    def _downsample_tile(self, map_to_tensors, voxel_size, keep_dims) -> None:
        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        if "red" in keep_dims and "green" in keep_dims and "blue" in keep_dims:
            keep_dims.remove("red")
            keep_dims.remove("green")
            keep_dims.remove("blue")

            keep_dims.append("colors")

        if "X" in keep_dims and "Y" in keep_dims and "Z" in keep_dims:
            keep_dims.remove("X")
            keep_dims.remove("Y")
            keep_dims.remove("Z")

            keep_dims.append("positions")

        map_to_tensors = {}
        for dims in keep_dims:
            map_to_tensors[dims] = getattr(pcd.point, dims).numpy()

        return map_to_tensors
