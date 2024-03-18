import json
from pathlib import Path
from typing import List
from warnings import warn

import pandas as pd
import rasterio
from shapely import box, Polygon
import geopandas as gpd
from tqdm import tqdm

from geodataset.utils import TileNameConvention, apply_affine_transform


class DetectionAggregator:
    def __init__(self,
                 geojson_output_path: Path,
                 labels_gdf: gpd.GeoDataFrame,
                 tiles_extent_gdf: gpd.GeoDataFrame,
                 min_score_threshold: float = None,
                 intersect_remove_ratio: float = 0.95):

        self.geojson_output_path = geojson_output_path
        self.labels_gdf = labels_gdf
        self.tiles_extent_gdf = tiles_extent_gdf
        self.min_score_threshold = min_score_threshold
        self.intersect_remove_ratio = intersect_remove_ratio

        assert self.labels_gdf.crs, "The provided labels_gdf doesn't have a CRS."
        assert 'tile_id' in self.labels_gdf, "The provided labels_gdf doesn't have a 'tile_id' column."
        assert self.tiles_extent_gdf.crs, "The provided tiles_extent_gdf doesn't have a CRS."
        assert 'tile_id' in self.tiles_extent_gdf, "The provided tiles_extent_gdf doesn't have a 'tile_id' column."

        self.remove_low_score_boxes()
        self.remove_intersecting_boxes_with_tiles()
        self.labels_gdf.to_file(str(geojson_output_path), driver="GeoJSON")

    @classmethod
    def from_coco(cls,
                  geojson_output_path: Path,
                  tiles_folder_path: Path,
                  coco_json_path: Path,
                  min_score_threshold: float = None,
                  intersect_remove_ratio: float = 0.95):

        # Read the JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Reading the boxes
        tile_ids_to_boxes = {}
        tile_ids_to_scores = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']

            xmin = annotation['bbox'][0]
            xmax = annotation['bbox'][0] + annotation['bbox'][2]
            ymin = annotation['bbox'][1]
            ymax = annotation['bbox'][1] + annotation['bbox'][3]
            annotation_box = box(xmin, ymin, xmax, ymax)

            if "score" in annotation:
                score = annotation["score"]
            elif ("other_attributes" in annotation
                  and annotation["other_attributes"]
                  and "score" in annotation["other_attributes"]):
                score = annotation["other_attributes"]["score"]
            else:
                score = None

            if image_id in tile_ids_to_boxes:
                tile_ids_to_boxes[image_id].append(annotation_box)
                tile_ids_to_scores[image_id].append(score)
            else:
                tile_ids_to_boxes[image_id] = [annotation_box]
                tile_ids_to_scores[image_id] = [score]

        for image_id, scores in tile_ids_to_scores.items():
            if all(score is None for score in scores):
                # All scores for this image_id are None
                tile_ids_to_scores = None
                break
            elif any(score is None for score in scores):
                # Some scores for this image_id are None (implies at least one score is not None)
                tile_ids_to_scores = None
                break
            else:
                # No scores for this image_id are None (all scores are not None)
                continue

        tile_ids_to_path = {}
        for image in coco_data['images']:
            image_path = tiles_folder_path / image['file_name']
            assert image_path.exists(), (f"Could not find the tile '{image['file_name']}'"
                                         f" in tiles_folder_path='{tiles_folder_path}'")
            tile_ids_to_path[image['id']] = image_path

        all_boxes_gdf, all_tiles_extents_gdf = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                                                 tile_ids_to_boxes=tile_ids_to_boxes,
                                                                                 tile_ids_to_scores=tile_ids_to_scores)

        return cls(geojson_output_path=geojson_output_path,
                   labels_gdf=all_boxes_gdf,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   min_score_threshold=min_score_threshold,
                   intersect_remove_ratio=intersect_remove_ratio)

    @classmethod
    def from_boxes(cls,
                   geojson_output_path: Path,
                   tiles_paths: List[Path],
                   boxes: List[List[box]],
                   scores: List[List[float]] or None,
                   min_score_threshold: float = None,
                   intersect_remove_ratio: float = 0.95):

        assert len(tiles_paths) == len(boxes), ("The number of tiles_paths must be equal than the number of lists "
                                                "in boxes (one list of boxes per tile).")

        ids = list(range(len(tiles_paths)))
        tile_ids_to_path = {k: v for k, v in zip(ids, tiles_paths)}
        tile_ids_to_boxes = {k: v for k, v in zip(ids, boxes)}
        tile_ids_to_scores = {k: v for k, v in zip(ids, scores)} if scores else None

        all_boxes_gdf, all_tiles_extents_gdf = DetectionAggregator.prepare_boxes(tile_ids_to_path=tile_ids_to_path,
                                                                                 tile_ids_to_boxes=tile_ids_to_boxes,
                                                                                 tile_ids_to_scores=tile_ids_to_scores)

        return cls(geojson_output_path=geojson_output_path,
                   labels_gdf=all_boxes_gdf,
                   tiles_extent_gdf=all_tiles_extents_gdf,
                   min_score_threshold=min_score_threshold,
                   intersect_remove_ratio=intersect_remove_ratio)

    @staticmethod
    def prepare_boxes(tile_ids_to_path: dict, tile_ids_to_boxes: dict, tile_ids_to_scores: dict):
        images_without_crs = 0
        all_gdfs_boxes = []
        all_gdfs_tiles_extents = []
        for tile_id in tile_ids_to_boxes.keys():
            image_path = tile_ids_to_path[tile_id]

            src = rasterio.open(image_path)

            if not src.crs or not src.transform:
                images_without_crs += 1
                continue

            # Making sure the Tile respects the convention
            TileNameConvention.parse_name(Path(image_path).name)

            gdf_boxes = gpd.GeoDataFrame(geometry=tile_ids_to_boxes[tile_id])
            gdf_boxes['geometry'] = gdf_boxes['geometry'].astype(object).apply(
                lambda geom: apply_affine_transform(geom, src.transform)
            )
            gdf_boxes.crs = src.crs
            gdf_boxes['tile_id'] = tile_id
            if tile_ids_to_scores:
                gdf_boxes['score'] = tile_ids_to_scores[tile_id]
            all_gdfs_boxes.append(gdf_boxes)

            bounds = src.bounds
            # Create a polygon from the bounds
            tile_extent_polygon = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom)
            ])
            gdf_tile_extent = gpd.GeoDataFrame(geometry=[tile_extent_polygon], crs=src.crs)
            gdf_tile_extent['tile_id'] = tile_id
            all_gdfs_tiles_extents.append(gdf_tile_extent)

        if images_without_crs > 0:
            warn(f"The aggregator found {images_without_crs} tiles without a CRS or transform")

        tiles_crs = [gdf.crs for gdf in all_gdfs_tiles_extents]
        n_crs = len(set(tiles_crs))
        if n_crs > 1:
            common_crs = tiles_crs[0].crs
            warn(f"Multiple CRS were detected (n={n_crs}), so re-projecting all boxes of all tiles to the"
                 f" same common CRS '{common_crs}', chosen from one of the tiles.")
            all_gdfs_boxes = [gdf.to_crs(common_crs) for gdf in all_gdfs_boxes if gdf.crs != common_crs]
            all_gdfs_tiles_extents = [gdf.to_crs(common_crs) for gdf in all_gdfs_tiles_extents if gdf.crs != common_crs]

        all_boxes_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_boxes, ignore_index=True))
        all_tiles_extents_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs_tiles_extents, ignore_index=True))

        return all_boxes_gdf, all_tiles_extents_gdf

    def remove_low_score_boxes(self):
        if self.min_score_threshold:
            if "score" in self.labels_gdf:
                n_before = len(self.labels_gdf)
                self.labels_gdf.drop(self.labels_gdf[self.labels_gdf["score"] < self.min_score_threshold].index, inplace=True)
                n_after = len(self.labels_gdf)
                print(f"Removed {n_before-n_after} out of {n_before} labels"
                      f" as they were under the min_score_threshold={self.min_score_threshold}.")
            else:
                warn(f"Could not apply min_score_threshold={self.min_score_threshold} as scores were not found"
                     f" in the labels_gdf. If you instanced the Aggregator from a COCO file,"
                     f" please make sure that ALL annotations have a 'score' key or a ['other_attributes']['score']"
                     f" nested keys. If you instanced the Aggregator from a list of boxes for each tile,"
                     f" make sure you also pass the associated list of scores for each tile.")

    def remove_intersecting_boxes_with_tiles(self):
        """
        Removes labels that intersect more than a given percentage with any other label within the same tile bounds.
        """

        n_before = len(self.labels_gdf)

        tile_iterator = tqdm(self.tiles_extent_gdf.iterrows(), total=self.tiles_extent_gdf.shape[0],
                             desc=f"Removing boxes intersecting at > {self.intersect_remove_ratio*100}%")

        for _, tile_row in tile_iterator:
            tile_id = tile_row['tile_id']
            tile_geom = tile_row.geometry

            # Find labels intersecting the current tile
            intersecting_labels = gpd.sjoin(self.labels_gdf,
                                            gpd.GeoDataFrame({'geometry': [tile_geom]}, crs=self.labels_gdf.crs),
                                            how='inner', predicate='intersects')

            if intersecting_labels.empty:
                continue

            # Finding the pairs of current tile labels and adjacent/overlapping tiles labels that intersect
            tile_labels = intersecting_labels[intersecting_labels["tile_id"] == tile_id]
            other_tiles_labels = intersecting_labels[intersecting_labels["tile_id"] != tile_id]
            label_pairs = gpd.sjoin(tile_labels, other_tiles_labels, how="inner", predicate="intersects",
                                    lsuffix='left2', rsuffix='right2')

            if label_pairs.empty:
                continue

            # Calculate the intersection area for each pair
            label_pairs["intersection_area"] = label_pairs.apply(
                lambda row: row.geometry.intersection(
                    other_tiles_labels.loc[row["index_right2"], "geometry"]).area, axis=1
            )

            # Calculate the intersection area ratio for left (the current tile labels)
            # and right (the adjacent/overlapping tiles labels)
            label_pairs["intersection_ratio_left"] = label_pairs["intersection_area"] / label_pairs.geometry.area
            label_pairs["intersection_ratio_right"] = label_pairs.apply(
                lambda row:
                row["intersection_area"] / other_tiles_labels.loc[row["index_right2"], "geometry"].area, axis=1
            )

            # Identify labels to remove based on intersection ratio
            labels_to_remove = label_pairs[
                (label_pairs["intersection_ratio_left"] >= self.intersect_remove_ratio) &
                (label_pairs["intersection_ratio_right"] >= self.intersect_remove_ratio)
            ]["index_right2"].unique()

            self.labels_gdf.drop(labels_to_remove, inplace=True, errors='ignore')

        n_after = len(self.labels_gdf)
        print(f"Removed {n_before-n_after} out of {n_before} labels after"
              f" applying intersect_remove_ratio={self.intersect_remove_ratio}.")
