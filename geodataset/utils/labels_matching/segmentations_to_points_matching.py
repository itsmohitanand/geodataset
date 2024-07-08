from warnings import warn

import cv2
import geopandas as gpd
import numpy as np
from shapely import MultiPolygon, Polygon, box
from shapely.affinity import rotate
import largestinteriorrectangle as lir


def normalize_polygon(polygon, target_range=(0, 100)):
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate the scaling factor
    scale_x = (target_range[1] - target_range[0]) / (maxx - minx)
    scale_y = (target_range[1] - target_range[0]) / (maxy - miny)

    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_x, scale_y)

    # Calculate the translation values
    trans_x = -minx * scale + target_range[0]
    trans_y = -miny * scale + target_range[0]

    def scale_and_translate(x, y, scale, trans_x, trans_y):
        return x * scale + trans_x, y * scale + trans_y

    # Apply the transformation to all coordinates
    normalized_coords = [scale_and_translate(x, y, scale, trans_x, trans_y) for x, y in polygon.exterior.coords]

    # Create a new polygon with the normalized coordinates
    normalized_polygon = Polygon(normalized_coords)

    return normalized_polygon, scale, trans_x, trans_y


def revert_normalization(normalized_polygon, scale, trans_x, trans_y):
    def unscale_and_untranslate(x, y, scale, trans_x, trans_y):
        return (x - trans_x) / scale, (y - trans_y) / scale

    # Apply the inverse transformation to all coordinates
    original_coords = [unscale_and_untranslate(x, y, scale, trans_x, trans_y) for x, y in normalized_polygon.exterior.coords]

    # Create a new polygon with the original coordinates
    original_polygon = Polygon(original_coords)

    return original_polygon


def largest_inner_rectangle(geometry):
    # Convert MultiPolygon to Polygon by taking the largest part
    if isinstance(geometry, MultiPolygon):
        geometry = max(geometry.geoms, key=lambda a: a.area)

    # Normalize the polygon to a target range
    normalized_polygon, scale, trans_x, trans_y = normalize_polygon(geometry)

    # Polygon to boolean mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    exterior_coords = np.array([list(normalized_polygon.exterior.coords)]).astype(np.int32)
    cv2.fillPoly(mask, exterior_coords, 1)
    mask = mask.astype(bool)

    rectangle = lir.lir(mask)

    rectangle_shapely = box(rectangle[0], rectangle[1], rectangle[0]+rectangle[2], rectangle[1]+rectangle[3])

    # Revert transform applied in normalize_polygon
    best_rectangle = revert_normalization(rectangle_shapely, scale, trans_x, trans_y)

    return best_rectangle


def segmentations_to_points_matching(segmentations, truth_points, output_path, relative_dist_to_inner_rectangle_centroid_threshold):
    # Check if the segmentations and truth_boxes are in the same crs
    if segmentations.crs != truth_points.crs:
        segmentations = segmentations.to_crs(truth_points.crs)

    # only keep segmentations intersecting the truth points
    segmentations['geometry'] = segmentations.buffer(0)
    segmentations = segmentations[segmentations.intersects(truth_points.unary_union)]

    # get the largest rectangle that fits inside the segmentation
    print("Calculating largest inner rectangles...")
    segmentations['largest_inner_rectangle'] = segmentations['geometry'].astype(object).apply(largest_inner_rectangle)
    segmentations_largest_inner_rectangle = segmentations.copy(deep=True)
    segmentations_largest_inner_rectangle['geometry'] = segmentations_largest_inner_rectangle['largest_inner_rectangle']
    print("Done.\n")

    # get the max distance between 2 points of the segmentation
    segmentations_largest_inner_rectangle['max_point_distance'] = segmentations_largest_inner_rectangle['geometry'].apply(lambda x: x.exterior.distance(x.centroid))

    labeled_segmentations = []

    for i, point in truth_points.iterrows():
        # get largest_inner_rectangle intersecting the point
        intersecting_segmentations = segmentations_largest_inner_rectangle[segmentations_largest_inner_rectangle.intersects(point.geometry)].copy(deep=True)

        # get the distance to the centroid normalized by the max distance between 2 points of the segmentation
        intersecting_segmentations['dist_to_centroid'] = intersecting_segmentations.centroid.distance(point.geometry)
        intersecting_segmentations['relative_dist_to_inner_rectangle_centroid'] = intersecting_segmentations['dist_to_centroid'] / intersecting_segmentations['max_point_distance']

        # only keep the segmentation with the smallest distance to the centroid
        segmentations_keep = intersecting_segmentations[intersecting_segmentations['relative_dist_to_inner_rectangle_centroid'] < relative_dist_to_inner_rectangle_centroid_threshold].index
        if len(segmentations_keep) > 0:
            best_segmentation_id = intersecting_segmentations.loc[segmentations_keep].sort_values('relative_dist_to_inner_rectangle_centroid').index[0]
            labeled_segmentation = point.copy(deep=True)
            labeled_segmentation.geometry = segmentations.loc[best_segmentation_id].geometry
            labeled_segmentations.append(labeled_segmentation)

    if len(labeled_segmentations) == 0:
        raise Exception("No segmentations matched the truth points. Please check the relative distance to the centroid.")

    labeled_segmentations_gdf = gpd.GeoDataFrame(labeled_segmentations, crs=truth_points.crs)

    print("Number of points in truth:", len(truth_points))
    print("Number of predicted segmentations matched to truth segmentations:", len(labeled_segmentations_gdf))

    labeled_segmentations_gdf.to_file(output_path)

    print(f'Labeled segmentations saved to {output_path}.\n')
    warn(f'Please make sure to double check the segments in a GIS software before using them!!!')


if __name__ == '__main__':
    segmentations_path = '/infer/20231213_zf2campirana_mini3pro_rgb_aligned/segmenter_aggregator_output/20231213_zf2campirana_mini3pro_rgb_aligned_gr0p08_infersegmenteraggregator.gpkg'
    truth_points_path = '/Data/raw/brazil_zf2/20240131_zf2campinarana_labels_points_species.gpkg'
    relative_dist_to_inner_rectangle_centroid_threshold = 0.8
    output_path = f'/Data/segmentations_SAM_matched/20231213_zf2campirana_mini3pro_rgb_aligned_SAMthreshold{str(relative_dist_to_inner_rectangle_centroid_threshold).replace(".", "p")}.gpkg'

    segmentations = gpd.read_file(segmentations_path)
    truth_points = gpd.read_file(truth_points_path)

    segmentations_to_points_matching(segmentations, truth_points, output_path, relative_dist_to_inner_rectangle_centroid_threshold)
