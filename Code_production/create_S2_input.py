import time
import os
from datetime import date
import pyproj
from shapely import geometry

from S2data.mosaic import Mosaic, reproject_geometry
from S2data.scene import SceneCollection
from S2data.aoi import AOI

from creodias_finder import query


def write_to_txtfile(path_to_file, log: str):
    with open(path_to_file, "a") as f:
        f.write("\n{}".format(log))


def create_mosaic(aoi_name: str, start_date_query, end_date_query, max_cloud_cover, product_type, path_to_test_folder):
    """
    method implements the full processing chain of how Sentinel 2 input data for GHS-S2Net is created
    Additionally to S2 mosaic, a metadata file and execution runtime file is created.
    """

    start_overall = time.time()

    path_to_time_file = os.path.join(path_to_test_folder, "execution_time.txt")
    with open(path_to_time_file, "w+") as f:
        f.write("Measure execution time of processes")

    # create geometry for aoi in utm and wgs84 projection
    aoi_crs = AOI[aoi_name]["crs"]
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS(aoi_crs)

    point_list = []
    for coords in AOI[aoi_name]["coordinates"]:
        point = geometry.Point(coords[0], coords[1])
        point_list.append(point)

    poly_4326 = geometry.Polygon([[p.x, p.y] for p in point_list])
    aoi_geom_utm = reproject_geometry(utm, wgs84, poly_4326)

    start = time.time()
    results = query.query(
        "Sentinel2",
        start_date=start_date_query,
        end_date=end_date_query,
        geometry=poly_4326.wkt,
        cloudCover=[0, max_cloud_cover]
    )
    time_elapsed = time.time() - start
    write_to_txtfile(path_to_time_file, "Query creodias: {} s".format(time_elapsed))

    # Create Scene Collection Object
    aoi_scene_collection = SceneCollection(results)
    aoi_scene_collection_2a = aoi_scene_collection.extract_scenes_by_product_type(product_type)
    nr_objects = len(aoi_scene_collection_2a.objects)

    # Calculate NDVI per scene
    start = time.time()
    aoi_scene_collection_2a.calculate_mean_ndvi_for_scenes()
    time_elapsed = time.time() - start
    time_elapsed_per_scene = time_elapsed / nr_objects
    write_to_txtfile(path_to_time_file, "Calculating NDVI: {} s; {} s per image; {} nr of scenes".
                     format(time_elapsed, time_elapsed_per_scene, nr_objects))

    # Sort scenes by NDVI
    start = time.time()
    aoi_scene_collection_2a.sort_scenes_by_ndvi()
    time_elapsed = time.time() - start
    time_elapsed_per_scene = time_elapsed / nr_objects
    write_to_txtfile(path_to_time_file, "Sorting by NDVI: {} s; {} s per image".
                     format(time_elapsed, time_elapsed_per_scene))

    # Create Mosaic object
    mosaic_name = "{}_mosaic_{}_m4".format(aoi_name, product_type)
    aoi_mosaic = Mosaic(mosaic_name, path_to_test_folder, aoi_scene_collection_2a, aoi_geom_utm, aoi_crs)

    aoi_mosaic.write_to_metafile("\nNr. of scenes given to mosaic object: {}".format(len(aoi_mosaic.objects)))
    for obj in aoi_mosaic.objects:
        aoi_mosaic.write_to_metafile("ID: {}; Record date: {}; Mean NDVI: {}; Product type: {}, Cloud cover: {}\n"
                                     .format(obj.id, obj.record_date, obj.ndvi_mean, obj.product_type, obj.cloud_cover))

    # Scene selection for mosaic
    start = time.time()
    aoi_mosaic.select_scenes_for_mosaic()
    time_elapsed = time.time() - start
    nr_objects = len(aoi_mosaic.objects)
    write_to_txtfile(path_to_time_file, "Selecting senes for Mosaic: {} s; {} scenes selected".format(time_elapsed, nr_objects))

    aoi_mosaic.write_to_metafile("\nThe following scenes are needed for AOI coverage:")
    for obj in aoi_mosaic.objects:
        aoi_mosaic.write_to_metafile("ID: {}; Record date: {}; Mean NDVI {}\n"
                                     .format(obj.id, obj.record_date, obj.ndvi_mean))

    # create image stacks
    start = time.time()
    aoi_mosaic.create_stacks(output_path_base=path_to_test_folder)
    time_elapsed = time.time() - start
    write_to_txtfile(path_to_time_file, "Creating stacks: {} s for {} scenes".format(time_elapsed, nr_objects))

    aoi_mosaic.write_to_metafile("The following scenes were baseline 4 corrected:")
    for obj in aoi_mosaic.objects:
        if obj.stack_baseline4_corrected:
            aoi_mosaic.write_to_metafile("ID: {}; Record Date: {}".format(obj.id, obj.record_date))
        else:
            aoi_mosaic.write_to_metafile("ID: {}; Record Date: {} - No correction needed".format(obj.id, obj.record_date))

    # harmonize reference systems
    aoi_mosaic.write_to_metafile("\nThe following scenes were reprojected to {}:".format(aoi_mosaic.crs))
    start = time.time()
    aoi_mosaic.harmonize_reference_systems_of_stacks()
    time_elapsed = time.time() - start
    write_to_txtfile(path_to_time_file, "Harmonize stacks: {} s for {} scenes".format(time_elapsed, nr_objects))

    # save mosaic
    start = time.time()
    aoi_mosaic.create_mosaic()
    time_elapsed = time.time() - start
    write_to_txtfile(path_to_time_file, "Save mosaic: {} s".format(time_elapsed))

    end_overall = time.time()
    time_elapsed_overall = end_overall - start_overall
    write_to_txtfile(path_to_time_file, "Overall time elapsed: {} s".format(time_elapsed_overall))


if __name__ == "__main__":

    aoi_name = "AOI2"  # must be specified in aoi.py file
    search_start_date = date(2021, 9, 1)
    search_end_date = date(2022, 8, 1)
    max_cloud_cover = 2
    processing_level = "L1C"  # L2A also possible
    path_to_folder = "<specify folder in which all files shall be saved>"

    create_mosaic(aoi_name,
                  search_start_date,
                  search_end_date,
                  max_cloud_cover,
                  processing_level,
                  path_to_folder)

