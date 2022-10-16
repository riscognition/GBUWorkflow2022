import os
import rasterio
from rasterio.merge import merge
import numpy as np
from shapely import geometry
from shapely.ops import transform, unary_union
import pyproj
from osgeo import gdal


def reproject_geometry(target_crs: object, source_crs: object, geometry: object):
    """
    :param target_crs: pyproj crs object
    :param source_crs: pyproj crs object
    :param geometry: shapely object
    :return: reprojected shapely object
    """
    project = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
    geom_reproject = transform(project, geometry)
    return geom_reproject


class Mosaic:

    def __init__(self,  mosaic_name: str, base_path: str, scenes: object, aoi_geom: object, aoi_crs: object):
        self.mosaic_name = mosaic_name
        self.objects = scenes.objects
        self.geometry = aoi_geom
        self.crs = aoi_crs
        self.path_to_mosaic = os.path.join(base_path, "{}.tif".format(mosaic_name))
        self.path_to_meta = os.path.join(base_path, "{}_meta.txt".format(self.mosaic_name))

        with open(self.path_to_meta, "w") as f:
            f.write("Meta data for creation of {}".format(self.mosaic_name))

    @ staticmethod
    def warp(path_to_image: str, target_crs: object, output_path: str):
        """
        :param path_to_image: path to image which shall be reprojected
        :param target_crs: as Pyroj object
        :param output_path: path where image shall be saved
        :return:
        """
        input_raster = gdal.Open(path_to_image)
        gdal.Warp(output_path, input_raster, dstSRS=target_crs)

    @staticmethod
    def save_raster(image: np.array, meta: dict, output_path: str):
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(image)

    def write_to_metafile(self, log: str):
        """
        :param log: text which is added to metadata file
        """
        with open(self.path_to_meta, "a") as f:
            f.write("\n{}".format(log))

    def select_scenes_for_mosaic(self):
        """
        method extracts scenes which are needed for covering mosaic boundaries.
        As many scenes are chosen as necessary with respect to scene ordering.
        Scenes associated with Mosaic (self.objects) are updated when calling this method.
        """
        scenes_for_mosaic = []
        search_geom = geometry.Polygon()
        for obj in self.objects:
            obj_geometry = geometry.Polygon(obj.bounds)
            if not search_geom.covers(obj_geometry):
                search_geom = unary_union([obj_geometry, search_geom])
                scenes_for_mosaic.append(obj)
                wgs84 = pyproj.CRS('EPSG:4326')
                utm = pyproj.CRS(self.crs)
                ext_utm = reproject_geometry(utm, wgs84, search_geom)
                if ext_utm.covers(self.geometry):
                    self.objects = scenes_for_mosaic
                    return None

    def harmonize_reference_systems_of_stacks(self):
        """
        method re-projects stacks to target reference system if needed
        """
        for obj in self.objects:
            with rasterio.open(obj.path_to_stack) as dst:
                stack_meta = dst.profile
            if stack_meta["crs"] != self.crs:
                log = "ID: {}: Reproject {} to {}".format(obj.id, stack_meta["crs"], self.crs)
                stack_meta.update(crs=self.crs)
                path_to_stack_reproj = obj.path_to_stack[:-4] + "_reproject.tif"
                Mosaic.warp(obj.path_to_stack, self.crs, path_to_stack_reproj)
                obj.path_to_stack = path_to_stack_reproj
            else:
                log = "ID: {}: No re-projecting of stacks necessary".format(obj.id)
            self.write_to_metafile(log)

    def create_stacks(self, output_path_base: str):
        """
        method creates stacks for all scenes within scene collection object
        """
        bands, resolutions = [4, 3, 2, 8], [10, 10, 10, 10]
        for obj in self.objects:
            date = obj.record_date.split("T")[0]
            output_path = os.path.join(output_path_base, "stack_{}_{}.tif".format(obj.id, date))
            obj.create_image_stack(bands, resolutions, output_path, radiometric_correction=True)

    def create_mosaic(self, merge_method: str = "first"):
        """
        method creates mosaic based on ranking in the scene collection object
        """
        source_files = [rasterio.open(obj.path_to_stack) for obj in self.objects if obj.path_to_stack is not None]
        mosaic, out_trans = merge(source_files, bounds=self.geometry.bounds, method=merge_method)
        mosaic_meta = {'driver': 'GTiff',
                       'dtype': 'float32',
                       'nodata': 0,
                       'width': mosaic.shape[2],
                       'height': mosaic.shape[1],
                       'count': 4,
                       'crs': self.crs,
                       'transform': out_trans,
                       'tiled': False,
                       'interleave': 'band'}
        Mosaic.save_raster(mosaic, mosaic_meta, self.path_to_mosaic)


if __name__ == "__main__":
    pass

