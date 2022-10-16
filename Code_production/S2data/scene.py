from pathlib import Path
from xml.dom import minidom
import glob
import os

import numpy as np
import rasterio

np.seterr(divide='ignore', invalid='ignore')


class SceneCollection:

    def __init__(self, meta: dict):

        def load_objects():
            """
            :return: returns list  of S2Scene objects
            """
            objects = []
            for product_id in meta.keys():
                s2_object = S2Scene(meta[product_id])
                if os.path.exists(s2_object.product_identifier):
                    objects.append(s2_object)
            return objects

        self.meta = meta
        self.objects = load_objects()

    def extract_scenes_by_product_type(self, product_type: str):
        """
        :param product_type: e.g. L2A
        :return: SceneCollection object
        """
        meta_new = {}
        obj_ls_2a = [obj for obj in self.objects if obj.product_type == product_type]
        for obj in obj_ls_2a:
            meta_new[obj.id] = obj.meta
        scene_collection_new = SceneCollection(meta_new)
        return scene_collection_new

    def calculate_mean_ndvi_for_scenes(self):
        """
        :return: calculate ndvi layer for each object in collection
        """
        for obj in self.objects:
            obj.calculate_mean_ndvi()

    def sort_scenes_by_ndvi(self):
        """
        sort objects in collection descendingly depending on ndvi mean
        :return: None
        """
        self.objects = sorted(self.objects, key=lambda obj: obj.ndvi_mean, reverse=True)


class S2Scene:

    def __init__(self, meta: dict):

        self.id = meta["id"]
        # product identifier will be used to access scenes stored in CreoDias repository
        # If you want to run this code locally adjust the line below
        # so that it points to the location where the EO data is stored
        # self.product_identifier = os.path.join("D:", meta["properties"]['productIdentifier'][1:])
        self.product_identifier = meta["properties"]['productIdentifier'][1:]
        self.bounds = meta["geometry"]["coordinates"][0]
        self.record_date = meta["properties"]["startDate"]
        self.product_type = meta["properties"]["productType"]
        self.cloud_cover = meta["properties"]["cloudCover"]
        self.processing_baseline = meta["properties"]["processingBaseline"]
        self.meta = meta
        self.path_to_stack = None
        self.stack_baseline4_corrected = False
        self.path_to_ndvi = None
        self.ndvi_mean = None

    @staticmethod
    def load_raster(image_path: str):
        """
        :param image_path:
        :return: None
        """
        with rasterio.open(image_path) as dst:
            raster_data = dst.read()
            profile = dst.profile
        return raster_data, profile

    @staticmethod
    def save_raster(image: np.array, meta: dict, output_path: str):
        """
        :param image: raster image which shall be saved
        :param meta: all necessary information for saving image
        :param output_path: path where raster shall be saved
        :return: None
        """
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(image)

    def get_crs(self):
        """
        :return: method returns crs in form 'EPSG:<epsg>'
        """
        path_to_raster = self.get_path_to_single_band_layer(2, 10)
        layer, meta = S2Scene.load_raster(path_to_raster)
        return meta["crs"]

    def get_path_to_single_band_layer(self, band: str, resolution: int):
        """
        :param band: Name of band (e.g. 'B02')
        :param resolution: Resolution of band. For some bands multiple resolutions are possible
        :return: Absolute path to image with band information
        """
        if self.product_type == "L1C":
            ends_with_str = "*_B0{}.jp2".format(band, resolution)
        elif self.product_type == "L2A":
            ends_with_str = "*_B0{}_{}m.jp2".format(band, resolution)
        paths = []
        for band_path in Path(self.product_identifier).rglob(ends_with_str):
            paths.append(str(band_path.absolute()))
        return paths[0]

    def get_path_to_multiple_band_layers(self, bands: list, resolutions: list):
        """
        :param bands: list of bands whose paths are requested
        :param resolutions: list of corresponding resolution to each band
        :return: list of absolute paths to bands
        """
        paths_to_bands = []
        for band, resolution in zip(bands, resolutions):
            path_to_band = self.get_path_to_single_band_layer(band, resolution)
            paths_to_bands.append(path_to_band)
        return paths_to_bands

    def create_ndvi_layer(self, output_path: str = None):
        """
        Method calculates ndvi layer of scene. If output path is given layer is saved to this path.
        Otherwise raster and meta data are returned
        :param output_path: ndvi layer shall be saved to this path
        :return None
        """
        red_band_path = self.get_path_to_single_band_layer(4, 10)
        nir_band_path = self.get_path_to_single_band_layer(8, 10)
        red_band, red_band_meta = S2Scene.load_raster(red_band_path)
        nir_band, nir_band_meta = S2Scene.load_raster(nir_band_path)

        if red_band_meta["nodata"] == 0:
            red_band = np.where(red_band == 0, np.nan, red_band)
        if nir_band_meta["nodata"] == 0:
            nir_band = np.where(nir_band == 0, np.nan, nir_band)
        red_band_refl = red_band / 10000
        nir_band_refl = nir_band / 10000
        ndvi = (nir_band_refl - red_band_refl) / (nir_band_refl + red_band_refl)

        if output_path:
            red_band_meta.update(driver="gtiff")
            red_band_meta.update(count=ndvi.shape[0])
            red_band_meta.update(dtype="float64")
            red_band_meta.update(nodata=np.nan)
            S2Scene.save_raster(ndvi, red_band_meta, output_path)
            self.path_to_ndvi = output_path
        else:
            return ndvi, red_band_meta

    def calculate_mean_ndvi(self):
        """
        :return: nothing is returned but object attribute ndvi_mean is updated
        """
        if self.path_to_ndvi:
            ndvi_layer, ndvi_meta = S2Scene.load_raster(self.path_to_ndvi)
        else:
            ndvi_layer, ndvi_meta = self.create_ndvi_layer()
        ndvi_na_masked = np.ma.array(ndvi_layer, mask=np.isnan(ndvi_layer))
        self.ndvi_mean = ndvi_na_masked.mean()

    def apply_radiometric_correction(self, image: np.array, band: int):
        """
        to recreate radiometric properties of S2-scenes before baseline4 correction, radiometric values have to be shifted.
        There metadata file is accessed to get shift values for each band.
        """
        if self.processing_baseline == 4:
            path_to_mdt_file = glob.glob(os.path.join(self.product_identifier, "MTD_*"))[0]
            file = minidom.parse(path_to_mdt_file)
            for element_name in ["BOA_ADD_OFFSET", "RADIO_ADD_OFFSET"]:
                list_of_elements = file.getElementsByTagName(element_name)
                if len(list_of_elements) != 0:
                    for element in list_of_elements:
                        band_file = int(element.attributes['band_id'].value)
                        shift_value = int(element.firstChild.data)
                        if band_file == band:
                            image_updated = np.where(image == 0, image, image + shift_value)
            self.stack_baseline4_corrected = True
            return image_updated
        else:
            return image

    def create_image_stack(self, bands: list, resolutions: list, output_path: str,
                           radiometric_correction: bool = False):
        """
        :param bands: list of bands which shall be stacked
        :param resolutions: list of corresponding band resolutions
        :param output_path: stacked image shall be saved to this path
        :param radiometric_correction: should image stack be radiometrically corrected if processing baseline == 4
        :return None
        """

        path_to_bands = self.get_path_to_multiple_band_layers(bands, resolutions)
        images_profiles_tuples = [(S2Scene.load_raster(path)) for path in path_to_bands]
        images = [tup[0] for tup in images_profiles_tuples]
        meta = images_profiles_tuples[0][1]
        meta.update(driver="gtiff")
        meta.update(dtype="uint16")
        meta.update(count=4)
        if radiometric_correction:
            images_corrected = []
            for image, band in zip(images, bands):
                image_corrected = self.apply_radiometric_correction(image, band)
                images_corrected.append(image_corrected)
                images = images_corrected
        images = [image.reshape(image.shape[1], image.shape[2]) for image in images]
        image_stack = np.stack(images)
        self.path_to_stack = output_path
        S2Scene.save_raster(image_stack, meta, output_path)


if __name__ == "__main__":
    pass


