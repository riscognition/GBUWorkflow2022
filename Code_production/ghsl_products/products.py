import rasterio
from rasterio import features
import numpy as np
from skimage.measure import regionprops_table
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import watershed
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from skimage import filters
from skimage.morphology import square
import cv2


class Bufrac:

    def __init__(self, path_to_bufrac_raw: str,
                 path_to_mosaic: str,
                 path_to_bufrac_output: str,
                 path_to_greenness_output: str):
        self.path_to_bufrac_raw = path_to_bufrac_raw
        self.path_to_mosaic = path_to_mosaic
        self.path_to_bufrac_output = path_to_bufrac_output
        self.path_to_greenness_output = path_to_greenness_output
        self.bufrac_raw = None
        self.bufrac_meta = None
        self.mosaic = None
        self.mosaic_meta = None
        self.bufrac_processed = None
        self.ndvi = None
        self.greenness = None

    @staticmethod
    def load_raster(path_to_raster: str):
        """
        method loads raster as numpy array and additionally return metadata
        """
        with rasterio.open(path_to_raster) as src:
            raster = src.read()
            meta = src.profile
        return raster, meta

    @staticmethod
    def save_raster(image: np.array, path_to_output: str, meta: dict):
        """
        method saves image to specified location with profile specified in meta file
        """
        with rasterio.open(path_to_output, 'w', **meta) as dst:
            dst.write(image)

    def load_data(self):
        """
        method loads data (paths were given at object creation) into memory
        """
        bufrac_raw, bufrac_meta_raw = Bufrac.load_raster(self.path_to_bufrac_raw)
        mosaic, mosaic_meta = Bufrac.load_raster(self.path_to_mosaic)
        self.bufrac_raw = bufrac_raw
        self.bufrac_meta = bufrac_meta_raw
        self.mosaic = mosaic
        self.mosaic_meta = mosaic_meta
        self.bufrac_processed = bufrac_raw

    def mask_by_bufrac_threshold(self, th: float):
        """
        method sets bufrac values below the specified threshold to 0
        """
        self.bufrac_processed = np.where(self.bufrac_processed > th, self.bufrac_processed, 0)

    def create_ndvi_layer(self):
        """
        method creates ndvi layer with given mosaic and stores it as numpy array in 'self.ndvi'
        """
        red_band = self.mosaic[0]
        nir_band = self.mosaic[3]
        if self.mosaic_meta["nodata"] == 0:
            red_band = np.where(red_band == 0, np.nan, red_band)
            nir_band = np.where(nir_band == 0, np.nan, nir_band)
        red_band_refl = red_band / 10000
        nir_band_refl = nir_band / 10000
        ndvi = (nir_band_refl - red_band_refl) / (nir_band_refl + red_band_refl)
        self.ndvi = ndvi

    def mask_by_ndvi_threshold(self, th: float):
        """
        method sets bufrac values to 0, where corresponding pixel have ndvi below specified threshold
        """
        self.bufrac_processed = np.where(self.ndvi > th, 0, self.bufrac_processed)

    def mask_by_streets(self, path_to_osm_streets: str, path_to_osm_street_width: str):
        """
        method sets bufrac values of pixels whose centers are covered by street buffer to 0
        """
        streets = gpd.read_file(path_to_osm_streets)
        street_widths = pd.read_csv(path_to_osm_street_width, delimiter=";")

        if streets.crs == self.bufrac_meta["crs"]:
            streets_merged = streets.merge(street_widths, on="fclass")
            streets_buffered = streets_merged.buffer(streets_merged["width_m"])
            geom = [shapes for shapes in streets_buffered.geometry]
            rasterized = features.rasterize(geom,
                                            out_shape=self.bufrac_processed[0].shape,
                                            fill=0,
                                            out=None,
                                            transform=self.bufrac_meta["transform"],
                                            all_touched=False,
                                            default_value=1,
                                            dtype=None)
            bufrac_processed = np.where(rasterized == 1, 0, self.bufrac_processed)
            self.bufrac_processed = bufrac_processed.reshape(self.bufrac_processed.shape)
        else:
            raise ValueError("{} from streets file and {} from bufrac are not the same"
                             .format(streets.crs, self.bufrac_meta["crs"]))

    def soften_street_masking(self, path_to_osm_streets: str, path_to_osm_street_width: str):
        """
        method multiplies bufrac of pixels which are intersected by street buffer with 0.5.
        Has a softening effect on bufrac image when applied after the mask_by_streets method
        """
        streets = gpd.read_file(path_to_osm_streets)
        street_widths = pd.read_csv(path_to_osm_street_width, delimiter=";")
        streets_merged = streets.merge(street_widths, on="fclass")
        streets_buffered = streets_merged.buffer(streets_merged["width_m"])
        geom = [shapes for shapes in streets_buffered.geometry]
        rasterized = features.rasterize(geom,
                                        out_shape=self.bufrac_processed[0].shape,
                                        fill=0,
                                        out=None,
                                        transform=self.bufrac_meta["transform"],
                                        all_touched=True,
                                        default_value=1,
                                        dtype=None)
        bufrac_processed = np.where(rasterized == 1, self.bufrac_processed * 0.5, self.bufrac_processed)
        self.bufrac_processed = bufrac_processed.reshape(self.bufrac_processed.shape)

    def create_greenness_layer(self, kernel_size: int = 60):
        """
        method creates the greenness layer, which stores ndvi values for pixels where bufrac value > 0.
        and additionally for those pixels that are inside a morphological closing of bufrac > 0, with a
        circular structuring elements of {kernel size / 2} m radius
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        bufrac_close = cv2.morphologyEx(self.bufrac_processed[0], cv2.MORPH_CLOSE, kernel)
        greenness = np.where(bufrac_close > 0, self.ndvi, 255.0)
        self.greenness = greenness.reshape(1, self.bufrac_meta["width"], self.bufrac_meta["height"])

    def save_layers(self):
        """
        method saves all required outputs to paths defined at object instantiation.
        """
        Bufrac.save_raster(self.bufrac_processed, self.path_to_bufrac_output, self.bufrac_meta)
        greenness_meta = self.mosaic_meta
        greenness_meta.update(nodata=255)
        greenness_meta.update(count=1)
        Bufrac.save_raster(self.greenness, self.path_to_greenness_output, self.mosaic_meta)


class NRes:
    """
    This class can be used to create the Residential (1) Non-Residential (2) layer
    from BUFRAC. This class contains an object-based approach and a pixel based approach to do this.
    """

    def __init__(self,
                 path_to_bufrac_pp: str,
                 path_to_mosaic: str,
                 path_to_n_res: str):
        self.path_to_bufrac_pp = path_to_bufrac_pp
        self.path_to_mosaic = path_to_mosaic
        self.path_to_nres = path_to_n_res
        self.bufrac_pp = None
        self.bufrac_meta = None
        self.mosaic = None
        self.mosaic_meta = None
        self.region_raster = None
        self.features = None
        self.n_res_layer = None

    @staticmethod
    def load_raster(path_to_raster: str):
        with rasterio.open(path_to_raster) as src:
            raster = src.read()
            meta = src.profile
        return raster, meta

    @staticmethod
    def save_raster(image: np.array, path_to_output: str, meta: dict):
        with rasterio.open(path_to_output, 'w', **meta) as dst:
            dst.write(image)

    def load_data(self):
        self.bufrac_pp, self.bufrac_meta = Bufrac.load_raster(self.path_to_bufrac_pp)
        self.bufrac_pp = self.bufrac_pp[0]
        self.mosaic, self.mosaic_meta = Bufrac.load_raster(self.path_to_mosaic)

    def segment_bufrac(self, min_distance: int, th_abs: float):
        """
        Bufrac raster is segmented into regions based on values utilizing peak_local_max and watershed analysis.
        For each region a set of morphological features is calculated for each region.
        Pixels with Bufrac = 0 are neglected.
        """
        bufrac = np.where(self.bufrac_pp == 255, np.nan, self.bufrac_pp)
        a = (bufrac > 0).astype(int)
        D = bufrac
        localMax = peak_local_max(D, indices=False,
                                  min_distance=min_distance,
                                  labels=a,
                                  exclude_border=0,
                                  threshold_abs=th_abs)
        markers = measure.label(localMax)
        labels = watershed(-D, markers, mask=a)

        table = regionprops_table(
            labels,
            properties=('label',
                        'area',
                        'eccentricity',
                        "equivalent_diameter_area",
                        "feret_diameter_max",
                        "extent",
                        "perimeter",
                        "solidity"),
        )

        self.features = pd.DataFrame(table)
        self.region_raster = labels

    def calculate_features(self):
        """
        Additional radiometric features for each region are calculated with this method.
        The feature data set is stored in self.features.
        """
        label_dic = {}
        for label in self.features.label:
            label_dic[label] = self.region_raster == label

        red_band = self.mosaic[0]
        green_band = self.mosaic[1]
        blue_band = self.mosaic[2]

        red_band = np.where(red_band == 0, np.nan, red_band)
        green_band = np.where(green_band == 0, np.nan, green_band)
        blue_band = np.where(blue_band == 0, np.nan, blue_band)

        self.features["bufrac_max"] = self.features["label"].apply(lambda x: self.bufrac_pp[label_dic[x]].max())
        self.features["bufrac_mean"] = self.features["label"].apply(lambda x: self.bufrac_pp[label_dic[x]].mean())
        self.features["bufrac_std"] = self.features["label"].apply(lambda x: self.bufrac_pp[label_dic[x]].min())
        self.features["bufrac_min"] = self.features["label"].apply(lambda x: self.bufrac_pp[label_dic[x]].std())
        self.features["red_mean"] = self.features["label"].apply(lambda x: red_band[label_dic[x]].mean())
        self.features["green_mean"] = self.features["label"].apply(lambda x: green_band[label_dic[x]].mean())
        self.features["blue_mean"] = self.features["label"].apply(lambda x: blue_band[label_dic[x]].mean())

    def predict_nres(self, object_based_model: str):
        """
        By using a trained model and the extracted features
        each region is classified into residential (1) or non-residential (2)
        """
        X = self.features.iloc[:, 1:].values
        y = object_based_model.predict(X)
        self.features["residential"] = y

    def classify_forgotten_pixels_res(self):
        """
        Assign residential class to pixels which have Bufrac > 0 but were not assigned to a region.
        """
        nres = self.n_res_layer == 2
        self.n_res_layer = np.where(self.bufrac_pp > 0, 1, self.n_res_layer)
        self.n_res_layer = np.where(nres, 2, self.n_res_layer)

    def classify_forgotten_pixels_nn(self):
        """
        Assign class of the nearest classified pixel to pixels which have Bufrac > 0 but were not assigned to a region.
        """
        forgotten_pixels = np.where((self.bufrac_pp > 0) & (self.n_res_layer == 0))
        classified_pixels = np.where(self.n_res_layer > 0)
        forgotten_pixels_coords = np.array([[x, y] for x, y in zip(forgotten_pixels[0], forgotten_pixels[1])])
        classified_pixels_coords = np.array([[x, y] for x, y in zip(classified_pixels[0], classified_pixels[1])])
        indices_nn_forgotten_points = cKDTree(classified_pixels_coords).query(forgotten_pixels_coords, k=1)[1]
        indies_nres_nn = classified_pixels_coords[indices_nn_forgotten_points]
        values_nres_nn = self.n_res_layer[indies_nres_nn[:, 0], indies_nres_nn[:, 1]]
        self.n_res_layer[forgotten_pixels[0], forgotten_pixels[1]] = values_nres_nn

    def create_nres_layer(self):
        """
        creates res nres raster from regions
        """
        n_res_layer = self.region_raster.copy()
        for index, row in self.features.iterrows():
            label = row["label"]
            residential = row["residential"]
            n_res_layer = np.where(n_res_layer == label, residential, n_res_layer)
        self.n_res_layer = n_res_layer

    def pixel_based_classification(self, pixel_based_model: str):
        """
        Each pixel is assigned the class residential (1) or non-residential (2) based on radiometric features only.
        """

        mask = np.where((self.bufrac_pp.ravel() > 0))
        df = pd.DataFrame()

        df['bufrac'] = pd.Series(self.bufrac_pp.ravel()[mask])
        df['red'] = pd.Series(self.mosaic[0].ravel()[mask])
        df['blue'] = pd.Series(self.mosaic[2].ravel()[mask])
        df['green'] = pd.Series(self.mosaic[1].ravel()[mask])
        df['nir'] = pd.Series(self.mosaic[3].ravel()[mask])

        X = df.values
        y = pixel_based_model.predict(X)
        nres = self.bufrac_pp.copy()
        nres.ravel()[mask] = y
        self.n_res_layer = nres.reshape(self.bufrac_meta["width"], self.bufrac_meta["height"])
        self.n_res_layer = self.n_res_layer.astype("uint8")

    def majority_filter(self, kernel_size: int):
        """
        Salt and pepper effect of pixel based classification is reduced
        by applying a majority filter to classification layer.
        """
        D = (self.n_res_layer > 0)
        no_bufrac_mask = (self.n_res_layer == 0)
        res_maj_filter = filters.rank.majority(self.n_res_layer, square(kernel_size), mask=D)
        res_maj_filter[no_bufrac_mask] = 0
        self.n_res_layer = res_maj_filter

    def save_layer(self):
        nres_meta = self.bufrac_meta.copy()
        nres_meta.update(dtype="uint8")
        self.n_res_layer = self.n_res_layer.reshape(1, self.bufrac_meta["width"], self.bufrac_meta["height"])
        NRes.save_raster(self.n_res_layer, self.path_to_nres, nres_meta)


if __name__ == "__main__":
    pass
