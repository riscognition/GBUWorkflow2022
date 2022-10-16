import numpy as np
import copy
import time
import gc
from sklearn.feature_extraction import image
import os
from osgeo import gdal, osr
import tensorflow as tf
from keras.models import load_model


def read_image(infile):
    src = gdal.Open(infile)
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()
    proj = osr.SpatialReference(wkt=src.GetProjection())
    EPSG = proj.GetAttrValue('AUTHORITY', 1)
    datatype = src.GetRasterBand(1).DataType
    datatype = gdal.GetDataTypeName(datatype)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)

    cols = src.RasterXSize
    rows = src.RasterYSize

    if src.RasterCount == 1:
        Im = src.GetRasterBand(1).ReadAsArray()
    else:
        Im = np.zeros((src.RasterCount, rows, cols), dtype=np.uint16)
        for q in range(src.RasterCount):
            # print(q, end=" ")
            Im[q, :, :] = src.GetRasterBand(q + 1).ReadAsArray()

    return Im, src, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection, src.RasterXSize, src.RasterYSize


def make_prediction(path_to_cnn_model: str, path_to_data: str, batch_size=10000, window=5):
    """compute Built-Up predictions based on S2 data for area covered by input image"""

    # load model
    model = load_model(path_to_cnn_model)

    # Read the 4-band Sentinel-2 image
    S2, srcS2, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection, rows, cols = read_image(path_to_data)

    Datamask = np.sum(S2.reshape(S2.shape[0], S2.shape[1] * S2.shape[2]) > 0, axis=0) == 4
    I2 = np.rollaxis(S2, 0, 3)

    I2 = np.float32(I2)
    del S2

    I2[I2 > 10000] = 10000
    I2 = I2 / 10000.

    # Prepare (expand) tiles for the model prediction
    # sliding window 5x5
    for q in range(I2.shape[2]):
        print(q, end=" ")
        tmp = copy.copy(I2[:, :, q])
        tmp = np.pad(tmp, (int(window / 2), int(window / 2)), 'reflect')
        tmp = image.extract_patches_2d(tmp, (window, window))
        if q == 0:
            T2 = np.expand_dims(tmp, axis=3)
        else:
            T2 = np.concatenate((T2, np.expand_dims(tmp, axis=3)), axis=3)
    del tmp

    gc.collect()

    # Run the prediction
    tt = time.time()
    with tf.device('/cpu:0'):
        Response = model.predict(T2, batch_size=batch_size * 8, verbose=1)
    prediction_time = str(time.time() - tt)

    del T2
    gc.collect()

    Response = Response * 10000
    Response = Response.astype(np.uint16)
    Response[~Datamask, :] = 65535

    Out = Response[:, 1].reshape(rows, cols)

    del Datamask, Response

    gc.collect()

    return Out, geotransform, projection, rows, cols


def pred_to_bufrac(pred_raster: np.array):
    """Transform image to product requirements; data range 0-1, no-data 255"""
    # rescale to data range 0 -1 (except no data values)
    pred_raster = np.where((pred_raster >= 0) & (pred_raster <= 10000), pred_raster / 10000, pred_raster)
    # change no data value from 65535 to 255
    bufrac = np.where(pred_raster == 65535, 255, pred_raster)
    return bufrac


def save_image_single_band(Out, path_out, cols, geotransform, projection, rows, no_data_value: int, datatype: object):
    # datatype (e.g. gdal.GDT_Float64)
    # save the results
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(path_out, cols, rows, 1, datatype, ['COMPRESS=DEFLATE', 'TILED=YES'])
    dst.SetGeoTransform(geotransform)
    dst.SetProjection(projection)
    dst.GetRasterBand(1).SetNoDataValue(no_data_value)
    dst.GetRasterBand(1).WriteArray(Out)
    dst.FlushCache()
    dst = None


def create_bufrac_image(path_to_input_image: str, tile_name: str, path_to_output_image: str):
    """accesses processed S2 image through aoi_name and image_record_date, predicts BUFRAC and stores image to folder"""

    # create path to model
    path_to_model = os.path.realpath(os.path.join(os.path.dirname(__file__), '..',
                                                  'CNN_models', 'MODEL_CNN_{}.h5'.format(tile_name)))

    # load image, make prediction
    prediction_raster, geotransform, projection, rows, cols = make_prediction(path_to_model, path_to_input_image)
    # transform predictions to required standards
    bufrac_raster = pred_to_bufrac(prediction_raster)
    # save image to output path
    save_image_single_band(bufrac_raster, path_to_output_image, cols, geotransform, projection, rows,
                           no_data_value=255, datatype=gdal.GDT_Float64)


if __name__ == "__main__":

    tile_name = "<insert name of model here>"  # e.g. 16T for AOI1
    path_to_src_image = "<insert path to S2 input image>"
    path_to_output_image = "<insert path and name of output file here>"
    create_bufrac_image(path_to_src_image, tile_name, path_to_output_image)













