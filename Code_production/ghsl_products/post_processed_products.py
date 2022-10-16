from osgeo import gdal


def create_post_processed_product(path_infile: str, path_outfile: str, projection: str, resolution: str):
    """
    resamples and aggregates image
    ex1:
    projection: "EPSG:4326"
    resolution: 0.0000833333, 0.000833333, 0.00833333 (0.3, 3, 30 arc seconds)

    ex2:
    projection: "ESRI:54009"
    resolution: 10, 100, 1000

    """

    outDs = gdal.Warp(path_outfile,
                      path_infile,
                      dstSRS=projection,
                      outputType=gdal.GDT_Float64,
                      xRes=str(resolution), yRes=str(resolution),
                      resampleAlg="average"
                      )
    del outDs
