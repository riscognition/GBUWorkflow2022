from ghsl_products.products import Bufrac, NRes
from ghsl_products.post_processed_products import create_post_processed_product
import joblib

if __name__ == "__main__":

    path_to_res_nres_classification_model = "<insert path to res nres classification model in pkl.format> "
                                            # Depends also on type of classification you want to undertake
                                            # Pixel based vs object based
    #pixel_based_model = joblib.load(path_to_res_nres_classification_model)

    path_to_bufrac_initial = "<insert path initial bufrac raster>"
    path_to_mosaic = "<insert path to Sentinel2 mosaic which was used to ceate initial bufrac>"
    path_to_osm_streets = "<insert path to osm streets shp.file covering the AOI>"
    path_to_osm_street_widths = "<insert path to csv file storing fclass and street width "
                                # of streets that shall be included in street masking "
    path_to_bufrac_processed = "<insert path where final bufrac layer shall be stored>"
    path_to_greenness = "<insert path where greenness layer shall be stored>"
    path_to_nres = "<insert path where residential non residential layer shall be stored>"

    path_to_butot_100m = "<insert path for butot in Mollweide projection 100m>"
    path_to_butot_3sec = "<insert path for butot in WGS84 4326 projection 3 arc seconds>"

    # Bufrac and Greenness
    obj = Bufrac(path_to_bufrac_initial, path_to_mosaic, path_to_bufrac_processed, path_to_greenness)
    obj.load_data()
    obj.mask_by_bufrac_threshold(th=0.1)
    obj.mask_by_streets(path_to_osm_streets, path_to_osm_street_widths)
    obj.soften_street_masking(path_to_osm_streets, path_to_osm_street_widths)
    obj.create_ndvi_layer()
    obj.mask_by_ndvi_threshold(th=0.3)
    obj.create_greenness_layer()
    obj.save_layers()

    # N_Res with pixel based classification
    nres_obj = NRes(path_to_bufrac_processed,
                    path_to_mosaic,
                    path_to_nres)
    nres_obj.load_data()
    pixel_based_model = joblib.load(path_to_res_nres_classification_model)
    nres_obj.pixel_based_classification(pixel_based_model)
    nres_obj.majority_filter(kernel_size=3)
    nres_obj.save_layer()

    create_post_processed_product(path_to_bufrac_processed, path_to_butot_100m, "ESRI:54009", 10)
    create_post_processed_product(path_to_bufrac_processed, path_to_butot_3sec, "EPSG:4326", 0.000833333)

