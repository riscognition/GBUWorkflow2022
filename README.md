# GBU Workflow 2022
This code is indended to produce the GHSL products from Sentinel-2 data using the JRC's GHS-S2Net.

Please download all CNN models you intend to you with this code from https://github.com/ec-jrc/GHS-S2Net and store them in a folder named CNN_models. 
The folder must be in the same location as the folder Code_production.

The folder code_production stores the code which was used to produce:
- Sentinel2 Input data of a specified AOI as input for the CNN model.
- Built-up area fraction (BUFRAC) (as predicted by GHS-S2Net)
- BUFRAC with further processing/cleaning steps
- The classification of the BUFRAC into residential and non-residential (RES NRES layer)
- A Greenness layer which states the NDVI for the Bufrac domain (including morphological closings)
- Postprocessed products (Reprojection and Aggregation of BUFRAC and RES NRES layer)

The code is intended to work in cooperation with the CreoDias infrastructure. 
The following steps have to be done so that the code can be used.

1. Download repository https://github.com/DHI-GRAS/creodias-finder.git
2. The folder creodias_finder must be moved into the code_production folder
3. The code must be hosted within a Virtual Machine (VM) of CreoDias to have access to their EO Data repository. For this purpose a Dockerfile is provided, which can be used to create an Image, that can be built and run within the Creodias VM.
4. Alternatively, if you have access to a EO repository on your local drive you can run the Code locally with minimal adjustments. (See line 61 in scene.py for instructions)

Furthermore to run the code the python packages as specified in the requirements.txt must be satisfied. (3.10.5)
However, the file run_GHS_S2Net.py has heavier requirements specified at https://github.com/ec-jrc/GHS-S2Net. 
It is recommended to create 2 seperate Virtual Environments. One for running GHS-S2Net and one for everything else. 

For the Code within the file run_GHS_S2Net.py credits are given to the developers of the repo https://github.com/ec-jrc/GHS-S2Net whose code have been used and adjusted. 
Credits to
Christina Corbane
Vasileios Syrris 
Filip Sabo 

For the execution of the code the files 
- create_ghls_products.py
- create_S2_input.py
- run_GHS_S2Net.py
The correct input and output paths have to be specified at the appropriate locations. 

The folder RES_NRES_model_training contains some jupyter notbooks which can be used to recreate the RES NRES pixel based classification model.
