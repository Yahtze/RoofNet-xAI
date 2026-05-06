# RoofNet (v1.0)

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: https://creativecommons.org/licenses/by/4.0/

## Overview 
RoofNet is the largest and most geographically diverse dataset for global roof material classification. It consists of high-resolution Earth Observation (EO) image records with structured metadata and curated textual prompts describing roofing characteristics across 14 material classes and an "Unknown" class. The dataset is designed to support hazard preparedness, resilience planning, and post-disaster supply-chain analysis.

RoofNet includes:

- 49,662 building records spanning across 101 countries, with rich per-sample metadata (e.g., building height, roof shape, solar panel presence). (See our [Kaggle page](https://www.kaggle.com/datasets/doubleblindreview/xbd-roof-images) for imagery and the "resources" folder for the metadata CSV. Please note that the xBD dataset is distributed under [CC-BY-NC-SA 4.0](https://xview2.org/terms), and is thus shared under derivative license terms.)
    - We also share the necessary code to reproduce the entire EO imagery dataset in the "download_preprocess" folder, including the code to fetch Google Satellite imagery ("download_from_csv.py"), parse xBD imagery ("parse_xview2_dataset.py"), and crop all fetched imagery ("roof_view.py").

- 14 roof material classes (see below).

- RemoteCLIP ViT-L/14 model fine-tuned for roof classification. (See our [Kaggle page](https://www.kaggle.com/datasets/doubleblindreview/xbd-roof-images).)

- Training and evaluation code for reproducible fine-tuning and VLM experimentation. (See the "training_evaluation" folder.)

- Example benchmarking code for an earthquake disaster simulation. (See the "example_application" folder.)

### Roof Classes
RoofNet includes 14 roofing material classes grouped into 4 categories:

1. Natural/Traditional: Thatch, Green Vegetative, Stone Slates
2. Manufactured Tiles: Asphalt Tiles, Concrete Tiles, Wood Tiles, Clay Tiles
3. Sheet Materials: Metal Sheet Materials, Polycarbonate Sheet Materials, Glass Sheet Materials
4. Synthetic/Amorphous: Amorphous Asphalt, Amorphous Concrete, Amorphous Membrane, Amorphous Fabric

### Applications
We envision a broad possible range of applications for RoofNet, including climate resilience planning, supply chain analysis, and insurance modeling. We present one example application in the "example_application" folder to illustrate the utility of roofing material-aware earthquake exposure modeling. While we regrettably cannot open-source all the data necessary to execute the example code (we apply census data from [INEGI](https://www.inegi.org.mx/default.html) via [IPUMS](https://www.ipums.org)), the interested reader is invited to examine our results further when such information becomes public.

### Script Execution
For the interested reader who may wish to execute the image collection and analysis scripts for themselves, we provide the "roofnet_execution.sh" script which will run a selection of the scripts as a batch. Please note that it is expected for users to provide/generate their own list of building coordinates to pull roof imagery from (see the "download_preprocess" folder for an example "sample_osm_polygons_gsat_imagery.ipynb" notebook to fulfill this requirement), as well as their own Google Static Maps API key (for use in "resources/keys.json"), if they wish to sample roof imagery from Google Maps. It will also be necessary to download the "best_clip_model_balanced.pth" file from the Kaggle page and place it in the "resources" folder in order to run the classification portion of this shell script successfully. Note that the script also expects the path to the virtual environment in which requirements.txt was downloaded to be provided as the second input argument.
