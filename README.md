# RoofNet (v1.0)
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

NOTE: ODbL terms apply for derivative geospatial data, please refer to our (forthcoming) publication for full license details.

## Overview 
RoofNet is the largest and most geographically diverse open-access dataset for global roof material classification. It consists of high-resolution Earth Observation (EO) image tiles paired with structured metadata and curated textual prompts describing roofing characteristics across 14 material classes and an "Unknown" class. The dataset is designed to support hazard preparedness, resilience planning, and post-disaster supply-chain analysis and will have benchmarked results shortly.

RoofNet includes:

- 49,830 EO image tiles spanning 162 urban regions across 103 countries.

- 14 roof material classes (see below).

- A multimodal CSV file with rich per-sample metadata (e.g., building height, roof shape, solar panel presence).

- RemoteCLIP ViT-L/14 models fine-tuned for roof classification.

- Training and evaluation code for reproducible fine-tuning and VLM experimentation. 

- Example application benchmarking code for an earthquake disaster simulation.

### Roof Classes
RoofNet includes 14 roofing material classes grouped into 5 categories:

1. Natural/Traditional: Thatch, Green Vegetative
2. Stone/Ceramic Tiles: Stone Slates, Clay Tiles
3. Asphalt/Concrete/Wood Tiles: Asphalt Tiles, Concrete Tiles, Wood Tiles
4. Sheet-Based: Metal Sheet Materials, Polycarbonate Sheet Materials, Glass Sheet Materials
5. Synthetic/Amorphous: Amorphous Asphalt, Amorphous Concrete, Amorphous Membrane, Amorphous Fabric

### Models
The models/ folder includes a fine-tuned version of RemoteCLIP ViT-L/14, adapted using 6,000 manually annotated samples, class re-balancing, and with prompts incorporating geographic and material cues. See the notebooks folder for reproducible training and evaluation pipelines.

### Applications
We envision a broad possible range of applications for RoofNet, including climate resilience planning, supply chain analysis, and insurance modeling. We present one example application in the example_application/ folder to illustrate the utility of roofing material-aware earthquake exposure modeling. While we regrettably cannot open-source all the data necessary to execute the example code (we apply census data from [INEGI](https://www.inegi.org.mx/default.html) via [IPUMS](https://www.ipums.org)), the interested reader is invited to examine our results further when such information becomes public.
