# Inverse Design of Catalytic Active Sites via Interpretable Topology-Based Deep Generative Models
This repo contains demonstrations of an a persistent path homology-based semi-supervised prediction and generation framework, empowered by our PathVAEs. This framework aims to predict the adsorption energy and design potential for High-entropy alloy catalysts.

This repository is adapted from the codebase used to produce the results in the paper "Path topology-assisted Semi-supervised Framework for High-Entropy Alloy Catalysts Prediction and Generation."

## Requirements

The code in this repo has been tested with the following software versions:
- Python>=3.7.0
- torch>=1.13.1
- numpy>=1.21.5
- scikit-learn>=0.24.2
- matplotlib>=3.3.4

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Data

The data for path complexes constructed for different HEA catalysts calculated by DFT is located in the directory
```
./get_data/rawfeature/
```
The data for path complexes constructed for simulation-generated different HEA catalysts is located in the directory
```
./get_data/rawfeature_simu/
```
The data for ligand features and coordination features of different HEA catalysts calculated by DFT, is available in the directory
```
./get_data/teacher_data/
```
The data for ligand features and coordination features of simulation-generated different HEA catalysts, is available in the directory
```
./get_data/student_data/
```
The data for ligand features and coordination features of simulation-generated different HEA catalysts for the VAE training, is available in the directory
```
./PathVAEs/fornn/
```
To obtain full data, please contact 2101212695@stu.pku.edu.cn 


## Files

This repo should contain the following files:
- 1 ./get_data/get_feature.py - The code employed in the retrieval of data pertains to the path complexes derived from DFT calculations for various HEA catalysts.
- 2 ./get_data/get_more_simulation.py - The code utilized for acquiring data corresponds to the path complexes generated through simulations for distinct HEA catalysts.
- 3 ./get_data/get_topo_feature.py - The code implemented is designed for the extraction of persistent path homology concerning diverse HEA catalysts. This encompasses catalysts for which calculations were performed using DFT, as well as those generated through simulations.
- 4 ./get_data/Digraph.py - The method code for computing persistent path homology for path complexes.
- 5 ./GBRT/getdata.py - The code employed for the data for the inputs to the GBRT model.
- 6 ./GBRT/GBRT.py - The main code for GBRT model.
- 7 ./GBRT/model.py - The code for GBRT model.
- 8 ./GBRT/model_test.py - The code employed for acquiring the labels of simulation-generated different HEA catalysts by trained GBRT.
- 9 ./PathVAEs/getdata.py - Method code for inputting data to PathVAEs.
- 10 ./PathVAEs/the_nn.py - The code for PathVAEs with semi-supervised learning.
- 11 ./latent_space/getdata.py - Method code for inputting data to PathVAEs.
- 12 ./latent_space/get_100_result.py - The code employed to get the latent spaces of PathVAEs and the high potential HEA catalyst structures.

## Other 

If you find any bugs or have questions, please contact 2101212695@stu.pku.edu.cn # PathVAEs
# PathVAEs
