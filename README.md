ShapGCN
==============================

Implementation of [**Explaining Human Activity Recognition with SHAP: Validating Insights with Perturbation and Quantitative Measures**](https://arxiv.org/abs/2411.03714),
extended with the experiments of [**Choose Your Explanation: A Comparison of SHAP and GradCAM in Human Activity Recognition**](https://arxiv.org/pdf/2412.16003)

### Libraries

The code is based on Python >= 3.10 and PyTorch (2.3.0). Run the following command to install all the required packages 
from setup.py:
```
pip install .
```

### Dataset 

The XAI procedure and the experiments are done on the **NTU RGB+D 60 X-View** datasets, which can be downloaded 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

The CP dataset remains private due to ethical considerations.

#### Folder Structure
Follow this data structure to ensure that the files are found.
Also, check the **root_folder** in the **config_node.yaml** to guide the algorithm to your data folder. 
``` 
data
├── npy_files
│   └── processed
│       ├── cp_19   
│       ├── cp_29  # put the 29-point npy files here
│       └── ntu60  # put the ntu60 npy files here
│   └── transformed --> These files are generated with the transform.py
│       └── ntu60  # put the transformed ntu60 npy files here

```

Also, please specify your node in the **config_node.yaml** where you can guide the algorithm.
This enables you to run the code on other clusters and put the right paths into place.

#### Preprocessing

To preprocess the NTU data, the user is referred to the instructions at:
```
./src/data/preprocess/README.md
```

# Run configuration
The user is provided with the following modes:

- "xai" Mode: Executes code for explainable artificial intelligence (XAI) using the SHAP library.

Check the config file and change the following parameters in the config_node:

```
work_dir    -> path where you want to store the output of the run
dataset     -> Choose the dataset and change the root folder to guide to the preprocessed npy-files
```

The modes can either be activated or deactivated by setting the flags to True or False - refer to the different 
config run files for further information.

Ensure you have the config file ready, which can be given as ``--config /path/to/your/config``.
Furthermore, you have to provide the path for the datasets via ``--node /path/to/your/node/config``.
The standard config files can be found in the config folder, where you can specify your node paths.

# RUN
The algorithm works on different datasets: NTU60, NTU120, CP19, and CP29.
Depending on which dataset you want to use, you must specify the right config path.

## Run XAI
**IMPORTANT:** This code is based on a forked version of SHAP!

Get the right package via:
```
pip install git+https://github.com/Tempelx/shap.git
```

In the config file, the **xai_...** parameters can be changed accordingly. 

To perform a SHAP estimation, check the config file for the settings appropriate settings and provide a pre-trained 
model path.

Afterward, execute:
```
python main.py -m xai  
```

## Results

The pre-trained models used in our study are stored in the `./logs/xai` folder.
The logs for the experiments can be found [here](https://studntnu-my.sharepoint.com/:f:/g/personal/fetempel_ntnu_no/EuKmjqRHMaNOiIc1bK5EEEcBgwz8Dvq84rhMjaBO8aMQgg?e=vUjQAR)

## Citation and Contact

If you have any questions, feel free to email `felix.e.f.tempel@ntnu.no`.

Please cite our papers if you use this code in your research. :)
```
@misc{tempelExplainingHumanActivity2024,
	title = {Explaining {Human} {Activity} {Recognition} with {SHAP}: {Validating} {Insights} with {Perturbation} 
	and {Quantitative} {Measures}},
	url = {https://arxiv.org/abs/2411.03714},
	doi = {10.48550/ARXIV.2411.03714},
	author = {Tempel, Felix and Ihlen, Espen Alexander F. and Adde, Lars and Strümke, Inga},
	year = {2024},
}

@misc{tempel2024chooseexplanationcomparisonshap,
      title={Choose Your Explanation: A Comparison of SHAP and GradCAM in Human Activity Recognition}, 
      author={Felix Tempel and Daniel Groos and Espen Alexander F. Ihlen and Lars Adde and Inga Strümke},
      year={2024},
      eprint={2412.16003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.16003}, 
}
```
