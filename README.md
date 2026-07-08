# MARLEY V2: EU-ETS under attack? The impact of carbon price suppression on the decarbonization of the power sector

## Overview

This repository contains code used for **EU-ETS under attack? The impact of carbon price suppression on the decarbonization of the power sector**. The Reinforcement Learning implementation is based on the Ray and RLLIB MARL implementations, available in https://github.com/ray-project. 

---

## Repository Structure
```
└── training_IPPO.py         # Training scripts and configuration for IPPO
└── training_MAPPO.py        # Training scripts and configuration for MAPPO
└── data_CM_CfD.py           # Scripts for evaluating trained agents
└── ray_environment.yml      # Conda environment specification
└── README.md                # This file
├── MAPPO/                   # MAPPO implementation
├── RL_env/                  # Environments for training and sampling
├── checkpoints/             # Exemplary trained checkpoint (CRM+CfD, 16 agents)
├── excel_scenario_files/    # Base excel files with information for the Italian scenarios studied in the publication
├── utils/                   # Utility scripts for harnessing training data from RLLIB and final results from trained agents, plotting and analysis

```

---

## Installation

### Main Requirements

- Python 3.9
- Conda or Miniconda
- Recommended Linux-based system

### Setup

1. **Clone this repository:**
```bash
   git clone https://github.com/jjgonzalez2491/MARLEY_V2.git
   cd MARLEY_V2
```

2. **Create and activate the conda environment:**
```bash
   conda env create -f ray_environment.yml
   conda activate ray_environment
```

3. **Adjust the corresponding directories**

4. **Modify the training script and the base excel files to set up the simulation**
---

## Usage

### Training

To train agents in the electricity market environment using IPPO and MAPPO configurations:
```bash
python training_IPPO.py
python training_MAPPO.py
```
To configure the scenario, it is necessary to modify the corresponding variables in the training script and in the corresponding excel file. 

> **Note:** The training script is configured for HPC environments with LSF job scheduling. Modifications may be required for local execution (parallel environments for sampling, maximum GPU memory limits, among others). 

### Evaluation

To evaluate trained agents (used the configuration file to select the corresponding checkpoint and algorithm):
```bash
python data_CM_CfD.py
```
> **Note:** The evaluation script reads a RLLIB checkpoint obtained after training. For testing purposes, an exemplary checkpoint for a system with **16 agents**, for both IPPO and MAPPO configurations, is provided in the `checkpoints/` folder.

### Data Files

Base Excel files containing market parameters, technology characteristics, and demand profiles are located in the `excel_scenario_files/` folder. These files define the stylized Italian electricity system used in the paper.

### Visualization

Plotting utilities are provided in the `utils/` folder for generating market outcome visualizations and performance metrics.

---

## Contact

For questions, issues, or contributions, please contact: javier.gonzalez@cmcc.it and/or carlos.rodriguez@cmcc.it. 

---

## Acknowledgments

Authors acknowledge support from the European Research Council, ERC grant agreement number 101044703 (EUNICE) CUP D87G22000340006,from European Union PNRR - Missione 4–Componente 2–Avviso 341 del 15/03/2022 - Next Generation EU, in the framework of the project GRINS - Growing Resilient, INclusive and Sustainable project (GRINS PE00000018 – CUP C83C22000890001), and from the ONESYSTEM research project (grant CPP2022-009711), funded by MICIU/AEI /10.13039/501100011033 and by the European Union NextGenerationEU/PRTR. Finally, MARLEY is only possible thanks to the Ray and RLLIB libraries, available in https://github.com/ray-project. 
