# MEvoN: Molecular Evolution Mechanism for Enhanced Molecular Representation

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-green)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)

## Overview

This project explores whether **Molecular Evolution Mechanism (MEvoN)** can enhance **Molecular Representation** for downstream tasks such as **Molecular Property Prediction (MPP)**. The framework consists of two main components:
1. **MEvoN Construction**: Simulates molecular evolution to generate enhanced molecular representations.
2. **MEvoN-based MPP Task**: Utilizes MEvoN-generated representations for molecular property prediction.

The experiments are conducted on the **QM7/9 dataset**, a widely used quantum chemistry dataset containing ~134,000 organic molecules with multiple property labels.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [MEvoN Construction](#mevon-construction)
3. [MEvoN-based MPP Task](#mevon-based-mpp-task)
4. [Visualization](#visualization)
5. [Contributing](#contributing)
6. [License](#license)

---

## Environment Setup

To set up the environment, follow these steps:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2. Create a conda environment and install dependencies:

```bash
conda create -n mevon python=3.8
conda activate mevon
```

For GPU support, ensure you have the correct version of CUDA installed. Then install PyTorch with:


```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```


## MEvoN Construction

To generate molecular representations using MEvoN, run the following script:

```
python graph\create\main_graph_qm9.py
```

This script calls ```graph\create\graph_v2.py``` for molecular graph generation. The generated results are saved in the ```graph/create``` directory with the following structure:

```
{
    "nodes": ["C#N", "C", "CC", ...],
    "edges": [
        ["C#C", "C"],
        ["C#N", "C"],
        ...
    ],
    "atom_groups": {
        "1": ["C", "N", "O"],
        "2": ["C#C", "C#N", "C=O", ...],
        ...
    }
}
```

Modify ```root/graph_v2.py``` to customize the generation process.

## MEvoN-based MPP Task
To evaluate the performance of MEvoN-enhanced representations, we use a GIN backbone for molecular property prediction. Run the following scripts to compare the results with and without MEvoN:

With MEvoN:

    python root/train_v2_qm9_GIN.py --label_name mu
Without MEvoN:

    python root/train_v2_qm9_only_gin.py --label_name mu
    
The model implementation can be modified in ```model_gin.py```, specifically in the drug_encoder method:


    def drug_encoder(self, graph):
        # Customize your molecular encoder here
        ...

    
## Visualization

Visualization scripts are provided to generate plots and insights:

Evolution Trends:           
```
python show_cases_tends_evo_mols.py
```
Hyperparameter Analysis:    
```
python show_hyperparam.py
```
Run these scripts to reproduce the visualizations in the paper.

## Contributing

We welcome contributions! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with a detailed description of your changes.


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Acknowledgments

For questions or feedback, please contact likun98@whu.edu.cn.


