# TMSGT-DDI

**TMSGT-DDI** is a Graph Transformer-based framework for **Drug–Drug Interaction (DDI) prediction**, which integrates **textual semantic initialization** and **multi-scale subgraph modeling** to enhance representation learning of drugs.

---

## 1. Environment Setup


### Requirements

* Python 3.6.13
* CUDA 11.3 / 11.6

### Installation

```bash
# 1. Create and activate environment
conda create -n TMSGT python=3.6.13
conda activate TMSGT

# 2. Install PyTorch (CUDA 11.3 example)
conda install pytorch==1.10.2 cudatoolkit=11.3 torchvision==0.11.3 torchaudio==0.10.2 -c pytorch

# 3. Install PyTorch Geometric and dependencies
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-geometric==2.0.2 \
-f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# 4. Install other dependencies
pip install rdkit-pypi==2021.9.4
```
---

## 2. Dataset Preparation

The datasets can be downloaded from:https://drive.google.com/file/d/1j57Y6ZbSjEngP_0efe4U-HtdICOVFTQG/view?usp=drive_link

### Directory Structure

Place datasets under the `dataset/` directory as follows:

```
.
├── dataset
│   ├── drugbank
│   └── kegg
├── data                       # Store subgraph data
├── model                  
├── best_save              # Saved models
├── main.py                
├── train_eval.py          
├── utils.py               
└── preprocess_embeddings.py
```

---

## 3. Usage

### A. Preprocess Drug Text Embeddings

If you want to regenerate drug textual embeddings (e.g., using PubMedBERT):

```bash
python preprocess_embeddings.py
```
https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

---

### B. Train and Evaluate

#### DrugBank 

```bash
python main.py --dataset drugbank
```

#### KEGG 

```bash
python main.py --dataset kegg
```

---

## 4. Output

* Model checkpoints will be saved in:

  ```
  ./best_save/
  ```
---

