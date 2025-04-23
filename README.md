## TransDRP
Source code and data for "Knowledge-guided domain adaptation model for transferring drug response prediction from cell lines to patients"

![Framework of TransDRP](https://github.com/liuxuan666/TransDRP/blob/main/pipeline.png)  

## Requirements
* Python >= 3.7
* PyTorch >= 1.5
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09
  
## Overview 
TransDRP follows a two-phase fashion of pre-training and adaptation. In the first phase, we pre-train an encoder to extract shared representations from genomic profiles across both domains, and a multi-label graph neural network (GNN) decoder to predict various drug responses simultaneously on the source domain. In the second phase, we categorize data into different cancers based on clinical knowledge, and then implement a global-local domain adversarial strategy to generalize the multi-label decoder to target domain.

## Installation
1. Install anaconda:
Instructions here: https://www.anaconda.com/download/
2. pip install -r Requirements.txt
3. The data can be downloaded from here：https://drive.google.com/file/d/1rfFjvBqxO8vRJXdLCh_MPZZhe8E8DVZy/view?usp=sharing
4. Run main.py

