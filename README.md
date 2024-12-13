# TransDRP
Source code and data for "Knowledge-guided domain adaptation model for transferring drug response prediction from cell lines to patients"

![Framework of TransDRP](https://github.com/liuxuan666/TransDRP/blob/main/pipeline.png)  

# Requirements
* Python >= 3.7
* PyTorch >= 1.5
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
* python main_cv.py \<parameters\>  #---Regression task with 5-fold CV
* python main_independent.py \<parameters\> #---Independent testing with 9(traing):1(testing) split of the dataset
* python main_classify.py \<parameters\> #---Binary classification task with IC50 values

The data can be downloaded from here： https://drive.google.com/file/d/15AcSmRdq2t4tlmOQd93Vn4Z7dohqmklg/view?usp=drive_link
