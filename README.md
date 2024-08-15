# Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation in Chemical Synthesis and Self-driving Lab
This is the implementation for our paper "Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation in Chemical Synthesis and Self-driving Lab".

## Setup
### Hardware requirements
The software requires only a standard computer with enough RAM.

### Software requirements
The software has been tested on the Colab / Ubuntu 18.04 system.

### Python Dependencies
```
 - openai==1.3.7
 - tensorflow==2.13
 - chromadb
 - langchain
 - sklearn
```
### Installation Guide
Clone this repo from github
```
git clone https://github.com/Nikki0526/ChemistX.git
```

## Workflow
![image](https://github.com/Nikki0526/ChemistX/blob/main/workflow.png)

## Phase 1
* ``$ phase1.py`` and ``$ phase1.ipynb`` demonstrate the process in Phase 1, including top match slice (TMS) selection and automatic code generation with GPT. 
* The output should be the API code which could lead to similar molecules.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QqEA0MwoUKaBm0K_CiLT5ukgcKeDH42N?usp=sharing). 

## Phase 2
* ``$ phase2.py`` demonstrates the process in Phase 2, including the prompt we used and automatic code generation with GPT. 
* The output should be the information retrived from the Internet.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12qmYG83HnN_mpt9GqoN2pgVWc6tNPJu2?usp=sharing).

## Phase 3
* ``$ phase3.py`` demonstrates the process in Phase 3, including CIMG descriptor and coarse yield labels generation, SCL Network, and ML models. 
* The output should be the recommended reaction condition.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1307u0ZY4kOX84CNBbCrVcJ74dsbHKVsy/view?usp=sharing).

## Data
All the data needed for training and testing are stored in the ``/data`` folder. Generally, time for the installation and demo on a "normal" desktop computer should be about 20 minutes.


