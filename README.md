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
* ``$ phase1.py`` and ``$ phase1.ipynb`` demonstrates the process in Phase 1, including top match slice (TMS) selection and automatic code generation with GPT. 
* The output should be the API code which could lead to similar molecules.
*  We also provide a colab version, which can help users easily access our code and environment by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QqEA0MwoUKaBm0K_CiLT5ukgcKeDH42N?usp=sharing). 
