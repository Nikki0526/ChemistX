# Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation in Chemical Synthesis and Self-driving Lab
This is the implementation for our paper [Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation in Chemical Synthesis and Self-driving Lab](https://arxiv.org/abs/2311.10776).

## Workflow
![image](https://github.com/Nikki0526/ChemistX/blob/main/workflow_diagram.PNG)

### Phase 1: Information Retrieval

- Given the provided professional chemical database, load the knowledge into the default OpenAI agent to enable generation of reaction condition optimization tasks.
- Validate the model stability of the Top Match Slice (TMS) selection and the automatic code generation using GPT.
- The output should be data fetched from online sources.

### Phase 2: Final Recommendation

- Using the CIMG descriptor, coarse yield label generation, the SCL network, and machine learning models, train the system to output recommended reaction conditions.

### Phase 3: Robotic System Control

- Deploy and execute the reactions on the machine to validate real-world performance.

## Setup
### Hardware requirements
The software requires only a standard computer with enough RAM.

### Software requirements
The software has been tested on the Colab / Ubuntu 18.04 system.

### Python Dependencies
```
 - openai==1.3.7
 - tensorflow==2.13
 - numpy==1.25.2
 - pandas==1.4.0
 - keras==3.8.0
 - chromadb
 - langchain
 - sklearn
```
### Repo Structure

```md
.
├── src/
│   ├── search_test.py
│   ├── search.py
│   ├── train_and_search.py
│   ├── data_train/
│   │   ├── data_additive.csv
│   │   ├── data_projected_vector.csv
│   │   └── data_test_20231007.csv
│   ├── data_test/
│   │   ├── data_additive_mean.csv
│   │   └── wetlab_subspace.csv
│   └── ...
├── phase1.ipynb
├── phase2.ipynb
├── phase3.ipynb
├── workflow_diagram.png
└── README.md
```
#### `src/` contains funciton codes for `phase 2`

TLDR, `train_and_search.py` is used for end-to-end training + inference during model development. `search.py` is used for inference only, when a trained model is already available. `search_test.py` is used for internal testing and development.

#### `search_test.py`

`search()` ranks chemical combinations based on predicted effectiveness using a machine learning model, leveraging chemical descriptors (CIMG vectors) retrieved via web automation.

```python
def search(p_raw, p_add, p_en, p_dim_reducer, p_model)
```
| Parameter        | Description |
|------------------|-------------|
| `p_raw`          | Path to raw SMILES dataset |
| `p_add`          | Path to additive dataset |
| `p_en`           | Path to pre-trained encoder model |
| `p_dim_reducer`  | Path to pre-trained dimension reducer model |
| `p_model`        | Path to pre-trained model |

#### `search.py`

`search()` ranks chemical compounds based on ther `SMILES` strings using a machine learning pipeline. It outputs the top 5 most promising candidates as predicted by a pre-trained model.

```python
def search(p_test, p_test_add, p_en, p_dim_reducer, p_model)
```
| Parameter        | Description |
|------------------|-------------|
| `p_test`         | Path to SMILES test dataset |
| `p_test_add`     | Path to additive test dataset |
| `p_en`           | Path to pre-trained encoder model |
| `p_dim_reducer`  | Path to pre-trained dimension reducer model |
| `p_model`        | Path to pre-trained model |

#### `train_and_search`

`train_and_search()` combines model training and prediction for chemical compound evaluation, using machine learning models such as `Random Forest`, `XGBoost`, or `FFTransformer`.

```python
def train_and_search(p_train, p_train_add, p_test, p_test_add, p_en, t_model, f_nums)
```
| Parameter        | Description |
|------------------|-------------|
| `p_train`        | Path to the training dataset (projected vectors) |
| `p_train_add`    | Path to additive data for training |
| `p_test`         | Path to test SMILES dataset |
| `p_test_add`     | Path to additive data for test set |
| `p_en`           | Path to pre-trained encoder model (for embedding) |
| `t_model`        | Model type to use: `"rf"` for Random Forest, `"xgb"` for XGBoost, `"ft"` for FTTransformer |
| `f_nums`         | Number of PCA components/features to retain |

#### Demo

`phase1.ipynb`, `phase2.ipynb`, and `phase3.ipynb` are demos for each phase. We also provide online scripts deployed on `Colab` that allows users to easily access our code and environment, enabling them to reproduce our results by clicking:
 - Phase 1 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1y7x6EVxC0fZhyOJDY-ES2QFKgZALpv88/view?usp=sharing)


## Phase 1
* ``$ phase1.py`` and ``$ phase1.ipynb`` demonstrate the process in Phase 1, including top match slice (TMS) selection and automatic code generation with GPT. 
* The output should be the API code which could lead to similar molecules.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1y7x6EVxC0fZhyOJDY-ES2QFKgZALpv88/view?usp=sharing). 

## Phase 2
* ``$ phase2.py`` demonstrates the process in Phase 2, including the prompt we used and automatic code generation with GPT. 
* The output should be the information retrived from the Internet.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1sODYNcptTlt7QJY_73TRsQyXsKvuw4g-/view?usp=sharing).

## Phase 3
* ``$ phase3.py`` demonstrates the process in Phase 3, including CIMG descriptor and coarse yield labels generation, SCL Network, and ML models. 
* The output should be the recommended reaction condition.
*  We also provide a colab demo, which can help users easily access our code and environment and reproduce our results by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1oEq04sl2zEP1yMm6hKdZi5UsbRQE6Ajb/view?usp=sharing).

## Data
All the data needed for training and testing are stored in the ``/data`` folder. Generally, time for the installation and demo on a "normal" desktop computer should be about 20 minutes.


