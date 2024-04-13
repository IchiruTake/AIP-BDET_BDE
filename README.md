# Bit2EdgeV2-BDE Repository

## Introduction

This is a repository for our study: 'AIP-BDET: Bond Dissociation Energy Prediction with Layers of Isolated Substructure in Organic Molecules under the Lightweight Deep Feed-Forward Networks'. This paper intended to provide a standardized way to leverage edge properties prediction such as Bond Dissociation Energy (BDE) and expected to Bond Dissociation Free Energy (BDFE) using only small-sized molecules. This research paves the way to predict these edge properties in the isolated locally substructure, proving that a full-sized molecule requirement can be sometimes un-necessary.

For any additional detail, please refer to the paper at this link: <link>

## Improvement and Limitation

**Improvements**: The uttermost improvement over many prvious research is the ability to predict these edge properties in the isolated locally substructure, and is safely trained without reaction duplication in the a molecule (e.h hydrogen-based bond), and delivers 600 to 800 reactions per second on a lower-end computing platform (tested with Intel i7-8850H and NVIDIA Quadro P1000). The possibility of heterogenous reaction data is also possible, but the accuracy may be varied and not be tested yet. A "sphere" definition and several other studies has aligned with this topic and hypothesize that the edge properties can be predicted in the isolated locally substructure.

**Limitations**: Ignoring the current model are trained on a small chemical space, this model is not suitable if the attempted molecule is not octet-rule satisfaction but a radical. For example, the model would fail on predict at [CH3] and [CH2] radicals, etc; or a complex molecule where the possibility of the number of radical atoms and its "radical" level is limitless.

**Performance**: In general you would observe a fast model training with comparable accuracy to many graph-based network such as ALFABET, only a minor accuracy loss on individual model is observed because the contribution of "outer" atom is ignored, and the atom at the boundary is the "radical" atom. Regards to the model training speed, we are capable of training the model in less than 20 minutes (or 12 minutes without Early Stopping) on Quadro P1000 over 265k reactions and took around 30 epochs to converge, proving the capability of scalability and interpretability.

## Installation

The installation is extremely simple, just ensure that the Anaconda is installed on your computer, and then install the environment at location /etc/requirements-conda.txt by running the following command:

```bash
conda create --name bit2edgev2-bde --file /etc/requirements-conda.txt
```

## Model Evaluation

**Configuration**: The model is trained under medium-sized molecule (up to 10 heavy atoms) with only 265k random unique reactions (~25k reactions are used for validation and testing). The environment's size is {6, 4, 2} and the fingerprint is ECFP4/1024 - FCFP6/1024 - AP.1024. The model is trained by default Adam optimizer with Huber loss function (alpha=3.5) and a step-wise learning rate. The model is trained with a simple Early Stopping with a patience of 20 epochs and a delta of 0.01. The model is trained three times each on five different seeds (0, 1, 2, 3, 42), and doing a Linear Regression with Excel across individual models.

*Note*: There are some different of model sizing between each individual seed model because of feature cleaning that operated over the bit vector. However, most changes are minor and the model is still comparable when the performance difference between the validation set and testing set is minor. The "S" stands for the seed or single model with best of 3, and the "E" stands for the ensemble model and best of 1.

- Significant Number: 2-digit after comma
- Epochs (Avg): 50.6 ± 2.13 
- Dataset: BDE-db dataset (PubChem-derived, Source: ALFABET) --> Train: 265k, Dev-Test: 12.5k (4.3 %) 
- Target: BDE (kcal/mol) ~ M06-2X/def2-TZVP

| Metrics      | Train-S       | Dev&Test-S   | Dev&Test-E   |
|--------------|---------------|--------------|--------------|
| MAE          | 0.18 ± 0.03   | 0.72 ± 0.02  | 0.65 ± 0.01  |
| RMSE         | 0.28 ± 0.04   | 1.35 ± 0.04  | 1.28 ± 0.05  |
| Acc (< 1.0)  | 99.36 ± 0.23  | 79.58 ± 0.89 | 82.12 ± 0.31 |
| Acc (< 2.5)  | 99.94 ± 0.02  | 95.50 ± 1.01 | 95.75 ± 0.18 |
| Out (> 5.0)  | 0.009 ± 0.003 | 1.04 ± 0.05  | 0.94 ± 0.03  |
| Out (> 10.0) | 0.003 ± 0.001 | 0.17 ± 0.02  | 0.15 ± 0.02  |
| -            | -             | -            | -            |

## Model Evaluation (Extended)
- Significant Number: 2-digit after comma
- 5 trains on 5 seeds
- Dataset: BDE-db dataset (PubChem-derived, Source: ALFABET) --> Train: 441k, Dev|Test: 38.9k (7.5%) 
- Target: BDE (kcal/mol) ~ M06-2X/def2-TZVP
- Arch: {6, 3} - 12.3M+ params - ECFP4/1024 FCFP6/1024 AP1024

| Metrics      | BDE-S        | BDFE-S       | BDSCFE-S     |
|--------------|--------------|--------------|--------------|
| MAE          | 0.80 ± 0.02  | 0.79 ± 0.01  | 0.81 ± 0.02  |
| RMSE         | 1.64 ± 0.07  | 1.63 ± 0.07  | 1.66 ± 0.07  |
| Acc (< 1.0)  | 77.80 ± 0.59 | 78.45 ± 0.45 | 77.17 ± 0.70 |
| Acc (< 2.5)  | 94.47 ± 0.17 | 94.87 ± 0.14 | 94.30 ± 0.20 |
| Out (> 5.0)  | 1.40 ± 0.09  | 1.35 ± 0.10  | 1.43 ± 0.11  |
| Out (> 10.0) | 0.32 ± 0.02  | 0.32 ± 0.03  | 0.33 ± 0.02  |
| -            | -            | -            | -            |


# References

[1] St. John, P.C., Guan, Y., Kim, Y. et al. Prediction of organic homolytic bond dissociation enthalpies at near chemical accuracy with sub-second computational cost. Nat Commun 11, 2328 (2020). <https://doi.org/10.1038/s41467-020-16201-z>

[2] Wen, M., Blau, S. M., Spotte-Smith, E. W. C., Dwaraknath, S., & Persson, K. A. (2021). BonDNet: a graph neural network for the prediction of bond dissociation energies for charged molecules. Chemical science, 12(5), 1858-1868. <https://doi.org/10.1039/D0SC05251E>

[3] Luo, Y.-R. (2007). Comprehensive Handbook of Chemical Bond Energies (1st ed.). CRC Press. <https://doi.org/10.1201/9781420007282>

[4] Wen, M., Blau, S. M., Xie, X., Dwaraknath, S., & Persson, K. A. (2022). Improving machine learning performance on small chemical reaction data with unsupervised contrastive pretraining. Chemical science, 13(5), 1446-1458. <https://doi.org/10.1039/D1SC06515G>
