# Bit2EdgeV2-BDE Repository

# Introduction

This is a repository for our paper: 'AIP-BDET: Populating A Toy-Modeling Network for Bond Dissociation Energy Prediction to Chemistry Accuracy by 30 minutes'. This paper intended to provide a standardized way to leverage edge properties prediction such as Bond Dissociation Energy (BDE) and Bond Dissociation Free Energy (BDFE) using only small-sized molecules. The feature engineering is well performance with approximate 950-1000 samples per seconds with four different type of rule-based hashed bit-type molecular fingerprints.

Download/Preview the paper at this link: <link>


**RESEARCH GAP**:
- According to the reference [3] (page 3), "the equilibrium constant Keq is very sensitive to any error in the BDEs. An error of 1, 2, or 3 kcal leads to an error of a factor of 5.4, 29.2, or 158, respectively, in the equilibrium constant Keq at 298 K. The current experimental uncertainty of absolute majority of BDE data is within 1–2 kcal/mol; therefore, the uncertainty is greater than chemical accuracy (1 kcal/mol)." 

- The training of each of two reference graph-based models (ALFABET [1] and BonDNet [2]) is quite costly whilst the task's requirement demands a high performance in the prediction. Moreover, to alleviate the model's bias and variance and boost the accuracy, multiple models should be deployed to compensate these errors. Moreover, the training dataset applied is currently limited with only 280k reactions on four common atoms (C, H, O, N) with at most ten heavy atoms. Whilst the current chemical space is enormous, which is up to 10^(60-80) possible medium-sized molecules with many common atoms such as the C, H, O, N, P, S, Cl, F, ... *To gain better performance, we would need around 500k molecules for these four common atoms and up to 10-50 millions for the extension above.*

- Additional question is that how many information within the input context is __*enough*__? And is it possible to predict these edge properties in the isolated locally substructure and/or viewing the input ONCE only ?


**HIGHLIGHTS**:
- The given model called AIP-BDET shown the possibility of predict these edge properties in the isolated locally substructure. This answer would help researchers to analyze the importance of these local substructure nearby to determine the average feature importance that contribute towards the prediction.
- Unlike ALFABET and BonDNet, the training dataset does not contain any duplicated reactions such as the C-H bond or mostly hydrogen-based bond. For example, C2H6 have 7 bonds in total, but only two of them are unique.
- This study does not attempt to leverage or beaten any state-of-the-art model achieving chemistry accuracy, but to derive a more reasonable and predictable edge-property (BDE) prediction under the hypothesized reference (DFT).
- This model can generate multiple variants using different input representations molecular fingerprints.


# Property-Performance

**Property - Behaviour**:

- Dataset: Small-sized molecules 42.5k mols and 290k unique BDEs (adapted from ALFABET paper: Current online version have more BDEs).
- Training: Only unique reactions are considered. The validation set and testing set holds around 6.6k data points each (2.3 % each).
- Model Components: 
    + G-model: Localized Bond Information (LBI) >-~-> Derive a base context representation.
    + M-model: Bond Environment(s) + Localized Bond Information >-~-> Derive an advanced representation in same time.
    + E-model: Bond Environment(s) + Localized Bond Information >-~-> Adjust an advanced representation on each bond environment relied on the base context information.

**Model Setup**:
- Input Scheme: BS-BDE; Environment's Size: 4 & 2; 
- Fingerprints: ECFP4/1024 - FCFP6/1024 - AP.1024 - TT.1024 
- Compilation: Huber(delta=3.5) loss + Adam Optimizer + Custom Learning Rate
- ZERO data augmentation, Dropout layer, BatchNorm layer or any strict set-up.
- Computer: Windows 10 Pro 21H2, 16 Gb RAM, NVIDIA Quadro P1000 (47W, 1.89 TFLOPS) Notebooks (Dell Precision 5530).
- Libraries/Packages: Python 3.8, RDKit 2022.03, TensorFlow 2.5.3, TensorFlow Addons 0.16, Cudatoolkit 11.2, ...


**Performance**:

- Dataset: 277.6k data points with DFT Reference (M06-2X corrected)
- Model: 9.1M params + BS-BDE (4, 2) 
- Training Time: 66.7 ± 15.1 epochs (18s/epoch, 20 epochs of Early Stopping, delta=0.01) or less than 30 minutes totally.
- Runtime Environment: Five most common seeds (0, 1, 2, 3, 42), each runs three times.


BS-BDE (4, 2) & ECFP4/1024 - FCFP6/1024 - AP.1024 - TT.1024 & 9.1 M+ params 
Epochs (Avg): 66.73 ± 15.15 ~ 18-21 (s/epoch)
-------------------------------------------------------
|     Metrics     |     Dev-Test    |      Train      |
|-----------------|-----------------|-----------------|
| MAE             |  0.802 ± 0.011  |  0.392 ± 0.034  |
| RMSE            |  1.462 ± 0.037  |  0.709 ± 0.036  |
| Acc (< 1.0, %)  |  76.28 ± 0.457  |  93.12 ± 0.958  |
| Acc (< 2.0, %)  |  91.39 ± 0.330  |  98.26 ± 0.127  |
| Acc (< 2.5, %)  |  94.24 ± 0.271  |  98.96 ± 0.062  |
| Acc (< 3.0, %)  |  95.94 ± 0.186  |  99.32 ± 0.043  |
| Out (> 5.0, %)  |  1.260 ± 0.083  |  0.185 ± 0.011  |
| Out (> 7.5, %)  |  0.440 ± 0.037  |  0.056 ± 0.005  |
| Out (> 10.0, %) |  0.200 ± 0.037  |  0.020 ± 0.002  |
| Out (> 20.0, %) |  0.029 ± 0.011  |  0.004 ± 0.001  |
-------------------------------------------------------

BS-BDE (6, 4, 2) & ECFP4/1024 - FCFP6/1024 - AP.1024 & XXX M+ params 
-------------------------------------------------------
|     Metrics     |     Dev-Test    |      Train      |
|-----------------|-----------------|-----------------|
| MAE             |  /////////////  |  /////////////  |
| RMSE            |  /////////////  |  /////////////  |
| Acc (< 1.0, %)  |  /////////////  |  /////////////  |
| Acc (< 2.0, %)  |  /////////////  |  /////////////  |
| Acc (< 2.5, %)  |  /////////////  |  /////////////  |
| Acc (< 3.0, %)  |  /////////////  |  /////////////  |
| Out (> 5.0, %)  |  /////////////  |  /////////////  |
| Out (> 7.5, %)  |  /////////////  |  /////////////  |
| Out (> 10.0, %) |  /////////////  |  /////////////  |
| Out (> 20.0, %) |  /////////////  |  /////////////  |
-------------------------------------------------------

RG-BDE (4, 2) & ECFP4/1024 - FCFP6/1024 - AP.1024 & 10.6 M+ params 
Epochs (Avg): 58.27 ± 11.28 ~ 41-43 (s/epoch) (Dev-Test: 4.6 %). 
G-model: 2 blocks (G-Scale: 1, 0.5)
-------------------------------------------------------------------------------------------
|     Metrics     |     Dev-Test    |      Train      |  Dev-Test (Ens) |   Train (Ens)   |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| MAE             |  0.804 ± 0.014  |  0.365 ± 0.041  |  0.747 ± 0.014  |  0.301 ± 0.009  |
| RMSE            |  1.474 ± 0.031  |  0.678 ± 0.040  |  1.410 ± 0.026  |  0.624 ± 0.007  |
| Acc (< 1.0, %)  |  76.20 ± 0.610  |  93.77 ± 1.041  |  78.54 ± 0.701  |  94.99 ± 0.077  |
| Acc (< 2.0, %)  |  91.28 ± 0.316  |  98.36 ± 0.135  |  92.14 ± 0.405  |  98.51 ± 0.025  |
| Acc (< 2.5, %)  |  94.12 ± 0.209  |  99.01 ± 0.073  |  94.67 ± 0.268  |  99.09 ± 0.015  |
| Acc (< 3.0, %)  |  95.90 ± 0.153  |  99.28 ± 0.026  |  96.23 ± 0.159  |  99.41 ± 0.015  |
| Out (> 5.0, %)  |  1.297 ± 0.080  |  0.176 ± 0.010  |  1.194 ± 0.091  |  0.164 ± 0.008  |
| Out (> 7.5, %)  |  0.476 ± 0.044  |  0.051 ± 0.005  |  0.433 ± 0.036  |  0.048 ± 0.004  |
| Out (> 10.0, %) |  0.225 ± 0.026  |  0.018 ± 0.002  |  0.211 ± 0.024  |  0.017 ± 0.003  |
| Out (> 20.0, %) |  0.026 ± 0.008  |  0.004 ± 0.000  |  0.026 ± 0.006  |  0.003 ± 0.000  |
-------------------------------------------------------------------------------------------


RG-BDE (4, 2) & ECFP4/1024 - FCFP6/1024 - AP.1024 & 10.38 M+ params 
Epochs (Avg): 61.60 ± 9.13 ~ 41-43 (s/epoch) (Dev-Test: 8.6 %). 
G-model: 1 blocks (G-Scale: 0.5)
-------------------------------------------------------------------------------------------
|     Metrics     |     Dev-Test    |      Train      |  Dev-Test (Ens) |   Train (Ens)   |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| MAE             |  0.820 ± 0.011  |  0.355 ± 0.026  |  0.803 ± 0.029  |  0.378 ± 0.088  |
| RMSE            |  1.497 ± 0.045  |  0.670 ± 0.024  |  1.457 ± 0.081  |  0.666 ± 0.049  |
| Acc (< 1.0, %)  |  75.59 ± 0.506  |  94.17 ± 0.451  |  76.69 ± 1.060  |  94.45 ± 0.730  |
| Acc (< 2.0, %)  |  90.99 ± 0.268  |  98.42 ± 0.056  |  91.75 ± 0.273  |  98.48 ± 0.044  |
| Acc (< 2.5, %)  |  93.96 ± 0.171  |  99.04 ± 0.030  |  94.53 ± 0.191  |  99.09 ± 0.021  |
| Acc (< 3.0, %)  |  95.75 ± 0.129  |  99.38 ± 0.017  |  96.12 ± 0.226  |  99.41 ± 0.008  |
| Out (> 5.0, %)  |  1.390 ± 0.071  |  0.169 ± 0.007  |  0.664 ± 0.053  |  0.147 ± 0.004  |
| Out (> 7.5, %)  |  0.490 ± 0.046  |  0.051 ± 0.003  |  0.237 ± 0.035  |  0.044 ± 0.003  |
| Out (> 10.0, %) |  0.214 ± 0.023  |  0.018 ± 0.004  |  0.102 ± 0.025  |  0.016 ± 0.001  |
| Out (> 20.0, %) |  0.027 ± 0.011  |  0.004 ± 0.001  |  0.015 ± 0.008  |  0.003 ± 0.001  |
-------------------------------------------------------------------------------------------


RG-BDE (4, 2) & ECFP4/1024 - FCFP6/1024 - AP.1024 & 10.38 M+ params 
Epochs (Avg): 61.60 ± 9.13 ~ 41-43 (s/epoch) (Dev-Test: 8.6 %). 
G-model: 2 blocks (G-Scale: 1, 0.5)
-------------------------------------------------------------------------------------------
|     Metrics     |     Dev-Test    |      Train      |  Dev-Test (Ens) |   Train (Ens)   |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| MAE             |  0.820 ± 0.011  |  0.355 ± 0.026  |  0.803 ± 0.029  |  0.378 ± 0.088  |
| RMSE            |  1.497 ± 0.045  |  0.670 ± 0.024  |  1.457 ± 0.081  |  0.666 ± 0.049  |
| Acc (< 1.0, %)  |  75.59 ± 0.506  |  94.17 ± 0.451  |  76.69 ± 1.060  |  94.45 ± 0.730  |
| Acc (< 2.0, %)  |  90.99 ± 0.268  |  98.42 ± 0.056  |  91.75 ± 0.273  |  98.48 ± 0.044  |
| Acc (< 2.5, %)  |  93.96 ± 0.171  |  99.04 ± 0.030  |  94.53 ± 0.191  |  99.09 ± 0.021  |
| Acc (< 3.0, %)  |  95.75 ± 0.129  |  99.38 ± 0.017  |  96.12 ± 0.226  |  99.41 ± 0.008  |
| Out (> 5.0, %)  |  1.390 ± 0.071  |  0.169 ± 0.007  |  0.664 ± 0.053  |  0.147 ± 0.004  |
| Out (> 7.5, %)  |  0.490 ± 0.046  |  0.051 ± 0.003  |  0.237 ± 0.035  |  0.044 ± 0.003  |
| Out (> 10.0, %) |  0.214 ± 0.023  |  0.018 ± 0.004  |  0.102 ± 0.025  |  0.016 ± 0.001  |
| Out (> 20.0, %) |  0.027 ± 0.011  |  0.004 ± 0.001  |  0.015 ± 0.008  |  0.003 ± 0.001  |
-------------------------------------------------------------------------------------------






* In the paper, the accuracy gained is focused within the range of 2.0 (or 2.5) kcal/mol if sharing the same target reference. For other upstream tasks without addition training or pre-training, the accuracy metric are kept at 3.0 (or 3.5) kcal/mol and 5.0 kcal/mol (for outlier).

* In some specific random weight initialization, the training can be stopped too soon, you should re-train the model to obtain better result, or reduce the update difference of EarlyStopping. 

* The `(Ens)` stands for ensembled model, performed by doing Linear Regression on all child models at each seed on the training set and then inferred back to the dev-test set.


# Benefits: 

- Further Applications: 
    + Classification problems on substructure behaviour to determine the which drug-like molecules better.
    + Determine the equilibrium constant Keq 
    + ...

- Users (Students/Lecturers): Who studied in this or any related field wanted to adapt the high accuracy BDE for the thesis or dissertation, but lacking the dataset and the adjusted version of AIP-BDET to suit their need. The re-verification or custom model adjustment even on extreme hardware constraint with different fingerprint setups is also possible. This can mitigate the need of requesting high-performance computer/cluster (HPC), and other incurred costs.

- Users (Researcher/Practicioner): Who worked in this or any related field and any practical applications wanted to achieve the high accuracy BDE with maximal human confidence and experience. With many variants can be customized, the reliance on one single model or many models with similar behaviours towards some specific bonds or familiar structure is significantly minimized with better accuracy gained and less random model's bias through model ensembling. 

# Reliability and Warranty:

- The current DFT reference (M06-2X/def2-TZVP), is good in general, but is not good for all cases. Other target reference may be helpful despite the support range (by our knowledge) is still uncertrain. Large-sized molecule is also a considered factor not to let the prediciton (BDE) exploded or falsely misplaced.

- The current chemical space, as said, is limited to four common atoms on neutral-stable molecules with maximum ten heavy atoms on single bond. We are missing double-triple bond, more complex neighboring substructure, ring-bond, uncommon atoms, and charged bond as well. (Limited) variants of GNN can be built but achieving top-level accuracy may NOT be possible. Extra approach on feature engineering and model scaling may not directly solve the problems.

# Additional Performance


# Citations:


# References:

[1] St. John, P.C., Guan, Y., Kim, Y. et al. Prediction of organic homolytic bond dissociation enthalpies at near chemical accuracy with sub-second computational cost. Nat Commun 11, 2328 (2020). https://doi.org/10.1038/s41467-020-16201-z

[2] Wen, M., Blau, S. M., Spotte-Smith, E. W. C., Dwaraknath, S., & Persson, K. A. (2021). BonDNet: a graph neural network for the prediction of bond dissociation energies for charged molecules. Chemical science, 12(5), 1858-1868. https://doi.org/10.1039/D0SC05251E
 
[3] Luo, Y.-R. (2007). Comprehensive Handbook of Chemical Bond Energies (1st ed.). CRC Press. https://doi.org/10.1201/9781420007282

[4] Wen, M., Blau, S. M., Xie, X., Dwaraknath, S., & Persson, K. A. (2022). Improving machine learning performance on small chemical reaction data with unsupervised contrastive pretraining. Chemical science, 13(5), 1446-1458. https://doi.org/10.1039/D1SC06515G
