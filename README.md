# Support Vector Machine applied to Electron Energy Loss Spectroscopy (EELS)

## Author: Daniel del Pozo Bueno

Soft-margin Support Vector Machine (SVM) applied to EEL spectra for determining the oxidation state of Iron oxides an Manganese oxides through the study of their white lines. This repository includes the testing methodology to evaluate the SVM algorithm performance and validate their correct working. Also, it includes all datasets used for this work, in combination with functions to easily use the SVM algorithm as a classification tool. 

This is not a maintained repository. It is a collection of functions.

## Libraries or Dependencies: 

- HyperSpy 1.5.2
- numpy 1.17.4
- scikit-learn 0.22.2.post1
- scipy 1.3.1
- pandas 0.25.1
- matplotlib 3.1.1

## Contents: 

-The repository contains the datasets built for this analysis in the folder /Data.

-The repository contains the results of the hyperparameter optimization in the folder /Gridsearch_results.

-The repository includes 3 python files where all the functions used in this study are presented:

	SVM_Preprocessing: contains  functions involved in the spectrum images processing to get the spectral dataset. 
	SVM_Test: contains functions involving the testing the performance of the SVM algorithm to classifying EEL spectra. 
	SVM_Train: contains functions involved in the training and optimization of the SVM estimator. 
	
-A jupyter notebook is also included with an example of use of all these functions included in the python files. 

## Usage information:

The use of the different functions is applied and detailed in the SVM_notebook included. 

## Spectral datasets information:

Spectral datasets included: 
- Fe_All = PCA clean iron spectra. (Set1)
- Fe_raw = original iron specra. (Set2)
- Fe_Mn = PCA clean iron and manganese spectra. (Set3)
- Ferw_Mn = PCA clean manganese spectra and original iron spectra. (Set4)

Each dataset has associated a label file, the labels values are related with the oxide and its oxidation state as follows:

- Class 0: Iron oxide = wüstite (Fe+2)
- Class 1: Iron oxide = magnetite (Fe+2 2Fe+3)
- Class 2: Manganese oxide = Mn+2
- Class 3: Manganese oxide = Mn+3
- Class 4: Manganese oxide = Mn+4

## Gridsearch_results information: 

For each kernel and dataset the hyperparameters are optimized, their results are presented in the Gridsearch_results folder. The files name indicate the kernel used and the training set used. The kernels keys used are: lin = linear, rbf = radial basis function, sig = sigmoid. 



