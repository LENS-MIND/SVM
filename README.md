# Support Vector Machine applied to Electron Energy Loss Spectroscopy (EELS)

## Author: Daniel del Pozo Bueno

Soft-margin Support Vector Machine (SVM) applied to EEL spectra for determining the oxidation state of Iron Oxides an Manganese oxides. 

This is not a maintained repository. It is a collection of scripts.

## Libraries or Dependencies: 

- HyperSpy 1.5.2
- numpy 1.17.4
- scikit-learn 0.22.2.post1
- scipy 1.3.1
- pandas 0.25.1
- matplotlib 3.1.1

## Content: 

-The repository contains the datasets built for this analysis in the folder /Data.

-The repository contains the results of the hyperparameter optimization in the folder /Gridsearch_results.

-The repository includes 3 python files where all the functions used in this study are presented:

	SVM_Preprocessing: contains the functions involved in the spectrum images processing to get the spectral dataset. 
	SVM_Test: contains all the functions involving the testing the performance of the SVM algorithm to classifying EEL spectra. 
	SVM_Train: contains all the functions involved in the training and optimization of the SVM estimator. 
	
-A jupyter notebook is also included with an example of use of all the functions include in the python files. 

## Usage:

The use of the different functions is applied in the SVM_notebook included. 

## Relevant Spectral datasets details:

Spectral datasets included: 
- Fe_All = PCA clean iron spectra. 
- Fe_raw = original iron specra. 
- Fe_Mn = PCA clean iron and manganese spectra.
- Ferw_Mn = PCA clean manganese spectra and original iron spectra. 

Each dataset has associated a label file, the labels values are realted with the element and oxidation state as follows:

-0: iron oxide -> wustite (Fe+2)

-1: iron oxide -> magnetite (Fe+2 2Fe+3)

-2: manganese oxide -> Mn+2

-3: manganese oxide -> Mn+3

-4: manganese oxide -> Mn+4

## Relevant Gridsearch_results details: 

