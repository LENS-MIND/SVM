# Support Vector Machine applied to Electron Energy Loss Spectroscopy (EELS)

## Author : Daniel del Pozo Bueno

Soft-margin Support Vector Machine (SVM) applied to EEL spectra for determining the oxidation state of Iron Oxides an Manganse oxides. 

This is not a maintained repository. It is a collection of scripts.

## Libraries or Dependencies: 

- HyperSpy 1.3
- numpy 1.13.1
- scikit-learn 1.18
- pandas 0.18.1
- matplotlib 1.5.3

## Content: 

-The repository contains the datasets built for this analysis in the folder /Data.

-The repository contains the results of the hyperparameter optimization in the folder /Gridsearch_results.

-The repository includes 3 python files where all the functions used in this study are presented:

	SVM_Preprocessing: functions involved in the spectrum images processing to get the spectral dataset. 
	SVM_Test: contain all the functions involving the testing the performance of the SVM algorithm to classifying EEL spectra. 
	SVM_Train: cotain all the functions involved in the training and optimization of the SVM estimator. 
	
-A jupyter notebook is also included with an example of use of all the functions include in the python files. 

## Usage: 
