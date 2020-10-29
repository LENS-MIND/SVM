import sys, os
import hyperspy.api as hs
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import cm
from time import time
from random import randrange
from sklearn.externals import joblib

def plot_randomly_spectra(samples,k,dig_spectra=None,fig_save=None):
    '''
    This functions plot k number of spectra from a given set of spectra.
    It also incopore digitized spectra if theu are provided. 

    -samples: array-like of shape (n_spectra,n_features).
    -k: int. 
    -dig_spectra: array-like of shape (n_spectra,n_features) (default=None).
        Index of the random spectra to be in the plot.
    -fig_save: string. 
        Name of the file to save the plot. 
    
    Returns:
    --------
    A plot with k spectra. 
    '''
    
    legend_elements = [mpatches.Patch(color='black')]
    comap = cm.get_cmap(name='rainbow',lut=100)

    fig = plt.figure(figsize=(8,8),dpi=200)
    if dig_spectra is not None: 
        for i, spec in enumerate(dig_spectra):
            plt.plot(np.arange(700.,730.,step=.1),spec,color='black',linewidth=0.7,linestyle='-')
        
    lenght = len(samples)
    for i in range(0,k):
        spectro_ind = randrange(0,lenght)
        plt.plot(np.arange(700.,730.,step=.1),samples[spectro_ind],color=comap(100-i*12),linewidth=0.7,linestyle='-')
        
    plt.xlabel('Energy Loss (eV)')
    plt.ylabel('Normalized Intensity')
    
    plt.vlines(x=710.,ymin=-0.1,ymax=1.1,linestyles=':',linewidth=2,color='red')
    plt.gca().set(xlim=(700.,730.),ylim=(-0.01,1.01))
    plt.xticks(np.arange(700.,735., step=5))
    plt.yticks(np.arange(0,1.1, step=0.1))
    
    labeles = ['Random Spectra ','Digitized Spectra']
    cmapa_hand= [mpatches.Patch(color='red'),mpatches.Patch(color='black')]
    
    #leyenda
    plt.legend(handles=cmapa_hand,
               labels=labeles,
               title='Iron oxide',
               fontsize=18,title_fontsize=18)    
    plt.show()
    
    if fig_save is not None:
        fig.savefig(fig_save,bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')

    return None

def train_and_test_svm_model(samples,labels,kernel,C=1.,gamma=None,coef0=None,test_size=0.35,cv=10):
    '''
    This function trains a SVM estimator with the parameters provided and tests this model via cross-fold validation. It returns the trained model. 

    -samples: array-like of shape (n_spectra,n_features).
        Whole spectral dataset.
    -labels: array-like of shape (n_spectra).
        Labels.
    -test_size: float between 0 and 1.0. (default=0.35)
        It represents the proportion of the dataset to include in the test split.
    -cv: int. (default=10)
        Cross validation fold.
    -kernel: string. 
        Specifies the kernel type to be used in the algorithm. 
        Valid values: 'rbf','linear','sigmoid'.
    -C: float. (default=1.) 
        Regularization parameter.
    -gamma: float. 
        Kernel coefficient for 'rbf' and 'sigmoid'.
    -coef0: float.
        Independent term of the kernel function.

    Returns:
    --------
    Trained (or fit) SVM estimador. 
    '''
    
    x_train, x_test, y_train, y_test = train_test_split(samples,labels,test_size=test_size)

    if kernel == 'rbf':
        model = SVC(C=C,kernel=kernel,gamma=gamma,probability=False,decision_function_shape='ovr',class_weight='balanced')
    elif kernel == 'sigmoid':
        model = SVC(C=C,kernel=kernel,gamma=gamma,coef0=coef0,probability=False,decision_function_shape='ovr',class_weight='balanced')
    else: 
        model = SVC(C=C,kernel=kernel,probability=False,decision_function_shape='ovr',class_weight='balanced')
    
    #Test the model:
    score = cross_val_score(estimator=model,X=samples,y=labels,cv=cv)
   
    #Train the model: 
    model.fit(x_train, y_train)

    #Test the model:
    print('Model test accuracy: ',model.score(x_test,y_test))
    print('Averaged 10-fold cross-validation accuracy: %0.2f (+/- %0.2f)' % (score.mean(), score.std()*2))
    
    return model

def optimize_SVM_clasifiers(parameters,samples,labels,test_size=0.35,file_save=None): #Tested
    '''
    This function optimize the hyperparameters of a Support Vector Machine model, 
    save the model in a file called by the file_save provided and return the models.
    
    -parameters: dict or list of dictionaries. 
        Dictionary with parameters names (string) as keys and lists of parameter settings to try as values. 
        The parameters keys are: kernel, C range, gamma range, and r range.
        Examples of parameters: 
        Linear kernel: parameters = {"kernel":["linear"],"C":np.logspace(-3,4,8)}
        RBF kernel: parameters = {"kernel":["rbf"],"C":np.logspace(-3,4,8),"gamma":np.logspace(-6,4,11)}
        Sigmoid kernel: parameters = {"kernel":["sigmoid"],"C":np.logspace(-3,4,8),
                                    "gamma":np.logspace(-6,4,11),"coef0":np.logspace(-5,3,9)}
    -samples: array-like of shape (n_spectra,n_features)
        Whole spectral dataset.
    -labels: array-like of shape (n_spectra).
        Labels.
    test_size: float from 0 to 1. (default 0.35) 
        It represents the proportion of the dataset to include in the test split.
    fig_save: string. 
        Name of the file to save the plot. 

    Returns
    -------
    Dict of numpy ndarrays with the results from the Grid search.
    '''
    
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size)
    svm = SVC(probability=False,decision_function_shape='ovr',class_weight='balanced')
    grid_time = time()
    results = GridSearchCV(svm,parameters,cv=10,n_jobs=-1,verbose=10)
    results.fit(x_train,y_train)
    grid_time = time() - grid_time

    print('Time comsuming of the Gridsearch: ',grid_time,'seconds. \n')
    print('The best model is: \n')
    print(results.best_estimator_)

    if file_save is not None:
        joblib.dump(results,file_save)
        print('Gridsearch results saved.')
    else: 
        print('Gridsearch results are not saved.')
    
    return results

def plot_gridsearch_results(grid_results,fig_save=None): #Tested!
    '''
    This function plot the grid results depending on the kernel function used. 

    -grid_results: Dict of numpy ndarrays with the results from the Gridsearch function.
    -fig_save: string. 
        Name of the file to save the plot. 

    Return:
    -------
    Plot the estimator accuracy results as a function of the parameters of each SVM estimator. 
    '''
    
    parameters = grid_results.get_params()['param_grid']
    comap = cm.get_cmap(name='Spectral',lut=150)

    if parameters['kernel']==['linear']:
        
        scores = [x for x in grid_results.cv_results_['mean_test_score']]
        errors = [x for x in grid_results.cv_results_['std_test_score']]
        C = parameters['C']
        
        fig = plt.figure(figsize=(7,7),dpi=200)
        plt.xscale('log')
        plt.xticks(C)
        plt.errorbar(C,scores,linestyle='-',linewidth='0.8',color='green',yerr=errors,capsize=2,ecolor='blue',elinewidth=1.,label='OVO Method')
        plt.grid(True,'major','both',linestyle = '-',color = 'lightgray')
        plt.title('Linear Kernel')
        plt.xlabel('C Values')
        plt.ylabel('Averaged 10-fold cross-validation accuracy')
        plt.legend(loc=2)
        plt.show()
    
        #Best estimator information:
        print('The optimum hyperparameters found by the Gridsearch are:')
        print(grid_results.best_params_,'\n')
        print('The best estimator gets an score of: ',grid_results.best_score_)
        
        if fig_save is not None:
            fig.savefig(fig_save,bbox_inches = 'tight')
            print('Figure saved with name: ',fig_save)
        else:
            print('Figure not saved.')
            return None

    elif parameters['kernel']==['rbf']:
        
        scores = [x for x in grid_results.cv_results_['mean_test_score']]
        errors = [x for x in grid_results.cv_results_['std_test_score']]   
        
        #Parameters arrays: 
        C = parameters['C']
        gamma = parameters['gamma']
        
        sco = np.array(scores).reshape(len(C), len(gamma))
        err = np.array(errors).reshape(len(C), len(gamma))
        
        fig = plt.figure(dpi=200,figsize=(10,8))
        plt.xscale('log')
        
        for ind, i in enumerate(C):
            plt.errorbar(gamma, sco[ind],color=comap(16*ind),linewidth='0.8',yerr=err[ind],capsize = 2,ecolor='blue',elinewidth=.5, label='C: ' + str(i))
        
        plt.grid()
        plt.xticks(gamma)
        plt.legend(loc=1)
        plt.title('RBF kernel')
        plt.xlabel('Gamma')
        plt.ylabel('Averaged 10-fold cross-validation accuracy')
        plt.show()
        
        #Best estimator information:
        print('The optimum hyperparameters found by the Gridsearch are:')
        print(grid_results.best_params_)
        print('The best estimator gets an score of: ',grid_results.best_score_)        
          
        if fig_save is not None:
            fig.savefig(fig_save,bbox_inches = 'tight')
            print('Figure saved with name: ',fig_save)
        else:
            print('Figure not saved.')
            return None

    elif parameters['kernel']==['sigmoid']:
        figs = []
        C = parameters['C']
        gamma = parameters['gamma']
        r = parameters['coef0'] 
        
        scores = [x for x in grid_results.cv_results_['mean_test_score']]
        errors = [x for x in grid_results.cv_results_['std_test_score']]
        
        sig = np.array(scores)
        sig = sig.reshape(len(C),len(r),len(gamma))
        
        for ind, i in enumerate(C):
            figs.append(plt.figure(dpi=150))
            for index,j in enumerate(r):
                plt.xticks(gamma)
                plt.ylim(0.3,1.)
                plt.xscale('log')
                plt.grid()
                plt.title('C value : '+str(i))
                plt.plot(gamma, sig[ind,index], label=str(j))
                plt.xlabel('Gamma values')
                plt.ylabel('Averaged 10-fold cross-validation accuracy')
                plt.legend(title='r values:',loc=1)
                plt.show()

        #Best estimator information:
        print('The optimum hyperparameters found by the Gridsearch are:')
        print(grid_results.best_params_)
        print('The best estimator gets an score of: ',grid_results.best_score_)       
        
        if fig_save is not None:
            for i in range(0,len(figs)):
                figs[i].savefig(fig_save+str(i),bbox_inches = 'tight')
            print('Figures saved.')
        else:
            print('Figure not saved.')
            return None
    else:
        print('No valid gridsearch results were provided.')
        return None