import matplotlib.pyplot as plt
import seaborn as sns;
from operator import add
from functools import reduce
from tqdm import tqdm

sns.set()
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample, shuffle

from data_preprocessing_pipeline import select_column
from plot_metrics import metrics, plot_cm, plot_pr_curve, plot_roc


from os.path import expanduser
home = expanduser("~")


from icecream import ic

LR_model = LogisticRegression()
LR_paragram_grid = [{'penalty': ['l1', 'l2'], 'C': [10 ** n for n in range(-4, 5)],
                     'solver': ['liblinear']},
                    {'penalty': ['none'], 'solver': ['newton-cg']}]

RF_model = RandomForestClassifier()
RF_paragram_grid = [{'n_estimators': [50, 100, 200], 'max_depth': [20, 50, None],
                     'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}]


LR_best_model = LogisticRegression(penalty='l2', C=1, solver='liblinear')
RF_best_model = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_leaf=1, min_samples_split=2,
                                       max_depth=50)
baseline_model = DummyClassifier(strategy='most_frequent')
baseline_paragram_grid=[]

#myDirectory = home+'/Timestamps_EHR_Deterioration_Predict/data/'
#mySampling = ["first"]
#myTime_of_day = [False, True]

def get_results(directory, input_data_frame_list, sampling, time_of_day, inAlgorithm=None,  inThreshold=None, calculate_CI=True):
    results = []
    #algorithm_list = ['Logistic_Regression', 'Random_Forest', 'Baseline']  #list of algorithms
    
    if inAlgorithm is None:
        algorithm_list = ['Logistic_Regression', 'Random_Forest', 'Baseline']  #list of algorithms
    else: 
        algorithm_list = [inAlgorithm]
    
    
    column = ['AUROC', 'AUPRC', 'sensibility', 'specificity', 'PPV', 'NPV', 'FScore'] ; # data that needs to be reported

    if inThreshold is None:
        thresholds = [0.6, 0.1, 0.5]
    else:
        thresholds = [inThreshold]
    #set the actual ml classifier map with the label
    algorithms = {'Logistic_Regression': LR_best_model, 'Random_Forest': RF_best_model, 'Baseline': baseline_model} 
    
    #set the search grid for each classifier
    paragram_grids = {'Logistic_Regression': LR_paragram_grid, 'Random_Forest': RF_paragram_grid, 'Baseline': baseline_paragram_grid}
    
    #match them up and go through each of the iteration
    for algorithm, threshold in zip(algorithm_list, thresholds):
        print (algorithm, threshold)
        for method in sampling: #each of the sampling method (first = counting the samples from admit time to predict outcome, 
                                #last = counting from the outcome time backwards
                                # random = either first or last. 
            for time in time_of_day: # to include or not include the time of day into prediction
                #build and search each classifier for the best one
                x = BuildAlgorithmsSinglePoint(directory=directory, input_data_frame_list=input_data_frame_list, sampling_method=method, timestep_length=60,
                                               algorithm=algorithms[algorithm], paragram_grid=paragram_grids[algorithm],
                                               time_of_day=time)
                #append the results / default is not to search for the best one (adjust later?) 
                results.append(list(x.get_results(search=False, threshold=threshold, calculate_CI=calculate_CI)))
            #return the results
    
    r_index = pd.MultiIndex.from_product([algorithm_list, sampling, time_of_day])
        
    result_df = pd.DataFrame(results, columns=column,index=r_index)
    
    return result_df


class BuildAlgorithmsSinglePoint(object):
    #_DIRECTORY = '/home/liheng/Mat/Source_code/dataset_sum/'

    _DIRECTORY = home+'/Timestamps_EHR_Deterioration_Predict/data/'

    def __init__(self, directory=None,input_data_frame_list=None, sampling_method='first', timestep_length=60, algorithm=None, paragram_grid=None,
                 vitals=True, v_order=True, med_order=True, comments=True, notes=True, time_of_day=True):
        self.__directory = directory or self._DIRECTORY
        self.__method = sampling_method
        self.__length = timestep_length
        self.__algorithm = algorithm
        self.__grid = paragram_grid
        self.__vitals = vitals
        self.__v_order = v_order
        self.__med_order = med_order
        self.__comments = comments
        self.__notes = notes
        self.__time_of_day = time_of_day
        self.__input_data_frame_list = input_data_frame_list
        
        
        # to search for the best model by searching or just by returning whatever that was last
    def get_best_model(self, search=False):
        # load the training data
        
        if len(self.__input_data_frame_list) != 6:
            point_train_data, _, train_label, _, _, _ = self._load_data(self.__method, self.__length)
        
        point_train_data = self.__input_data_frame_list[0]
        train_label =  self.__input_data_frame_list[2]
        
        
        point_train_data = self._standardize_numeric_col(point_train_data)
        # adjust for feature (feature selection is done manually)
        point_train_data = \
            self._feature_selection(ds=point_train_data, vitals=self.__vitals, v_order=self.__v_order,
                                    med_order=self.__med_order, comments=self.__comments,
                                    notes=self.__notes, time_of_day=self.__time_of_day)
    
        #oversample to make sure that the number of positive cases are same as negative cases
        point_train_data, train_label = self._oversampling(point_train_data, train_label)
        if search: #hyperparameter searching or not
            best_model = model_searching(self.__algorithm, self.__grid, point_train_data, train_label)
        else:
            best_model = model_training(self.__algorithm, point_train_data, train_label)
        return best_model #return the best model, by searching or by whatever that last fitted

    
    #now use the model with the testing data
    def get_results(self, search=False, threshold=0.5, calculate_CI=True):
            # load the test data
        
        if len(self.__input_data_frame_list) != 6:
            _, _, _, point_test_data, _, test_label = self._load_data(self.__method, self.__length)
        
        point_test_data = self.__input_data_frame_list[3]
        test_label =  self.__input_data_frame_list[5]
        
        point_test_data = self._standardize_numeric_col(point_test_data)
        #same features as training
        point_test_data = \
            self._feature_selection(ds=point_test_data, vitals=self.__vitals, v_order=self.__v_order,
                                    med_order=self.__med_order, comments=self.__comments,
                                    notes=self.__notes, time_of_day=self.__time_of_day)
        
        if calculate_CI: # if we wanted to calculate the confidence interval
            
            #first init the list of results
            AUROC_list, AUPRC_list, sensibility_list, specificity_list, PPV_list, NPV_list, FScore_list \
                = list(), list(), list(), list(), list(), list(), list()
            variables = [AUROC_list, AUPRC_list, sensibility_list, specificity_list, PPV_list, NPV_list, FScore_list]

            def append_list(variables, values):
                for var, val in zip(variables, values): #match the result list with ethe value list.
                    var.append(val)

            for i in tqdm(list(range(100))):
      
                best_model = self.get_best_model(search) # get the best model from training
                AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore \
                    = model_validation(best_model, point_test_data, test_label, threshold) # validate and return the values
            
            
                values = [AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore]
                append_list(variables, values) #append the values into the result list.
            AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore \
                = list(map(self._calculate_CI, variables)) # for each of the variables calculate CI
                        
        else:
            best_model = self.get_best_model(search) #if CI not wanted, just return the list of results
            AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore \
                = model_validation(best_model, point_test_data, test_label, threshold)
        print('AUROC:{}, AUPRC:{}, sensitivity:{}, specificity:{}, PPV:{}, NPV:{}, F-Score:{}'
              .format(AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore))
        
        return AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore

    #load the data from processing
    def _load_data(self, sampling_method, timestep_length):
        directory = self.__directory
        folder = sampling_method
        ic (folder)
        timestep_length = str(timestep_length)
        point_train_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'point_train_data.npy')
        train_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'train_data.npy')
        train_label = np.load(directory + folder + '/len' + timestep_length + '_' + 'train_label.npy')
        point_test_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'point_test_data.npy')
        test_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'test_data.npy')
        test_label = np.load(directory + folder + '/len' + timestep_length + '_' + 'test_label.npy')
        return point_train_data, train_data, train_label, point_test_data, test_data, test_label

    #standarize any numeric values into between 0-1
    def _standardize_numeric_col(self, ds):
        nm_ds = ds[:, :15] #exclude  the last column for now, it is the hour column. fake data has all the same hours. original 15
        mean = nm_ds.mean(axis=0)
        std = nm_ds.std(axis=0)
        diff = nm_ds-mean
        #standardized_ds = (nm_ds - mean) / std
        standardized_ds = np.divide(diff,std)
      
        standardized_ds = np.concatenate((standardized_ds, ds[:, 15:]), axis=1)
        return standardized_ds

    #turning features on or off (done manually in the code here)
    def _feature_selection(self, ds, vitals=True, v_order=False, med_order=False, comments=False,
                           notes=False, time_of_day=True):
        position = list(range(ds.shape[-1]))
        _VITALSIGNS = [0, 1, 2, 3, 4]
        _VITALORDERENTRY = [5, 6]
        _MEDORDERENTRY = [7, 8]
        _FLOAWSHEETCOMMENT = [9, 10, 11, 12, 13]
        _NOTES = [14]
        deposit = []

        if not time_of_day:
            position = position[:15]
        if not vitals:
            deposit += _VITALSIGNS
        if not v_order:
            deposit += _VITALORDERENTRY
        if not med_order:
            deposit += _MEDORDERENTRY
        if not comments:
            deposit += _FLOAWSHEETCOMMENT
        if not notes:
            deposit += _NOTES
        position = list(filter(lambda x: x not in deposit, position))
        return np.take(ds, position, axis=-1)
    

        
        
        
    #sameple to make sure pos cases are same count as neg cases
    def _oversampling(self, train_data, train_label):
        pos_data, pos_label = resample(train_data[train_label == 1], train_label[train_label == 1],
                                       n_samples=(train_label == 0).sum(), replace=True)
        neg_data = train_data[train_label == 0]
        neg_label = train_label[train_label == 0]
        new_data = np.concatenate((pos_data, neg_data))
        new_label = np.concatenate([pos_label, neg_label], axis=None)
        new_data, new_label = shuffle(new_data, new_label)
        return new_data, new_label

    #I guess this was used for troubleshooting? not referenced here
    def _column_name(self, freq):
        periods = int(1440 / freq)
        freq = str(freq) + 'T'
        col = select_column(base=False, vitals=True, v_order=True, med_order=True, comments=True, notes=True)
        col += ['Time_of_day_' + str(t)[7:] for t in pd.timedelta_range(0, periods=periods, freq=freq)]
        return col

    #calculate CI
    
    def _calculate_CI(self, value_list, alpha=0.95):
        p = (1-alpha)/2*100
        lower = np.percentile(value_list, p).round(3)
        upper = np.percentile(value_list, 100-p).round(3)
        mean = np.mean(value_list).round(3)
        
        #return '{} ({}, {})'.format(mean, lower, upper)
        #return mean ,lower,upper
        return mean 


# search for the best hyperparameters for the classifier
def model_searching(algorithm, gridsearch_dict, training_data, training_label):
    search = GridSearchCV(algorithm, gridsearch_dict, cv=4)
    search.fit(training_data, training_label)
    best_model = search.best_estimator_
    # print(pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score'))
    print('The best model hyperparameters: ', search.best_params_)
    return best_model


def model_training(best_model, training_data, training_label):
    best_model.fit(training_data, training_label)
    return best_model

#valdiate the model and plot the sample
def model_validation(model, test_data, true_label, threshold, plot=False):
    pred_label = model.predict_proba(test_data)[:, 1]
    if plot:
        sensibility, specificity, PPV, NPV = plot_cm(true_label, pred_label, p=threshold)
        AUPRC = plot_pr_curve('AUPRC', true_label, pred_label)
        AUROC = plot_roc('AUROC', true_label, pred_label)
        FScore = 2 / (sensibility ** -1 + PPV ** -1)
    else:
        AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore = metrics(true_label, pred_label, p=threshold)
    return AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore

#### Jacob's New area for running the code ####
#this is to execute all the stuff above
##with open("ml_output.txt","a") as o:
##    o.write(get_results(myDirectory, mySampling, myTime_of_day).to_string())
