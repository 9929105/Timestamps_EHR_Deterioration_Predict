
import os, os.path
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from icecream import ic
from data_preprocessing_pipeline import DataPipeline
import modeling_pipeline_single_point as ml
from os.path import expanduser

def append_list(variables, values):
     #match the result list with ethe value list.
    for var, val in zip(variables, values):
        var.append(val)  

def calculate_CI(value_list, alpha=0.95):
    p = (1-alpha)/2*100
    lower = np.percentile(value_list, p).round(3)
    upper = np.percentile(value_list, 100-p).round(3)
    mean = np.mean(value_list).round(3)
        
    return '{} ({}, {})'.format(mean, lower, upper)
    #return (mean,lower,upper)

home = expanduser(".")
iteration_cnt = 100;

#myfile_name = home+'/../data/one_iteration.csv'
myfile_name = home+'/../data/all_nurse_data.csv'
icu_df = pd.read_csv(myfile_name)

with open("ml_output.txt","w") as o:
    now = datetime.datetime.now()
    o.write(" --- execute at ---" + now.strftime("%Y-%m-%d %H:%M:%S")+"\r\n")


def log_progress(inStr, opt):
    with open("ml_output.log",opt) as o:
        now = datetime.datetime.now()
        o.write(now.strftime("%H:%M:%S")+"---- ----"+inStr+"\r\n")
        

log_progress(" -- Begin Execution --- ","w")
# split dataset into control and outcome set
# get the unique encntr ids 

## get all unique IDs
control_list = icu_df[icu_df['outcome'] == 0]['dummy_encounter_id'].unique() 
outcome_list = icu_df[icu_df['outcome'] == 1]['dummy_encounter_id'].unique() #161


def load_data():
    
    ##samples
    #tmp_control_list = np.random.choice(control_list, size = 6559, replace=False) #6559
    #tmp_outcome_list = np.random.choice(outcome_list, size = 161, replace=False) #161

    #pull all the data rows for those encoutner ids under the pos or neg cases
    #tmp_control_cohort = icu_df[icu_df['dummy_encounter_id'].isin(tmp_control_list)]
    #tmp_outcome_cohort = icu_df[icu_df['dummy_encounter_id'].isin(tmp_outcome_list)]


   # tmp_df = pd.concat([tmp_control_cohort, tmp_outcome_cohort])
   
    toReturn = shuffle(icu_df)
    
    return toReturn


algorithms = ['Logistic_Regression', 'Random_Forest'] 
thresholds = [0.6, 0.1, 0.5]
#foo = DataPipeline(starttime='first', data_frame = tmp_df)
column = ['AUROC', 'AUPRC', 'sensibility', 'specificity', 'PPV', 'NPV', 'FScore'] ; # data that needs to be reported
    
    #myDirectory = home+'/../data/'
myDirectory = None
mySampling = ["first"]
myTime_of_day = [True]

AUROC_list, AUPRC_list, sensibility_list, specificity_list, PPV_list, NPV_list, FScore_list \
        = list(), list(), list(), list(), list(), list(), list()
variables = [AUROC_list, AUPRC_list, sensibility_list, specificity_list, PPV_list, NPV_list, FScore_list]


def single_process(al, td):
    iteration_df = load_data()
    foo = DataPipeline(starttime='first', data_frame = iteration_df)


    point_train_data, train_data, train_label, point_test_data, test_data, test_label=foo.get_results()


    in_data_frame_list = [point_train_data, train_data, train_label, point_test_data, test_data, test_label]
    #foo.save_array(point_train_data, train_data, train_label, point_test_data, test_data, test_label)

    #for al, td in zip(algorithms, thresholds):
  


    result_df = ml.get_results(directory = mySampling, input_data_frame_list=in_data_frame_list, \
                                    sampling=mySampling,time_of_day=myTime_of_day, inAlgorithm = al, inThreshold = td )
    AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore = result_df.iloc[0]
    values = [AUROC, AUPRC, sensitivity, specificity, PPV, NPV, FScore]

    append_list(variables, values) #append the values into the result list.


#endSINGLE FUNCTION





for al,td in zip(algorithms,thresholds): 
    
    for i in range(iteration_cnt):
        
        log_progress(" -- --- "+al+ "---- itera = "+str(i), "a")
        single_process(al=al,td=td)
    
    
    total_raw=pd.DataFrame(data=range(iteration_cnt), columns = ["iteration"])
    
    for v,c in zip(variables,column):
        one_column=pd.DataFrame(data=v,columns=[c])
        total_raw=total_raw.join(one_column)
    
    now = datetime.datetime.now()
    
    results = np.array(list(map(calculate_CI, variables))) # for each of the variables calculate CI

    result_df = pd.DataFrame(data=results)
    result_df = result_df.T
    result_df.columns = column


    with open("ml_output.txt","a") as o:
        o.write(" \r\n\\r\n  --- --------- Algorithm =    "+al+ "       ------ \r\n")
        o.write(" --- raw data  ---" + now.strftime("%Y-%m-%d %H:%M:%S")+"\r\n")
        o.write(total_raw.to_string()+"\r\n")
        o.write(" --- Summary --- ---\r\n")
        o.write(result_df.to_string())



        
log_progress(" -- -------------------- WORK COMPLETE -------------------","a")
        