# Ryan McShane
# 03-08-2024

# Takes in data for preprocessing in preparation to train the MLM
# OR Takes in model.joblib file for existing model already trained

import etc

def kdd_preprocessing(data):
    # Get path to kdd.names
    kdd_10percent = etc.trim(etc.search_for_file_path('kddcup.data_10_percent.gz'))
    if data == 10:
        return kdd_10percent
    else:
        kdd_names_path = etc.trim(etc.search_for_file_path('kdd.names')) #
        #kdd_names_path = 'F:/senior-project-2024-group-1/kdd.names' #Replace <- with ^^statement 
        print ("\nkdd_names_path = ", kdd_names_path)

        # Read list of features
        with open(kdd_names_path,'r') as kdd_names:
            print(kdd_names.read())

        # print(addCols.add_cols()) ## Debugging appending target data labels

        # Get path to training_attack_types
        training_attacks_path = etc.trim(etc.search_for_file_path('training_attack_types'))
        #training_attacks_path = 'F:/senior-project-2024-group-1/training_attack_types.txt' #Replace <- with ^^statement 
        print ("\ntraining_attacks_types = ", training_attacks_path)
        with open(training_attacks_path,'r') as training_attack_types:
            print(training_attack_types.read())
        # Reading Dataset
        return kdd_10percent
            
        