import os
import pandas as pd

from utility_methods import dump_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from simpletransformers.ner import NERModel, NERArgs


column_names =["sentence_id", "words", "labels"]

ENTITIES = ["AMENITY", "CUISINE", "DISH", "HOURS", "LOCATION", "PRICE", "RATING", "RESTAURANT_NAME"]

# Clean up data by switching 'columns' to follow CoNLL 2003 NER format (word | tag) and write to new file for easier use.
def clean_data(file_name_old, file_name_new, path=r"./data/"):
    
    with open(os.path.join(path, file_name_old), "r") as old:    
        with open(os.path.join(path, file_name_new), "w") as new:   
            
            for line in old.readlines():
                if line == "\n":
                    new.write(line)
                else:
                    new.write("\t".join(line.strip().split("\t")[::-1]) + "\n")


def load_data_file(file_name, path=r"./data/"):
    
    word_data  = []
    label_data = []
    
    temp_data  = []
    temp_label = []
    
    with open(os.path.join(path, file_name), "r") as file: 
        
        for line in file.readlines():
            
            if line == "\n":
                word_data.append(temp_data)
                label_data.append(temp_label)
                
                # Reset temporary lists
                temp_data  = []
                temp_label = []
                
            else:
                temp_data.append(line.strip().split("\t")[0])
                temp_label.append(line.strip().split("\t")[1])
    
    return word_data, label_data


def load_dataframe(word_data, label_data):
    
    data = []
    
    for word_list, label_list in zip(word_data, label_data):
        for word, label in zip(word_list, label_list):
            
            # Note: Value for 'sentence_id' (dfColumnNames[0]) is wordData.index(listOfWords) so that words from the same sentence are grouped together
            data.append({column_names[0]: word_data.index(word_list),
                         column_names[1]: word, 
                         column_names[2]: label.upper()})
                       
    return pd.DataFrame(data)


def create_dataframe(data, labels):
    
    return pd.DataFrame({column_names[0]: data[column_names[0]],
                         column_names[1]: data[column_names[1]],
                         column_names[2]: labels})


# TRAINING OF MODEL TO BE PICKLED
##########################################################################################################################

def train_named_entity_recognitior():

    # %conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    # import torch
    # torch.cuda.is_available()

    # Base/Core NER dataset - found here: https://groups.csail.mit.edu/sls/downloads/restaurant/
    ner_train_old_name = "restauranttrain.bio"
    ner_test_old_name  = "restauranttest.bio"
    ner_train_new_name = "restauranttrain_cleaned.bio"
    ner_test_new_name  = "restauranttest_cleaned.bio"

    clean_data(ner_train_old_name, ner_train_new_name, path=r"./data/")
    clean_data(ner_test_old_name, ner_test_new_name, path=r"./data/")

    # Load word and label data from training and testing files
    train_word_data, train_label_data = load_data_file(ner_train_new_name, path=r"./data/")
    test_word_data,  test_label_data  = load_data_file(ner_test_new_name, path=r"./data/")

    # Combine train and test data so I can later split it to experiment with different splits when training model
    main_word_data  = train_word_data  + test_word_data
    main_label_data = train_label_data + test_label_data

    # Load formatted data into data frames to use in training and evaluating model
    main_data = load_dataframe(main_word_data, main_label_data)

    # List of lables/classes within my dataset.
    label_classes = list(main_data["labels"].unique())

    # Comment out for quicker training.
    # main_data = main_data.head(1000)

    # Separate dataset into data (stX) and labels (stY)
    stX = main_data[[column_names[0], column_names[1]]]
    stY = main_data[column_names[2]]

    # Split data into train and test sets using sklearn train_test_split
    x_train, x_test, y_train, y_test = train_test_split(stX, stY, test_size=0.3)

    # Create training and testing data frames
    st_train_data = create_dataframe(x_train, y_train)
    st_test_data  = create_dataframe(x_test, y_test)

    # NER Model Arguments
    batch_size = 32

    modelArgs = NERArgs()
    modelArgs.num_train_epochs = 2
    modelArgs.learning_rate = 1e-4
    modelArgs.weight_decay = 0.01
    modelArgs.optimizer = "AdamW"
    modelArgs.train_batch_size = batch_size
    modelArgs.eval_batch_size = batch_size
    modelArgs.overwrite_output_dir = True
    modelArgs.use_multiprocessing = False
    modelArgs.use_multiprocessing_for_evaluation = False
    modelArgs.process_count = 1
    modelArgs.silent = True

    named_entity_recognitior = NERModel(model_type="distilbert", model_name="distilbert-base-cased", labels=label_classes, args=modelArgs, use_cuda=True)
    named_entity_recognitior.train_model(train_data=st_train_data, eval_data=st_test_data, acc=accuracy_score)

    return named_entity_recognitior

    
if __name__ == "__main__":

    named_entity_recognitior = train_named_entity_recognitior()
    dump_model(named_entity_recognitior, "named_entity_recognitior.pickle", path=r"./models/")