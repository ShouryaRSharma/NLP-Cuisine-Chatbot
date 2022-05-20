import os
import json
import nltk
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import random
import logging

from copy import deepcopy
from collections import namedtuple
from utility_methods import dump_model
from words import wordnet_POS_conversion , AMERICAN, EUROPEAN, ASIAN, STOP_WORDS, FARWELL, GREETING, REJECT, CONFIRM, CANCEL, UPDATE, ORDER, order, greetingOrder, changeOrder, reviseOrder
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


Cuisine = namedtuple("cuisine", "name foodtypes")

american  = Cuisine("American", AMERICAN)
european  = Cuisine("European", EUROPEAN)
asian     = Cuisine("Asian",    ASIAN)

CUISINES_ITEMS = [american, european, asian] 
CUISINES_STR = ["American", "European", "Asian"]

cuisine_label = {"American":0, "European":1, "Asian":2} 


def load_json(path=r"", file_name=""):
    """ 
    Loads a json dataset into a python object.
    For the food-ordering dataset, it returns a list of conversation dictionaries.
    
    Parameters:
    path      (string): the path for the file
    file_name (string): the file name
    
    Returns:
    data      (object): converted json to python object    
    """
    path = os.path.join(path, file_name)      
    
    try:
        with open(path,"r") as read_file:
            data = json.load(read_file)
    except FileNotFoundError:
        print(f"Path: '{path}' does not exist.")
    else:
        return data


def annotations_seperated(segments): 
    """ 
    Formats the segment into a dictionary with a new annotation type key 
    per annotated text, for a single utterance.
    
    Parameters:
    segments         (list): a list of annotations for a single utterance
    
    Returns:
    annotations (dictionary): a dicionary of annotation types, mapped to the annoted text
    """
    annotations = {}
    
    for indx in segments:
        annotation_type = indx["annotations"][0]["name"].removeprefix("food_order.")
        annotated_text = indx["text"]
        
        count = 1
        while f"{annotation_type}_{count}" in annotations:
            count += 1   
            
        # Update the dictionary with the next count index if there 
        # are more than one of the same annotation type per utterance.
        annotations[f"{annotation_type}_{count}"] = annotated_text
        
    return annotations

    
def annotations_merged(segments, delim): 
    """ 
    Formats the segment into a dictionary with a single annotation type key 
    for multiple annotated texts, for a single utterance.
    
    Parameters:
    segments         (list): a list of annotations for a single utterance
    delim          (string): choice of delimiter between the merged annotations
    
    Returns:
    annotations (dictionary): a dicionary of annotation types, mapped to the annoted text
    """
    annotations = {}
    
    for indx in segments:
        annotation_type = indx["annotations"][0]["name"].removeprefix("food_order.")
        annotated_text = indx["text"]
        
        # Join the annotated texts, under the same annotation type, with ;.
        previous_annotated_text = annotations.get(annotation_type, None)
        if previous_annotated_text:
            annotations[annotation_type] = delim.join([previous_annotated_text, annotated_text])
        else:
            annotations[annotation_type] = annotated_text
       
    return annotations
    
    
def annotated_conversation(indx, dataset, delim=None):
    """ 
    Takes an conversation dictionary, and updates it to desired format,
    including keys such as:
        -index 
        -speaker 
        -text 
        (-annotations) by annotation type 
    
    Parameters:
    indx               (integer): the index of the conversation
    dataset               (list): the list of conversations
    delim               (string): if the annotations should be under one key per type
    
    Returns:
    conversation    (dictionary): a list of utterances forming a conversation
    """
    conversation = []
    
    for utterance in dataset[indx]["utterances"]:
        utterance = deepcopy(utterance)
        
        # Remove the segements list of annotations, to update with a
        # dictionary mapping of annotations as part of each utterance.
        segments = utterance.pop("segments", None)     
        if segments:
            
            if delim:
                annotations = annotations_merged(segments, delim)
            else:
                annotations = annotations_seperated(segments)   
                
            utterance.update(annotations)
                
        conversation.append(utterance)
        
    return conversation


def conversation_table(conversation, annotations_keep=[], columns_drop=[]):
    """ 
    Converts the annotated conversation dicitionary into a dataframe.
    
    Parameters:
    conversation     (dictionary): the annotated conversation dictionary
    annotations_keep       (list): annotation types to display, if empty include all
    columns_drop           (list): columns not to display
    
    Returns:
    df                (dataframe): dataframe of the conversation   
    """         
    df = pd.DataFrame(conversation)   
    annotation_names = []
    
    if annotations_keep:
        # Only traverse through the annotation columns.
        for column in df.columns[3:]:

            for annotation in annotations_keep:
                # Startswith due to seperated annotation instances may occur.
                if column.startswith(annotation):
                    annotation_names.append(column)
    else:
        annotation_names.extend(df.columns[3:].tolist())

    df = df[["index", "speaker", "text", *sorted(annotation_names)]]
    
    # Ignore if column name does not exist.
    for column in columns_drop:
        try:
            df = df.drop(columns=column)
        except:
            pass
               
    return df

def conversation(dataset, indx, delim=None, annotations_keep=[], columns_drop=[]):
    
    conversation = annotated_conversation(indx=indx, dataset=dataset, delim=delim)
    df = conversation_table(conversation, annotations_keep=annotations_keep, columns_drop=columns_drop)
    
    return df


def food_type_table(dataset, delim=" "):
    
    conversation_amount = len(dataset)
    food_type_utterances = []
    
    for indx in range(conversation_amount):

        df = conversation(dataset, indx, delim=delim)     
        
        if all(x in df for x in ["type.food", "name.item"]):
            
            # Utterances from the user, that include annotated food items.
            food_items = df[(df["speaker"] == "USER") & (df["name.item"].notna())][["text"]]
            
            # Annotated food types for the entire conversation, merged into a single string.
            food_type = df[df["type.food"].notna()][["type.food"]].transpose()
            food_type = delim.join(food_type.values.tolist()[0])
            food_items["food_type"] = food_type
                
            food_items["conversation_indx"] = indx 
            
            # Append the necessary utterances mapped to its corresponding food types from 
            # this conversation, to the list of necessary utterances accross all the conversations.
            food_type_utterances.append(food_items)

    return pd.concat(food_type_utterances)

def dataset_apply(dataset, func, column_to, column_from=None):
    
    if not column_from:
        column_from = column_to
        
    dataset[column_to] = dataset[column_from].apply(func)
    

def assign_cuisine(dataset):
  
    dataset = deepcopy(dataset)
    
    # Normalize and Tokenize the food type column
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"[a-zA-Z]+")
    normalize_tokenize = vectorizer.build_analyzer()
    dataset_apply(dataset, func=normalize_tokenize, column_to="food_type", column_from="food_type")
    
    # Create cuisine column based on food types of that region
    cuisine_assignment = lambda foods: [cuisine.name for cuisine in CUISINES_ITEMS
                                        for food in foods 
                                        if any(cuisine_food in food for cuisine_food in cuisine.foodtypes)]
    dataset_apply(dataset, func=cuisine_assignment, column_to="cuisine", column_from="food_type")
    
    # If there a match against food types assign the first cuisine from list, otherwise NaN
    check_NaN = lambda cuisines: cuisines[0] if len(cuisines) != 0 else np.NaN
    dataset_apply(dataset, func=check_NaN, column_to="cuisine", column_from="cuisine")
    
    # Apply integer labels for corresponding cuisine
    apply_labels = lambda cuisine: cuisine_label[cuisine] if cuisine is not np.NaN else np.NaN
    dataset_apply(dataset, func=apply_labels, column_to="label", column_from="cuisine")
 
    # Only keep the text, cuisine, and label columns and reset index
    dataset = dataset[dataset["cuisine"].notna()][["text", "cuisine", "label"]].reset_index(drop=True) 
    dataset = dataset.astype({"text": str, "cuisine": str, "label": int})
    
    return dataset


def drop_random_rows(dataset, column, label, amount):
    
    # Drop an amount of random indices among corresponding label
    random_generator = np.random.default_rng()
    indx = dataset[dataset[column] == label].index
    dataset = dataset.drop(index=random_generator.choice(indx, size=amount, replace=False))
    
    return dataset
    
def balance_dataset(dataset, column, min_size=None):

    dataset_balanced = deepcopy(dataset)

    if not min_size:
        # Calculate label with the least amount of utterances
        min_size = min(dataset[column].value_counts().values)

    # Calculate the amount to reduce from the labels depending on the min value
    for label, size in dataset[column].value_counts().iteritems():
        amount = size - min_size
        dataset_balanced = drop_random_rows(dataset_balanced, column, label, amount)
    
    return dataset_balanced


class TextPreparationVectorizer(CountVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"[a-zA-Z]+",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
        normalization=None,
    ):
        super(TextPreparationVectorizer, self).__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.normalization = normalization
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        
    def build_analyzer(self):
        analyzer = super(TextPreparationVectorizer, self).build_analyzer()
        
        if self.normalization == "stem":
            return lambda doc: ([self.stemmer.stem(token) for token in analyzer(doc)])
    
        if self.normalization == "lemmatize":
            return lambda doc: [self.lemmatizer.lemmatize(token, wordnet_POS_conversion(tag)) for token, tag in nltk.pos_tag(analyzer(doc))]
        
        return analyzer


class NormalizationScaler(MinMaxScaler):
    def fit(self, X, y=None, sample_weight=None):
        if sp.issparse(X):
            X = X.toarray()
        return super(NormalizationScaler, self).fit(X)
      
    def transform(self, X, copy=None):
        if sp.issparse(X):
            X = X.toarray()
        return super(NormalizationScaler, self).transform(X)


class TextPreparation(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"[a-zA-Z]+",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
        normalization=None,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.normalization = normalization
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        global TextPreparationVectorizer
        self.vectorizer = TextPreparationVectorizer(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            normalization=normalization,
        )
        self.vector_analyzer =  None

    def fit(self, raw_documents, y=None):
        
        self.vector_analyzer =  self.vectorizer.build_analyzer()
        return self

    def transform(self, raw_documents):
        return [self.vector_analyzer(doc) for doc in raw_documents]


class SubTagClassifier(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        Tag = namedtuple("tag", "name tags conditions")
    
        # Either farewell or greeting or none of them, farewell has precedence 
        self.farewell  = Tag("farewell", FARWELL, [])
        self.greeting  = Tag("greeting", GREETING, [self.farewell])

        # Either reject or confirm or none of them, reject has precedence
        self.reject   = Tag("reject", REJECT, [])
        self.confirm  = Tag("confirm", CONFIRM, [self.reject])

        # Either cancel, update or order or none of them, cancel, update, order is the precedence order
        self.cancel  = Tag("cancel", CANCEL, [])
        self.update  = Tag("update", UPDATE, [self.cancel])
        self.order   = Tag("order", ORDER, [self.cancel, self.update])
        
        self.tag_categories = [self.farewell, self.greeting,
                               self.reject, self.confirm,
                               self.cancel, self.update, self.order]
        
    def __sub_tag_classifer(self, utterance, tag):
        
        # Check if possible tag strings are included in utterance, as well as making sure there is no conflict between tag types
        if any(tag_str in utterance for tag_str in tag.tags) and all(tag_c.name not in self.sub_tags for tag_c in tag.conditions):
            self.sub_tags.append(tag.name) 

    def fit(self, doc, y=None):
        return self

    def predict(self, doc):
        
        # Iterate through all tokenized utteranaces from list
        self.all_sub_tags = []
        for utterance in doc:
            
            # Create individual sub-tag list for each individual utterance 
            self.sub_tags =  []
            for tag in self.tag_categories:                
                self.__sub_tag_classifer(utterance, tag)
                
            # Save the individual list in a combined list for the entire doc
            self.all_sub_tags.append(self.sub_tags)
                
        return self.all_sub_tags


sub_tag_classifier = Pipeline(steps=[
                ("process", TextPreparation(normalization="lemmatize")),
                ("class", SubTagClassifier()), 
])

SUB_TAG_CLASSIFIER = sub_tag_classifier.fit("")


# TRAINING OF MODEL TO BE PICKLED
##########################################################################################################################

def train_cuisine_classifier():

    ic_dataset = load_json(path=r"./data/", file_name="food-ordering.json")
    df_food_type = food_type_table(ic_dataset, delim=";")
    df_unbalanced = assign_cuisine(df_food_type)
    ic_data = balance_dataset(df_unbalanced, "label", min_size=280)

    train, _ = train_test_split(ic_data, test_size=0.2, stratify = ic_data["label"], random_state=0)
    train_data = (train["text"].to_list(), train["label"].to_list())

    cuisine_classifier = Pipeline(steps=[
                    ("selct", TextPreparationVectorizer(stop_words=STOP_WORDS, normalization="stem", ngram_range=(1,1))),
                    ("extct", TfidfTransformer()), 
                    ("scale", NormalizationScaler()), 
                    ("class", MLPClassifier(solver="adam", activation="relu", hidden_layer_sizes=(100,100), random_state=0, max_iter=600)),
    ])

    cuisine_classifier.fit(*train_data)

    return cuisine_classifier

# helper function for deciding what dictionary and subsequent array to append the food sentence on to
def add_data(train, test, key):
    FOOD_SENTENCE_LIMIT = 85
    return train[key] if len(train[key]) < FOOD_SENTENCE_LIMIT else test[key]

def generateDataset():
    # read in the food csv file
    food_dataset = pd.read_csv("./data/food.csv")
    # diaqualify foods with special characters, lowercase and extract results from "description" column
    foods = food_dataset[food_dataset["description"].str.contains("[^a-zA-Z ]") == False]["description"].apply(lambda food: food.lower())

   
    foods = foods[foods["description"].str.contains("[^a-zA-Z ]") == False]["description"].apply(lambda food: food.lower())
    
    # filter out foods with more than 2 words, drop any duplicates
    foods = foods[foods.str.split().apply(len) <= 2].drop_duplicates()
    one_worded_foods = foods[foods.str.split().apply(len) == 1]
    two_worded_foods = foods[foods.str.split().apply(len) == 2]
    total_num_foods = round(one_worded_foods.size / 45 * 100)

    # shuffle the 2-worded foods since we'll be slicing them
    two_worded_foods = two_worded_foods.sample(frac=1)
    # append the foods together 
    foods = one_worded_foods.append(two_worded_foods[:round(total_num_foods * 0.30)])

    train_data = {
        "order": [],
        "greetingOrder": [],
        "changeOrder": [],
        "reviseOrder": []
    }

    test_data = {
        "order": [],
        "greetingOrder": [],
        "changeOrder": [],
        "reviseOrder": []
    }
    # the pattern to replace from the template sentences
    pattern_to_replace = "{}"

    # shuffle the data before starting
    foods = foods.sample(frac=1)

    # the count that helps us decide when to break from the for loop
    food_entity_count = foods.size - 1

    # start the while loop, ensure we don't get an index out of bounds error
    while food_entity_count >= 2:
        # pick a random food template
        sentence = order[random.randint(0, len(order) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)

        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            food = foods.iloc[food_entity_count]
            food_entity_count -= 1

            # replace the pattern, but then find the match of the food entity we just inserted
            sentence = sentence.replace(match, food, 1)
        # append the sentence and the position of the entities to the correct dictionary and array
        add_data(train_data, test_data, "order").append([sentence, (0, 0, 0, 0, 0, 0, 1)])

        # pick a random food template
        sentence = greetingOrder[random.randint(0, len(greetingOrder) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)
        # print(matches)
        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            food = foods.iloc[food_entity_count]
            food_entity_count -= 1

            # replace the pattern, but then find the match of the food entity we just inserted
            sentence = sentence.replace(match, food, 1)
        add_data(train_data, test_data, "greetingOrder").append([sentence, (0, 1, 0, 0, 0, 0, 1)])

        # pick a random food template
        sentence = changeOrder[random.randint(0, len(changeOrder) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)
        # print(len(matches))
        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            food = foods.iloc[food_entity_count]
            food_entity_count -= 1

            # replace the pattern, but then find the match of the food entity we just inserted
            sentence = sentence.replace(match, food, 1)
        add_data(train_data, test_data, "changeOrder").append([sentence, (0, 0, 0, 0, 0, 0, 1)])

        # pick a random food template
        sentence = reviseOrder[random.randint(0, len(changeOrder) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)
        if matches:
            for match in matches:
                food = foods.iloc[food_entity_count]
                food_entity_count -= 1

                # replace the pattern, but then find the match of the food entity we just inserted
                sentence = sentence.replace(match, food, 1)
            add_data(train_data, test_data, "reviseOrder").append([sentence, (0, 0, 0, 0, 0, 1, 0)])
        else:
            add_data(train_data, test_data, "reviseOrder").append([sentence, (0, 0, 0, 0, 0, 1, 0)])
        
    trainData = train_data['order'] + train_data['greetingOrder'] + train_data['changeOrder'] + train_data['reviseOrder']
    greetingResponses = ["Hi", "Hello", "Greetings!", "Hey!", "Hola", "Hi there", "Hey there!", "Hello!", "Hiyah!" "Hello", "Hi", "Hey", "Hi there", "Hello can I get some help", "Hey there", "Hello"]
    farewellResponses = ["Thanks so much! Bye.", "Thanks for the help.", "Bye bye.", "Goodbye.","Cheers for the help.","Great thanks.","Bye","That's all I want thanks.", "No that's everything thanks."]
    cancelResponses = [
        "I'd like to cancel the order sorry",
        "I want to remove everything",
        "I want to scap everything ",
        "Can we get rid of my order please",
        "I'd like to drop it actually",
        "Let's just scrap that and get rid of everything",
        "I want to cancel my order",
        "I want to scrap my order",
        "I want to get rid of everything",
        "Can I cancel my order please",
    ]
    confirmResponses = ["Yes.", "Yeah", "Okay", "Yes please that works", "Sure okay yeah", "That would be great", "Sure", "Right yeah",
                        "Okay yes", "I'd like that thanks", "Yeah that works for me", "Absolutely", "Thanks yeah", "Okay great", "Perfect",
                        "Sounds good", "Sounds great to me", "Yeah that would be perfect thanks", "I can confirm that yeah", "Sure that works"]
    rejectionResponses = ["No thanks actually", "No", "I'm okay actually", "I'm good thanks", "Nope", "I think I'm alright", 
                        "I don't need anything else", "That's all thanks", "I don't want anything else", "That's fine actually",
                        "No I don't need anything else"]
    for i in range(30):
        greeting = random.choice(greetingResponses)
        farewell = random.choice(farewellResponses)
        cancel = random.choice(cancelResponses)
        confirm = random.choice(confirmResponses)
        rejection = random.choice(rejectionResponses)
        trainData.append([greeting, (0, 1, 0, 0, 0, 0, 0)])
        trainData.append([farewell, (1, 0, 0, 0, 0, 0, 0)])
        trainData.append([cancel, (0, 0, 0, 0, 1, 0, 0)])
        trainData.append([confirm, (0, 0, 0, 1, 0, 0, 0)])
        trainData.append([rejection, (0, 0, 1, 0, 0, 0, 0)])
    return trainData

def train_intent_classifier():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_df = pd.DataFrame(generateDataset())
    train_df.columns = ["text", "labels"]
    train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=42)
    # Preparing eval data

    # Optional model configuration
    model_args = MultiLabelClassificationArgs(num_train_epochs=5)
    model_args.overwrite_output_dir = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.silent = True
    # Create a ClassificationModel
    model = MultiLabelClassificationModel(
        'roberta',
        'roberta-base',
        num_labels=7,
        args=model_args,
        use_cuda=True,
        
    ) 
    model.train_model(train_df, output_dir="actual")
    return model
if __name__ == "__main__":

    intent_classifier = train_intent_classifier()
    dump_model(intent_classifier, "intent_classifier.pickle", path=r"./models/")
    cuisine_classifier = train_cuisine_classifier()
    dump_model(cuisine_classifier, "cuisine_classifier.pickle", path=r"./models/")
