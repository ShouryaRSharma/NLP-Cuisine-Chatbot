from .stop_words    import STOP_WORDS
from .cuisines_food import AMERICAN, EUROPEAN, ASIAN, DISHES
from .tags          import FARWELL, GREETING, REJECT, CONFIRM, CANCEL, UPDATE, ORDER
from .responds      import FARWELLS, ORDERING, REJECTIONS, CONFIRMATIONS, CANCELS, UPDATES, FOLLOWUPS, STUPID, APPROVAL

import os
import nltk

# Necessary libraries for the lemmatization. 
paths = [r"taggers/averaged_perceptron_tagger", r"corpora/omw-1.4", r"corpora/wordnet"]

for path in paths:
    try:
        nltk.data.find(path)
    except LookupError:   
        nltk.download(os.path.basename(path))
    
from .pos_tags import wordnet_POS_conversion