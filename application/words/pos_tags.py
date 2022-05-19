from nltk.corpus import wordnet

def wordnet_POS_conversion(tag):
    
    tags = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
        }
    
    return tags.get(tag[0], wordnet.NOUN)