import os
import pickle 

def dump_model(model, file_name, path=r"./models/"):
    
    with open(os.path.join(path, file_name), "wb") as file:
        pickle.dump(model, file)
        
def load_model(file_name, path=r"./models/"):
    
    with open(os.path.join(path, file_name), "rb") as file:
        model = pickle.load(file)
        
    return model