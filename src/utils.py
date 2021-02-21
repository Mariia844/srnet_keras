from datetime import datetime

def save_model(model):
    model_path_invalid = f"model_{datetime.now()}"
    model_path = make_valid_path(model_path_invalid)
    model.save(f"{model_path}.hdf5")

def make_valid_path(path):
    model_path = ""
    for char in path:
        if (char.isalnum()):
            model_path += char
        else:
            model_path += '_'
    return model_path