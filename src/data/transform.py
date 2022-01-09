import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def parse_value_labels(data_dictionary: pd.DataFrame) -> dict:
    """Parses data dictionary labels."""
    vls = data_dictionary.loc[:, "Value labels (vls)"]
    t = []
    for x in vls:
        if x is np.nan:
            t.append(None)
        else:
            a = x.split(",")
            b = [s.replace('"', "").split(" = ") for s in a]
            d = {int(e[0]): e[1] for e in b}
            t.append(d)

    return {
        v: repl 
        for v, repl 
        in zip(data_dictionary.loc[:, "Variable name (vn)"], t)
        if repl is not None
    }


def label_categorical(X, cat_features): 
    le = LabelEncoder()
    X.loc[:, cat_features] = X.loc[:, cat_features].apply(le.fit_transform)
    return X