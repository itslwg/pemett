import pandas as pd

from typing import Optional
from src.visualization.visualize import create_sample_characteristics_table

def generate_sample_characteristics_table(outcome_variable: str,
                                          file_name: Optional[str] = None,
                                          tables_dir: str = "./reports/tables/"):
    """Prints the table 1 for input outcome variable.
    
    Wrapper to tidy manuscript notebook
    
    Args:
        outcome_variable: Name of the outcome variable. E.g. s30d.
        file_name: File to which the table should be saved.
        tables_dir: Directory in which the tables lay.
        
    Returns:
        Table 1 as pandas dataframe.
    """
    # Read relevant files
    data_dir = "./data/"
    
    cat_features = ['moi', 'sex', 'mot', 'tran', 'egcs', 'mgcs', 'vgcs', 'avpu']
    cont_features = ['age', 'hr', 'sbp', 'dbp', 'spo2', 'rr', 'delay']
    
    df = pd.read_csv(
        data_dir + "interim/table_sample_{ov}.csv".format(ov=outcome_variable),
        index_col=0)

    data_dictionary = pd.read_csv(
        data_dir + "raw/data_dictionary.csv",
        delimiter = ","
    )
    cat_features = cat_features + [outcome_variable, "tc"]
    t = create_sample_characteristics_table(
        df=df,
        data_dictionary=data_dictionary,
        categorical=cat_features,
        nonnormal=cont_features, 
        groupby="partition"
    )
    tto = t.tableone
    tto.columns = tto.columns.droplevel()
    ho = tto.loc[:, "Holdout"]
    tto = tto.drop(
        columns=["Holdout"], 
    ).assign(
        Holdout=ho
    )
    if file_name is not None: 
        tto.to_csv(tables_dir + file_name)

    return tto
    