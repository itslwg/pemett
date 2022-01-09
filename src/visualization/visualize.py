from plotnine import *

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)

from tableone import TableOne

from src.data.transform import parse_value_labels


def create_sample_characteristics_table(df: pd.DataFrame,
                                        data_dictionary: pd.DataFrame,
                                        **kwargs) -> TableOne:
    """Generates sample characteristics table
    
    Args:
        df: Full sample.
        kwargs: Key-word arguments for tableone.TableOne.
        
    Returns:
        The summary table.
    """
    # Get variable value labels
    vvls = parse_value_labels(data_dictionary)
    df.replace(vvls, inplace = True)
    
    # Better labels for all variables
    vns = data_dictionary.loc[:, "Variable name (vn)"]
    ls = data_dictionary.loc[:, "Label (l)"]
    labels = {vn: l for vn, l in zip(vns, ls)}

    return TableOne(
        df,
        rename=labels,
        **kwargs
    )


def plot_roc(y_true, y_pred):
    """Plots ROC using plotnine."""
    r = roc_curve(y_true = y_true, y_score = 1 - y_pred)
    plot_data = pd.DataFrame(
        np.column_stack(r),
        columns=["tpr", "fpr", "thresholds"]
    )
    p = (
        ggplot(plot_data, aes(x="fpr", y="tpr")) +
        geom_line() + 
        xlab("False Positive Rate") + 
        ylab("True Positive Rate")
    )
    return p


def plot_roc_base_clfs(y_preds: np.ndarray, y_true: np.array, keys: list):
    """Plots ROC for each base classifier."""
    def f(p, y_true):
        fpr, tpr, _ = roc_curve(y_true = y_true, y_score = p)
        a = np.column_stack((fpr, tpr))
        df = pd.DataFrame(a, columns=["fpr", "tpr"])
        return df

    df = pd.DataFrame(y_preds)
    dfs = [f(df[l], y_true = y_true) for l in df.columns]
    df_plt = pd.concat(dfs, keys=keys).reset_index(level=0)
    plt = (
        ggplot(df_plt, aes(x = "fpr",y = "tpr", color = "level_0")) +
        geom_line() + 
        xlab("False Positive Rate") + 
        ylab("True Positive Rate") +
        theme(legend_title=element_blank())
    )

    return plt


def plot_precision_recall(y_true, y_pred):
    """Plots precision and recall using plotnine."""
    prec, rec, thres = precision_recall_curve(y_true = y_true, probas_pred = 1 - y_pred)
    plot_data = pd.DataFrame(
        np.column_stack((prec, rec)),
        columns=["precision", "recall"]
    )
    p = (
        ggplot(plot_data, aes(x="precision", y="recall")) +
        geom_line() + 
        xlab("Precision") + 
        ylab("Recall")
    )
    return p


def plot_category_vs_outcome(y: pd.Series, y_prob_cut: np.array, tc: pd.Series):

    ## Replace numbers with nice names
    categories = ["Green", "Yellow", "Orange", "Red"]
    
    plt_df = pd.DataFrame(
        np.column_stack((y_prob_cut, tc, y)),
        columns = ["Model Category", "Triage Category", "target"]
    )
    plt_df_melt = pd.melt(
        plt_df, 
        value_vars = ["Model Category", "Triage Category"],
        id_vars = "target"
    )
    return (
        ggplot(plt_df_melt, aes("value", fill="factor(target)")) + 
        geom_bar(stat = "count") + 
        facet_wrap("variable") + 
        xlab("Category") + 
        scale_fill_discrete(name = "Dead in 30 days")
    )


def prepare_triage_plot_df(df_model):
    """Helper to create dataframe for plotting survivial within tc."""
    a = (
        df_model.groupby(["Label", "Category", "Target"])
                .size()
                .unstack()
                .reset_index()
                .melt(id_vars = ["Label", "Category"], value_vars=["0.0", "1.0"])
    )
    b = a.loc[a.Target == "1.0", :]
    totals = a.groupby(["Category"]).sum()
    b.index = a.Category.unique()
    prop = (b.value / totals.value * 100).reset_index()
    
    # Merge proportions with the counts
    df_model_merged = (
        a.merge(prop, left_on = "Category", right_on = "index")
         .drop(columns=["index"])
         .rename(columns={"value_x": "Count", "value_y": "Perc"})
    ).round(2)
    df_model_merged.loc[:, "Perc"] = df_model_merged.loc[:, "Perc"].astype(str) + "%"
    df_model_merged.loc[df_model_merged.Target == "0.0", "Perc"] = None
    
    return df_model_merged


def plot_triage_comparison(y, y_prob_cut, tc):
    """Plots categories stratified by outcome.
    
    Stacked bar plot of outcome in each category.
    
    Args:
        y: Targets.
        y_prob_cut: Cut probabilities.
        tc: Clinicians triage categories
        
    Returns:
        Plot of categories stratified by outcome, plots split by model.
    """
    columns = ["Label", "Category", "Target"]
    # Split dataframes for simplicity
    df_model = pd.DataFrame(
        np.column_stack((["Model"] * len(y_prob_cut), y_prob_cut, y)),
        columns = columns
    )
    df_clinicians = pd.DataFrame(
        np.column_stack((["Clinicians"] * len(tc), tc, y)),
        columns = columns
    )
    # Merge the prepared data frames
    t1 = prepare_triage_plot_df(df_model)
    t2 = prepare_triage_plot_df(df_clinicians)
    df_plt = pd.concat([t1, t2])

    return (
        ggplot(df_plt) + 
        geom_col(aes(x="Category", y="Count", label="Perc", fill="Target")) + 
        geom_text(aes(x="Category", y="Count", label = "Perc"), va="bottom") + 
        scale_fill_discrete(name = "Dead in 30 days", labels=["No", "Yes"]) + 
        scale_x_discrete(labels = ["Green", "Yellow", "Orange", "Red"]) + 
        facet_wrap("Label") + 
        xlab("Category") + 
        ylab("Number of patients")
    )
    