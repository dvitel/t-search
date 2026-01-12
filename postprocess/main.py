import os
import pandas as pd
from typing import Literal, Optional

import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times'] 

import json

def load(source: str, metric: str) -> pd.DataFrame: 
    with open(source, 'r') as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines]
    def get_metric(r):
        if metric in r:
            return r[metric]
        else:
            return None
    data = [ {"cfg": r["config_name"], "dataset": r["dataset"], "seed": r["seed"], "metric": get_metric(r) } for r in records ]
    df = pd.DataFrame.from_records(data)
    return df

def draw_table(input: str, metric: str = "test_nmse", transpose:bool=False) -> None:
    ''' Builds table of setting vs dataset with mean +- std in the cell for a given metric '''
    df = load(source=input, fields=[metric])
    summary = df.groupby(["cfg", "dataset"])["metric"].agg(["mean","std"]).reset_index()
    # Format as "mean ± std"
    summary["metric_summary"] = summary.apply(lambda row: f"{row['mean']:.3f} $\pm$ {row['std']:.2f}", axis=1)

    # Select only the columns for the table
    table_data = summary[["cfg","dataset","metric_summary"]]

    # Generate LaTeX table using tabulate
    # latex_table = tabulate(table_data, headers="keys", tablefmt="latex_booktabs", showindex=False)
    # print(latex_table)    

    # Pivot table
    table = summary.pivot(index="cfg", columns="dataset", values="metric_summary")
    
    # Optional transpose
    if transpose:
        table = table.T
    
    # Generate LaTeX
    latex_table = tabulate(table.reset_index(), headers="keys", tablefmt="latex_booktabs", showindex=False)
    print(latex_table)
    pass


def draw_chart(input: str, metric: str = "iter_fitness", 
               output: str = "charts/chart.pdf",
               figsize = (8, 3)) -> None:
    
    data = load(source=input, fields=[metric])
    
    plt.ioff()

    fig, axes = plt.subplots(num_rows, num_in_row, figsize=figsize)

    fig.tight_layout(rect=[0.015, 0.1, 1, 1], pad = 0, h_pad=0.1, w_pad=0.1)
    outfile = os.path.join(base_path, "plots", f"fig-{metric_name}-{suffix}.pdf")
    fig.savefig(outfile)  

    plt.close(fig)      

# def main():
#     pass 

if __name__ == "__main__":
    draw_chart(input="data/raw/tst.jsonlist",
               output="data/charts/koza_2.pdf")