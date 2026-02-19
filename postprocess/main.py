import math
import os
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
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
    data = [ {"cfg": r["config_name"], "dataset": r["dataset"], "seed": r["seed"], "metric": get_metric(r) } for r in records if r['test_pred_num_invalid'] < 10 ]
    df = pd.DataFrame.from_records(data)
    return df

def draw_table(input: str, 
               metric: str = "test_nmse", 
               transpose:bool=False,
               fmt="latex_booktabs") -> None:
    ''' Builds table of setting vs dataset with mean +- std in the cell for a given metric '''
    df = load(source=input, metric=metric)
    summary = df.groupby(["cfg", "dataset"])["metric"].agg(["mean","std"]).reset_index()
    # Format as "mean ± std"
    summary["metric_summary"] = summary.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.2f}", axis=1)

    # Select only the columns for the table
    # table_data = summary[["cfg","dataset","metric_summary"]]

    # Generate LaTeX table using tabulate
    # latex_table = tabulate(table_data, headers="keys", tablefmt="latex_booktabs", showindex=False)
    # print(latex_table)    

    # Pivot table
    table = summary.pivot(index="cfg", columns="dataset", values="metric_summary")
    
    # Optional transpose
    if transpose:
        table = table.T
    
    # Generate LaTeX
    latex_table = tabulate(table.reset_index(), headers="keys", tablefmt=fmt, showindex=False)
    print(latex_table)
    pass

def get_confidence_interval(data_np_t: np.ndarray, confidence_level=0.95):
    # data_np_t = data_df.T

    # adding -log - log scale 
    # data_np_t = -np.log10(data_np_t)

    data_np_t = np.where(np.isfinite(data_np_t), data_np_t, np.nan)

    data_mean = np.nanmean(data_np_t, axis=0)
    degrees_freedom = data_np_t.shape[0] - 1 # num of measures - 1
    # sample_standard_error = stats.sem(data_np_t, axis=0)
    sample_std = np.nanstd(data_np_t, axis=0, ddof=1)  # ddof=1 for sample std
    sample_count = np.sum(~np.isnan(data_np_t), axis=0)  # Count valid samples
    sample_standard_error = sample_std / np.sqrt(sample_count)
        
    confidence_interval = stats.t.interval(confidence_level, 
                                           sample_count - 1, 
                                           data_mean, 
                                           sample_standard_error)
    min_v = confidence_interval[0]
    max_v = confidence_interval[1]
    res_df = {"min_v": min_v, "mean": data_mean, "max_v": max_v}
    return res_df

def draw_trace(base_path: str, 
                    results_file: str, 
                    metric: str, 
                    datasets: list, 
                    config_names: list, 
                    seeds: list, 
                    xs: list[int | float],
                    output_prefix: str = '',
                    num_in_row: int = 3,
                    config_labels: dict = {},
                    dataset_names: dict = {},
                    colors: dict = {},
                    figsize=(8, 3)) -> None:
    file_to_read = f"{base_path}/{results_file}"
    metric_path = metric.split('.')
    with open(file_to_read, 'r') as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines]
    traces_map = {}
    for r in records:
        dataset = r["dataset"]
        if dataset not in datasets:
            continue
        config_name = r["config_name"]
        if config_name not in config_names:
            continue
        seed = r["seed"]
        if seed not in seeds:
            continue
        trace = r
        for mp in metric_path:
            trace = trace.get(mp, {})
        if len(trace) == 0:
            print(f"Warning: Metric {metric} not found in {config_name} for dataset {dataset}, config {config_name}, seed {seed}. Skipping.")
            trace = [0]
        else:
            trace = [-math.log(v, 10) for v in trace] # convert to log scale, handle 0 or negative values
            # continue
        num_missing = len(xs) - len(trace)
        if num_missing > 0:
            trace = trace + [trace[-1]] * num_missing
        if len(trace) > len(xs):
            trace = trace[:len(xs)]
        traces_map.setdefault(dataset, {}).setdefault(config_name, {})[seed] = trace

    datasets = [ds for ds in datasets if ds in traces_map ]

    default_seeds = set(seeds)
    dataset_seeds = {}
    for dataset in datasets:
        # common_seeds = set(seeds)
        for config_name in config_names:
            if dataset in traces_map and config_name in traces_map[dataset]:
                seeds_with_trace = set(traces_map[dataset][config_name].keys())
                missing_seeds = default_seeds - seeds_with_trace
                if len(missing_seeds) > 0:
                    print(f"Warning: Missing trace for dataset {dataset}, config {config_name}, seeds {missing_seeds}")
                    for mseed in missing_seeds:
                        traces_map[dataset][config_name][mseed] = [0] * len(xs)
                # common_seeds = common_seeds.intersection(seeds_with_trace)
        # if len(common_seeds) == 0:
        #     print(f"Error: No common seeds found for dataset {dataset} across all configs. Skipping {dataset}.")
        #     continue
        dataset_seeds[dataset] = seeds #[s for s in seeds if s in common_seeds]

    datasets = [ds for ds in datasets if ds in dataset_seeds and len(dataset_seeds[ds]) > 0]

    dataset_configs = {}
    for dataset in datasets:
        configs_with_traces = []
        for config_name in config_names:
            if dataset in traces_map and config_name in traces_map[dataset]:
                configs_with_traces.append(config_name)
        dataset_configs[dataset] = configs_with_traces

    avg_traces = {}
    for dataset in datasets:
        ds_seeds = dataset_seeds[dataset]
        ds_config_names = dataset_configs[dataset]
        for config_name in ds_config_names:
            np_arr = np.array([traces_map[dataset][config_name][s] for s in ds_seeds])
            res_traces = get_confidence_interval(np_arr)
            avg_traces.setdefault(dataset, {})[config_name] = res_traces
    
    num_rows = len(datasets) // num_in_row + (1 if len(datasets) % num_in_row > 0 else 0)
    plt.ioff()
    ordered_handles = []
    ordered_labels = []    
    fig, axes = plt.subplots(num_rows, num_in_row, figsize=figsize, sharex=True)
    # fig.subplots_adjust(wspace=0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
    for ds_id, dataset in enumerate(datasets):
        row_id = ds_id // num_in_row
        col_id = ds_id % num_in_row
        ax = axes[row_id, col_id] if num_rows > 1 else axes[col_id]
        ds_config_names = dataset_configs[dataset]
        y_max = 0
        for config_name in ds_config_names:
            res_traces = avg_traces[dataset][config_name]
            max_v = res_traces["max_v"]
            # Handle inf values from -log(0)
            finite_max = max_v[np.isfinite(max_v)]
            if len(finite_max) > 0:
                y_max = max(y_max, np.max(finite_max) * 1.1)
                
        for config_name in ds_config_names:
            res_traces = avg_traces[dataset][config_name]
            min_v = res_traces["min_v"]
            y_mean = res_traces["mean"]
            max_v = res_traces["max_v"]
            color = colors.get(config_name, None)
            ln = ax.plot(xs, y_mean, label=config_labels[config_name], linewidth=1, color=color)
            if ds_id == 0:
                ordered_handles.append(ln[0])
                ordered_labels.append(config_labels[config_name])
            ax.fill_between(xs, min_v, max_v, color=ln[0].get_color(), alpha=0.2)
        
        ax.set_ylim(0, math.ceil(y_max)) 
        ax.set_xlim(1, len(xs))
        dataset_name = dataset_names.get(dataset, dataset)
        ax.set_title(f"{dataset_name}", fontsize=10, y=0.99, loc='right', pad=-7)

    # for ds_id in range(len(datasets), num_rows * num_in_row):
    #     row_id = ds_id // num_in_row
    #     col_id = ds_id % num_in_row
    #     ax = axes[row_id, col_id] if num_rows > 1 else axes[col_id]
    #     ax.axis('off')       
        
        # plt.title(f"Loss Trace for {dataset}")
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid(True)
        # output_file = os.path.join(base_path, "plots", f"{output_prefix or ''}loss_trace_{dataset}.pdf")
        # plt.savefig(output_file)
        # plt.close()
        ax.tick_params(axis='both', labelsize=8)
        # ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        if y_max > 5:            
            custom_ticks = [2, 5, 9]
        else:
            custom_ticks = [1, 2, 5]
        y_min, y_max_limit = ax.get_ylim()
        visible_ticks = [t for t in custom_ticks if y_min <= t <= y_max_limit]
        ax.set_yticks(visible_ticks)    
        
    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )
    # fig.text(0.01, 0.5, 'Accuracy \\%', ha='center', va='center', rotation='vertical', fontsize=10)
    # fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)
    fig.tight_layout(rect=[0, 0.06, 1, 1], pad=0, h_pad=0.2, w_pad=0.2)

    # plt.tight_layout()
    outfile = f"{base_path}/{output_prefix}{metric}.pdf"
    fig.savefig(outfile)  
    plt.close(fig)
    pass      

def draw_bars(base_path: str, 
                    results_file: str, 
                    metric: str, 
                    datasets: list, 
                    config_names: list, 
                    seeds: list, 
                    output_prefix: str = '',
                    num_in_row: int = 3,
                    config_labels: dict = {},
                    dataset_names: dict = {},
                    colors: dict = {},
                    figsize=(8, 3)) -> None:
    file_to_read = f"{base_path}/{results_file}"
    metric_path = metric.split('.')
    with open(file_to_read, 'r') as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines]
    traces_map = {}
    for r in records:
        dataset = r["dataset"]
        if dataset not in datasets:
            continue
        config_name = r["config_name"]
        if config_name not in config_names:
            continue
        seed = r["seed"]
        if seed not in seeds:
            continue
        trace = r
        for mp in metric_path:
            trace = trace.get(mp, {})
        if isinstance(trace, dict) or trace is None:
            print(f"Warning: Metric {metric} not found in {config_name} for dataset {dataset}, config {config_name}, seed {seed}. Skipping.")
            continue
        # num_missing = len(xs) - len(trace)
        # if num_missing > 0:
        #     trace = trace + [0] * num_missing
        # if len(trace) > len(xs):
        #     trace = trace[:len(xs)]
        if trace > 1: 
            print(f"{dataset}:{config_name}:{seed}, {trace} high loss value, assume 0")
            trace = 0
        else:
            trace = -math.log(trace, 10)
        traces_map.setdefault(dataset, {}).setdefault(config_name, {})[seed] = trace # scalar value here

    datasets = [ds for ds in datasets if ds in traces_map ]

    default_seeds = set(seeds)
    dataset_seeds = {}
    for dataset in datasets:
        # common_seeds = set(seeds)
        for config_name in config_names:
            if dataset in traces_map and config_name in traces_map[dataset]:
                seeds_with_trace = set(traces_map[dataset][config_name].keys())
                missing_seeds = default_seeds - seeds_with_trace
                # if len(missing_seeds) > 0:
                #     print(f"Warning: Missing trace for dataset {dataset}, config {config_name}, seeds {missing_seeds}")
                for mseed in missing_seeds:
                    print(f"Warning: Missing trace for dataset {dataset}, config {config_name}, seed {mseed}")
                    traces_map[dataset][config_name][mseed] = 0
                # common_seeds = common_seeds.intersection(seeds_with_trace)
        # if len(common_seeds) == 0:
        #     print(f"Error: No common seeds found for dataset {dataset} across all configs. Skipping {dataset}.")
        #     continue
        dataset_seeds[dataset] = seeds #[s for s in seeds if s in default_seeds]

    datasets = [ds for ds in datasets if ds in dataset_seeds and len(dataset_seeds[ds]) > 0]

    dataset_configs = {}
    for dataset in datasets:
        configs_with_traces = []
        for config_name in config_names:
            if dataset in traces_map and config_name in traces_map[dataset]:
                configs_with_traces.append(config_name)
        dataset_configs[dataset] = configs_with_traces

    avg_traces = {}
    for dataset in datasets:
        ds_seeds = dataset_seeds[dataset]
        ds_config_names = dataset_configs[dataset]
        for config_name in ds_config_names:
            np_arr = np.array([traces_map[dataset][config_name][s] for s in ds_seeds]) # 1d array
            # res_traces = get_confidence_interval(np_arr)
            avg_traces.setdefault(dataset, {})[config_name] = np_arr
    
    num_rows = len(datasets) // num_in_row + (1 if len(datasets) % num_in_row > 0 else 0)
    plt.ioff()
    ordered_handles = []
    ordered_labels = []    
    fig, axes = plt.subplots(num_rows, num_in_row, figsize=figsize, sharex=True)
    # fig.subplots_adjust(wspace=0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
    for ds_id, dataset in enumerate(datasets):
        row_id = ds_id // num_in_row
        col_id = ds_id % num_in_row
        ax = axes[row_id, col_id] if num_rows > 1 else axes[col_id]
        ds_config_names = dataset_configs[dataset]

        violin_data = []
        violin_positions = []
        violin_colors = []
        
        for idx, config_name in enumerate(config_names):
            points = avg_traces[dataset].get(config_name, [])  # 1D array of values across seeds
            if len(points) == 0: # add all 0 for common seeds
                points = [0] * len(dataset_seeds[dataset])
            violin_data.append(points)
            violin_positions.append(idx)
            violin_colors.append(colors.get(config_name, 'gray'))
        
        parts = ax.violinplot(
            violin_data,
            positions=violin_positions,
            widths=0.7,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )        

        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
        
        if ds_id == 0:
            for config_name, color in zip(ds_config_names, violin_colors):
                from matplotlib.patches import Patch
                handle = Patch(facecolor=color, edgecolor='black', alpha=0.7)
                ordered_handles.append(handle)
                ordered_labels.append(config_labels.get(config_name, config_name))


        ax.set_xticks(violin_positions)
        ax.set_xticklabels([config_labels.get(cn, cn) for cn in config_names], 
                           fontsize=8, ha='center')

        all_data = np.concatenate(violin_data)        
        y_min = max(0, np.min(all_data) * 0.9)
        y_max = np.max(all_data) * 1.1
        ax.set_ylim(y_min, y_max)
        dataset_name = dataset_names.get(dataset, dataset)
        ax.set_title(f"{dataset_name}", fontsize=10, y=0.99, loc='right', pad=-7)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)


    # for ds_id in range(len(datasets), num_rows * num_in_row):
    #     row_id = ds_id // num_in_row
    #     col_id = ds_id % num_in_row
    #     ax = axes[row_id, col_id] if num_rows > 1 else axes[col_id]
    #     ax.axis('off')       
        
        # plt.title(f"Loss Trace for {dataset}")
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid(True)
        # output_file = os.path.join(base_path, "plots", f"{output_prefix or ''}loss_trace_{dataset}.pdf")
        # plt.savefig(output_file)
        # plt.close()
        # ax.tick_params(axis='both', labelsize=8)
        # ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        # if y_max > 5:            
        #     custom_ticks = [2, 5, 9]
        # else:
        #     custom_ticks = [1, 2, 5]
        # y_min, y_max_limit = ax.get_ylim()
        # visible_ticks = [t for t in custom_ticks if y_min <= t <= y_max_limit]
        # ax.set_yticks(visible_ticks)    

    for ds_id in range(len(datasets), num_rows * num_in_row):
        row_id = ds_id // num_in_row
        col_id = ds_id % num_in_row
        ax = axes[row_id, col_id] if num_rows > 1 else axes[col_id]
        ax.axis('off')        
        
    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )
    # fig.text(0.01, 0.5, 'Accuracy \\%', ha='center', va='center', rotation='vertical', fontsize=10)
    # fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)
    fig.tight_layout(rect=[0, 0.06, 1, 1], pad=0, h_pad=0.2, w_pad=0.2)

    # plt.tight_layout()
    outfile = f"{base_path}/{output_prefix}{metric}.pdf"
    fig.savefig(outfile)  
    plt.close(fig)
    pass      


if __name__ == "__main__":
    # data = load(source="data/raw/r.jsonlist", metric="final_time")
    # data = load(source="data/results.jsonlist", metric="final_time")
    # data["metric"] = data["metric"] / 60.0 / 1000.0  # convert to minutes
    # print(data)

    base_folder = "data/results-02-19"
    results_file = "1.jsonlist"
    datasets = ['r_1', 'r_2', 'keijzer_3', 'keijzer_4', 'keijzer_11', 'nguyen_12', 'pagie_1', 'vladislavleva_1', 'koza_3', 'keijzer_6', 'vladislavleva_8', 'korns_14'] #, 'korns_14', 'korns_15']
    config_names = ['koza', 'semantic', 'geometric', 'competent', 'point_optim']
    max_seed = 30
    seeds = list(range(max_seed))
    config_labels={'koza': 'Kz', 'semantic': 'Sem', 'geometric': "Geom", 'competent': 'CO', 'point_optim': 'PO'}
    dataset_names = {
        'r_1': 'R1', 'r_2': 'R2', 'keijzer_3': 'Kj3', 'keijzer_4': 'Kj4', 'keijzer_11': 'Kj11',
        'nguyen_12': 'Ng12', 'pagie_1': 'Pg1', 'vladislavleva_1': 'Vl1', 'koza_3': 'Kz3',
        'keijzer_6': 'Kj6', 'vladislavleva_8': 'Vl8', 'korns_13': 'Kn13', 'korns_14': 'Kn14', 'korns_15': 'Kn15'
    }
    colors = {"koza": "blue", "semantic": "orange", "point_optim": "green", "geometric": "red", "competent": "purple"}

    draw_trace(base_path = base_folder, results_file = results_file,
                    metric="evaluator.loss_trace", datasets=datasets, 
                    config_names=config_names, seeds=seeds,
                    xs = np.arange(1, 51), 
                    config_labels=config_labels,
                    dataset_names=dataset_names,
                    colors=colors,
                    output_prefix="c0-", figsize=(6, 4))
    draw_bars(
        base_path=base_folder, 
        results_file=results_file,
        metric="test_nmse", 
        datasets=datasets, 
        config_names=config_names, 
        seeds=seeds,
        config_labels=config_labels,
        dataset_names=dataset_names,
        colors=colors,
        output_prefix="v0-", 
        figsize=(6, 4)
    )    
    pass
    # draw_table(input="data/raw/tst2.jsonlist",
    #            metric="test_nmse",
    #            transpose=True)
    # draw_chart(input="data/raw/tst2.jsonlist",
    #            output="data/charts/koza_2.pdf")