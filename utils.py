from timeit import repeat

import matplotlib.pyplot as plt
import numpy as np
import json

BENCH_REPEATS = 5

def bench_func(func, *args, **kwargs):
    def closure_func():
        return func(*args, **kwargs)
    return repeat(closure_func, number=1, repeat=BENCH_REPEATS)

def perf_bench(setup_f, kernels_f, n_range):
    results = []
    for idx, kernel in enumerate(kernels_f):
        ker_res = []
        for n in n_range:
            test_df = setup_f(int(n))
            t_ = bench_func(kernel, test_df)
            average = np.mean(t_)
            print (f'N - {int(n)}, kernel #{idx}: {average:.4g} sec.')
            ker_res.append(average)
        results.append(ker_res)
    return results


def plot_results(res_arr, legends, pow_range, filename=None, bar=True, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(16, 10))
    else:
        fig = plt.gcf()

    x_len = len(res_arr[0])
    width = 0.3     # the width of the bars
    bar_delta = len(res_arr) * width + width
    ind = np.array([(i+1)*bar_delta for i in range(x_len)])
    for i in range(len(res_arr)):
        if bar:
            ax.bar(ind + (i * width), res_arr[i], width)
        else:
            ax.plot(ind, res_arr[i], linewidth=5)
        for idx, height in zip(ind, res_arr[i]):
            st_vl = '{0:.2f}' if height > 1 else '{0:.2g}'
            x, y = idx, height
            if bar:
                x = x + (i * width)
            ax.annotate(st_vl.format(height),
                         xy=(x, y),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    x_labels = [f'$10^{i}$' for i in range(pow_range[0], pow_range[1] + 1)]
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('N rows')
    ax.legend(legends)
    ax.set_yscale('log')
    ax.set_ylabel('Time (s)')
    if title:
        ax.set_title(title, fontsize=18)
    if filename:
        fig.savefig(f'static/{filename}', formar='png', dpi=300, bbox_inches = 'tight', pad_inches = 0)


def mean_word_len(line):
    for i in range(6):
        words = [len(i) for i in line.split()]
        res = sum(words) / len(words)
    return res



def dump(filename, obj):
    with open(f'benchs/{filename}', 'w') as f:
        json.dump(obj, f)


def load(filename):
    with open(f'benchs/{filename}', 'r') as f:
        return json.load(f)