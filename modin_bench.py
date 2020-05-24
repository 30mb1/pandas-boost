import numpy as np


import pandas as pd
from utils import plot_results, bench_func, perf_bench, BENCH_REPEATS, dump, load, mean_word_len
from math import log10
import os

BENCH_APPLY = False
BENCH_READ = False
BENCH_MATH = True


if __name__ == '__main__':
    import ray

    ray.init(num_cpus=8)
    # "ray" for ray backend
    engine = 'ray'
    os.environ["MODIN_ENGINE"] = engine
    import modin.pandas as mpd

    if BENCH_READ:
        mpd_read_res = bench_func(mpd.read_csv, 'big_csv.csv', header=0)
        print(f'Modin read time: {sum(mpd_read_res) / len(mpd_read_res)}')
        # mpd.read_csv('big_csv.csv', header=0)

        pd_read_res = bench_func(pd.read_csv, 'big_csv.csv', header=0)
        print(f'Pandas read time: {sum(pd_read_res) / len(mpd_read_res)}')
        # pd.read_csv('big_csv.csv', header=0)

    if BENCH_APPLY:
        df_mpd = mpd.read_csv('abcnews-date-text.csv', header=0)
        df_mpd = mpd.concat([df_mpd] * 10)

        log_n = int(log10(len(df_mpd)))
        n_range = np.logspace(2, log_n, log_n - 1)

        md_results = perf_bench(
            setup_f=lambda n: df_mpd.iloc[:n].headline_text,
            kernels_f=[
                # modin functions are lazy. Get first item of result to force computation
                lambda df: df.apply(mean_word_len)[0],
            ],
            n_range=n_range,
        )

        # concatenate with results form pandarallel testing
        pdr_results = load('pdr_results.json')
        # we dont need multiprocessing results
        md_results = md_results + [pdr_results[0], pdr_results[2]]
        dump(f'md_results_{engine}.json', md_results)

    if BENCH_MATH:
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000000, 6)), columns=list('abcdef'))
        df_mpd = mpd.DataFrame(np.random.randint(0, 100, size=(10000000, 6)), columns=list('abcdef'))

        log_n = int(log10(len(df_mpd)))
        n_range = np.logspace(3, log_n, log_n - 2)

        md_results = perf_bench(
            setup_f=lambda n: df_mpd.iloc[:n],
            kernels_f=[
                lambda df: df.mean(axis=1)[0],
                lambda df: df.prod(axis=1)[0],
                lambda df: df.median(axis=1)[0],
                lambda df: df.nunique(axis=1)[0],
            ],
            n_range=n_range,
        )

        pd_results = perf_bench(
            setup_f=lambda n: df.iloc[:n],
            kernels_f=[
                lambda df: df.mean(axis=1),
                lambda df: df.prod(axis=1),
                lambda df: df.median(axis=1),
                lambda df: df.nunique(axis=1),
            ],
            n_range=n_range,
        )

        results = list(zip(md_results, pd_results))
        dump(f'md_results_{engine}_math_2.json', results)

        md_results = perf_bench(
            setup_f=lambda n: df_mpd.iloc[:n],
            kernels_f=[
                lambda df: df.mean(axis=0)[0],
                lambda df: df.prod(axis=0)[0],
                lambda df: df.median(axis=0)[0],
                lambda df: df.nunique(axis=0)[0],
            ],
            n_range=n_range,
        )

        pd_results = perf_bench(
            setup_f=lambda n: df.iloc[:n],
            kernels_f=[
                lambda df: df.mean(axis=0),
                lambda df: df.prod(axis=0),
                lambda df: df.median(axis=0),
                lambda df: df.nunique(axis=0),
            ],
            n_range=n_range,
        )

        results = list(zip(md_results, pd_results))
        dump(f'md_results_{engine}_math.json', results)
