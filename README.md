#  6 ways to significantly speed up Pandas with a couple lines of code

This repo contains all code used in [my article](https://alievmagomed.com/6-ways-to-significantly-speed-up-pandas-with-a-couple-lines-of-code/?utm_source=github&utm_medium=post&utm_campaign=social-networks).

Main file that generates all charts and benchmarks - `main.ipynb`. To reproduce results you should set `read_dumped` flag to `False`  (it is defined in the beginning) otherwise it would read dumped data from `benchs` directory.

##### Note:

Benchmarks for Modin framework were created separately in `modin_bench.py` file.