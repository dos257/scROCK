# scROCK

scROCK (single-cell Refinement Of Cluster Knitting) is an algorithm for correcting cluster labels for scRNA-seq data, based on [Xinchuan Zeng and Tony R. Martinez. 2001. An algorithm for correcting mislabeled data. Intell. Data Anal. 5, 6 (December 2001), 491–502.](https://dl.acm.org/doi/10.5555/1294000.1294004).


## Installation

```pip install https://github.com/dos257/ADE/tarball/master```

For private repository use:
```pip install git+https://{token}@github.com/dos257/ADE.git```

Use keys `--upgrade --no-deps --force-reinstall` for forced update from git repository.


## Usage

If `X` is log1p-preprocessed `numpy.array` of shape `(n_samples, n_genes)` and `y` is integer clustering labels (from Leiden algorithm),

```python
from scrock import scrock
y_fixed = scrock(X, y)
```


## Docker
For convenience, `scrock` supports simplified command line:
```
python3 -m scrock refine_clusters data.h5ad
```
or `find_doublets` instead of `refine_clusters`

For `refine_clusters` task, from file (here `data.h5ad`) `scrock` tries to read (in that order) `.obs["seurat_clusters"]`, `.obs["leiden"]`, `.obs["cell_line"]`.

Also, this command line could be run inside Docker.

Build Docker image:
```
docker build --tag scrock-image .
```
Run Docker image passing host path with input file:
```
docker run --name scrock --volume /host-path-to-input/data:/data scrock-image refine_clusters /data/sce_sc_10x_5cl_qc.h5ad
```
Output will be written to stdout.


## Known issues
If code consumes high CPU percent (but still works slowly), try:

```python
torch.set_num_threads(1)
```

Torch imperfect CPU parallelization spends most of the time in thread synchronization and slows down all process.
