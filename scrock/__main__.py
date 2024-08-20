import sys

import numpy
import sklearn.preprocessing
import torch

import hdf5plugin
import scanpy

from .scrock import refine_clusters, find_doublets



def usage():
    print('Usage:')
    print('python3 -m scrock refine_clusters <data.h5ad>')
    print('or:')
    print('python3 -m scrock find_doublets <data.h5ad>')
    print('or (if you want to run from docker):')
    # TOFIX
    print('docker build --tag scrock-image .')
    print('docker run --name scrock --volume $(realpath ../data):/data scrock-image find_doublets /data/<data.h5ad>')
    print('docker rm scrock -f')
    return -1



def main(argv):
    print('argv:', argv)
    if len(argv) != 3:
        return usage()
    _, call, filename = argv
    adata = scanpy.read_h5ad(filename)

    torch.set_num_threads(1)

    if call == 'refine_clusters':
        X = adata.raw.X.toarray()
        if 'seurat_clusters' in adata.obs:
            y = numpy.array(list(adata.obs['seurat_clusters']))
        elif 'leiden' in adata.obs:
            y = numpy.array(list(adata.obs['leiden'].astype(numpy.int64)))
        elif 'cell_line' in adata.obs:
            y = sklearn.preprocessing.LabelEncoder().fit_transform(adata.obs['cell_line'])
        else:
            assert False, f'Can find cluster indices only in adata.obs["seurat_clusters" | "leiden" | "cell_line"], found keys: {adata.obs.keys()}'
        # TODO: insert into adata.obs['scrock_clusters'] and save
        print(list(refine_clusters(X, y)))
    elif call == 'find_doublets':
        X = adata.raw.X.toarray()
        # TODO: insert into adata.obs['scrock_is_doublet'] and save
        print(list(find_doublets(X)))



if __name__ == '__main__':
    main(sys.argv)
