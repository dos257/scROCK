import os

import numpy
import pandas
import hdf5plugin
import scanpy
import sklearn.preprocessing



def file_get(filename):
    with open(filename, 'rb') as fp:
        return fp.read()

def file_put(filename, content):
    with open(filename, 'wb') as fp:
        fp.write(content)

def pickle_put(filename, obj):
    import pickle
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_get(filename):
    import pickle
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def md5_file(filename):
    import hashlib
    h = hashlib.md5()
    with open(filename, 'rb') as fp:
        while True:
            data = fp.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()



# TODO: download source files, run R preprocessing
# TOFIX: return raw counts for pbmc_*, sc_mixology_*
ROOT = '.'

from contextlib import contextmanager
@contextmanager
def cd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)



def download(url, to, md5hash, unpack_to=None):
    import requests
    if not os.path.exists(to):
        f = requests.get(url).content
        file_put(to, f)
    md5hash_real = md5_file(to)
    assert md5hash_real == md5hash, f'For {to} md5 = {md5hash_real}, expected {md5hash}'
    if unpack_to:
        import gzip
        with gzip.GzipFile(to) as gz:
            content = gz.read()
            for filename in (unpack_to if type(unpack_to) == list else [unpack_to]):
                file_put(filename, content)



sources = '''\
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file&file=GSE96583%5Fbatch1%2Egenes%2Etsv%2Egz | GSE96583_batch1.genes.tsv.gz | ccbf1a8150929c7a8aa6052fc6d9d402 | GSM2560245_genes.tsv,GSM2560246_genes.tsv,GSM2560247_genes.tsv
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file&file=GSE96583%5Fbatch1%2Etotal%2Etsne%2Edf%2Etsv%2Egz | GSE96583_batch1.total.tsne.df.tsv.gz | 606bd42aba49319925274b27fbf7af16 | GSE96583_batch1.total.tsne.df.tsv

https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560245&format=file&file=GSM2560245%5FA%2Emat%2Egz | GSM2560245_A.mat.gz | 23bb57d79d9ecd36e65f2d19e8342d25 | GSM2560245_matrix.mtx
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560245&format=file&file=GSM2560245%5Fbarcodes%2Etsv%2Egz | GSM2560245_barcodes.tsv.gz | a64a6232a47a59a71fdfe05fb4f807e5 | GSM2560245_barcodes.tsv

https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560246&format=file&file=GSM2560246%5FB%2Emat%2Egz | GSM2560246_B.mat.gz | 9b6095d3ffc090cc64760107ad026db0 | GSM2560246_matrix.mtx
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560246&format=file&file=GSM2560246%5Fbarcodes%2Etsv%2Egz | GSM2560246_barcodes.tsv.gz | 4bbe93ccdcc43dd3436a85506f54a782 | GSM2560246_barcodes.tsv

https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560247&format=file&file=GSM2560247%5FC%2Emat%2Egz | GSM2560247_C.mat.gz | 57f7ab1fb598eaedb90d94837faa678c | GSM2560247_matrix.mtx
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560247&format=file&file=GSM2560247%5Fbarcodes%2Etsv%2Egz | GSM2560247_barcodes.tsv.gz | 520f1d4f3ffe1ecede5f5cfc84e307ce | GSM2560247_barcodes.tsv


https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file&file=GSE96583%5Fbatch2%2Egenes%2Etsv%2Egz | GSE96583_batch2.genes.tsv.gz | 464dcbb35efbda7dee8a8e59a09e049f | GSM2560248_genes.tsv,GSM2560249_genes.tsv
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96583&format=file&file=GSE96583%5Fbatch2%2Etotal%2Etsne%2Edf%2Etsv%2Egz | GSE96583_batch2.total.tsne.df.tsv.gz | 080d02ff2f80606430cff33cffea644a | GSE96583_batch2.total.tsne.df.tsv

https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560248&format=file&file=GSM2560248%5F2%2E1%2Emtx%2Egz | GSM2560248_2.1.mtx.gz | fbac682dfe89b625d08f35f1fb3d4f8c | GSM2560248_matrix.mtx
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560248&format=file&file=GSM2560248%5Fbarcodes%2Etsv%2Egz | GSM2560248_barcodes.tsv.gz | e9805aba1649174af49f5473aadeeadf | GSM2560248_barcodes.tsv

https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560249&format=file&file=GSM2560249%5F2%2E2%2Emtx%2Egz | GSM2560249_2.2.mtx.gz | d27845a289fa51c58fb20574f307ecc7 | GSM2560249_matrix.mtx
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2560249&format=file&file=GSM2560249%5Fbarcodes%2Etsv%2Egz | GSM2560249_barcodes.tsv.gz | f3f0ecba22c2261148f412dd045f0446 | GSM2560249_barcodes.tsv
'''.splitlines()



def download_source_files():
    cwd = os.getcwd()
    os.chdir(ROOT)

    for source in sources:
        if not source: continue
        url, filename, md5hash, unpack_to = source.split(' | ')
        if ',' in unpack_to:
            unpack_to = unpack_to.split(',')
        download(url, to='archives/' + filename, md5hash=md5hash, unpack_to=unpack_to)

    os.chdir(cwd)



# def load_pbmc_kevin():
#     counts = pandas.read_csv('donorA_norm_counts_all.txt', sep="\t", index_col=0)
#     celltypes = pandas.read_csv('donorA_celltypes.txt', sep="\t")
#     labels = list(celltypes['Celltype'])
#     X = numpy.array(counts)
#     y = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
#     return X, y



# def load_pbmc_kevin_clip5():
#     X, y = load_pbmc_kevin()
#     X = numpy.clip(X, 0.0, 5.0)
#     return X, y



def load_pbmc_codeocean():
    '''
    https://codeocean.com/capsule/3839794/tree/v1

    https://files.codeocean.com/files/verified/62550df2-c376-401d-a495-edecd41c92c2_v1.0/results.2f148e27-d780-42ef-9958-bf73d27c1068/pbmc3k_final.rds

    md5 = d5820048fe4c3f3d25e94ae86e6276f1

    R:
    rds <- readRDS("pbmc3k_final.rds")
    rds <- UpdateSeuratObject(rds)
    library(reticulate)
    np <- import("numpy")

    # not rds[["RNA"]]$data
    r_to_py(rds[["RNA"]]$counts)$toarray()$dump("pbmc_codeocean.npy")

    a <- np$array(rds$seurat_clusters)
    r_to_py(a)$dump("pbmc_codeocean_y.npy")
    '''
    with cd(ROOT):
        X = numpy.load("pbmc_codeocean.npy", allow_pickle=True).T
        y = numpy.load("pbmc_codeocean_y.npy", allow_pickle=True).T
        y = y.astype(int)
        return X, y

    '''
    Alternatively:
    data <- readRDS("pbmc3k_final.rds")
    data <- UpdateSeuratObject(data)
    SaveH5Seurat(data, filename = "pbmc3k_final.h5Seurat")
    # raw.X will be filled with data if X is filled with scale.data; otherwise, it will be filled with counts. If counts is not present, then raw will not be filled 
    Convert("pbmc3k_final.h5Seurat", dest = "h5ad")

    adata = scanpy.read_h5ad('pbmc3k_final.h5ad')
    X = adata.raw.X
    y = numpy.array(list(adata.obs['seurat_clusters']))
    '''



def load_sc_mixology_3cl():
    '''
    https://github.com/LuyiTian/sc_mixology/blob/master/data/sincell_with_class.RData

    https://github.com/LuyiTian/sc_mixology/raw/master/data/sincell_with_class.RData

    md5 = 2da088bc06012394136e9690eb09c3a1

    R:
    load("sincell_with_class.RData")
    library(SeuratDisk)
    SaveH5Seurat(as.Seurat(sce_sc_10x_qc), filename = "sce_sc_10x_qc_seurat.h5Seurat")
    Convert("sce_sc_10x_qc_seurat.h5Seurat", dest = "h5ad")
    '''
    with cd(ROOT):
        adata = scanpy.read_h5ad('sce_sc_10x_qc_seurat.h5ad')
        X = adata.raw.X.toarray()
        y = sklearn.preprocessing.LabelEncoder().fit_transform(adata.obs['cell_line'])
        return X, y



def load_sc_mixology_5cl():
    '''
    https://github.com/LuyiTian/sc_mixology/blob/master/data/sincell_with_class_5cl.RData

    https://github.com/LuyiTian/sc_mixology/raw/master/data/sincell_with_class_5cl.RData

    md5 = 6924ea5e9759857a6acfa3abf84e718a

    R:
    load("sincell_with_class_5cl.RData")
    library(SeuratDisk)
    SaveH5Seurat(as.Seurat(sce_sc_10x_5cl_qc), filename = "sce_sc_10x_5cl_qc.h5Seurat")
    Convert("sce_sc_10x_5cl_qc.h5Seurat", dest = "h5ad")
    '''
    with cd(ROOT):
        adata = scanpy.read_h5ad('sce_sc_10x_5cl_qc.h5ad')
        X = adata.raw.X.toarray()
        y = sklearn.preprocessing.LabelEncoder().fit_transform(adata.obs['cell_line'])
        return X, y



def load_impact2024():
    '''
    Data from paper "The impact of package selection and versioning on single-cell RNA-seq analysis" https://doi.org/10.1101/2024.04.04.588111

    https://caltech.app.box.com/s/i4dk3iwdg1ufmryyblg9pcszl5b08gsc/file/1471626680707
            RMEJLBASBMP_2024 > Seurat_Scanpy_Version_Control > scanpy_version_control > scanpyv1_9.tar.gz
    
    (direct download link is not available)

    md5 = 1c98baeb5f47dc5cd9de485ef6881f7f

    unpack scanpyv1_9.tar.gz
    '''
    with cd(ROOT):
        adata = scanpy.read_h5ad('scanpyv1_9/adata_all_genes.h5ad')
        X = adata.X.toarray()
        adata = scanpy.read_h5ad('scanpyv1_9/adata.h5ad')
        y = numpy.array(list(adata.obs['leiden'].apply(int)))
        return X, y



def load_GSM(prefix, metadata_filename, name_column, name):
    with cd(ROOT):
        adata = scanpy.read_10x_mtx('.', prefix = prefix + '_')

        df = pandas.read_csv(metadata_filename, sep='\t')
        name_mask = numpy.array(df[name_column] == name)
        mask = df[name_mask].multiplets != 'ambs'

        X = adata.X.toarray()[mask, :]
        y = numpy.array(df[name_mask].multiplets[mask] == 'doublet').astype(numpy.int32)

        return X, y



def load_GSM2560245():
    return load_GSM('GSM2560245', 'GSE96583_batch1.total.tsne.df.tsv', 'batch', 'A')

def load_GSM2560246():
    return load_GSM('GSM2560246', 'GSE96583_batch1.total.tsne.df.tsv', 'batch', 'B')

def load_GSM2560247():
    return load_GSM('GSM2560247', 'GSE96583_batch1.total.tsne.df.tsv', 'batch', 'C')

def load_GSM2560248():
    return load_GSM('GSM2560248', 'GSE96583_batch2.total.tsne.df.tsv', 'stim', 'ctrl')

def load_GSM2560249():
    return load_GSM('GSM2560249', 'GSE96583_batch2.total.tsne.df.tsv', 'stim', 'stim')



def load_mnist():
    import sklearn.datasets
    X, y = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X, y.astype(int)



def dataset_metadata(target, loader):
    return {'target': target, 'loader': loader}

available_datasets = {
    # 'pbmc_kevin': dataset_metadata('cluster index', load_pbmc_kevin),
    # 'pbmc_kevin_clip5': dataset_metadata('cluster index', load_pbmc_kevin_clip5),
    'pbmc_codeocean': dataset_metadata('cluster index', load_pbmc_codeocean),
    'sc_mixology_3cl': dataset_metadata('cluster index', load_sc_mixology_3cl),
    'sc_mixology_5cl': dataset_metadata('cluster index', load_sc_mixology_5cl),
    'impact2024': dataset_metadata('cluster index', load_impact2024),

    'GSM2560245': dataset_metadata('doublet', load_GSM2560245),
    'GSM2560246': dataset_metadata('doublet', load_GSM2560246),
    'GSM2560247': dataset_metadata('doublet', load_GSM2560247),
    'GSM2560248': dataset_metadata('doublet', load_GSM2560248),
    'GSM2560249': dataset_metadata('doublet', load_GSM2560249),

    'mnist': dataset_metadata('classification', load_mnist),
}

def load_dataset(name):
    if name not in available_datasets:
        raise ValueError(f'Unknown dataset {name}')
    X, y = available_datasets[name]['loader']()
    return X, y



# TOFIX: include in cluster index datasets
def load_impact2024_fig1_seurat():
    '''
    https://caltech.app.box.com/s/i4dk3iwdg1ufmryyblg9pcszl5b08gsc/folder/266758165968 RMEJLBASBMP_2024 > Fig1_objects
        seu.rds | 23ea79d5334ff506e5ae7378c52dee88
            https://caltech.app.box.com/s/i4dk3iwdg1ufmryyblg9pcszl5b08gsc/file/1542998855386
    '''
    '''
    R:
    library(scCustomize)
    data <- readRDS("seu.rds")
    data <- Convert_Assay(seurat_object = data, convert_to = "V3", assay = "RNA")
    SaveH5Seurat(data, filename = "seu.h5Seurat")
    Convert("seu.h5Seurat", dest = "h5ad", assay = "RNA")
    '''
    with cd(ROOT):
        adata = scanpy.read_h5ad('seu.h5ad')
        X = adata.raw.X.toarray()
        y = numpy.array(list(adata.obs['seurat_clusters']))
        return X, y



def load_impact2024_fig1_scanpy():
    '''
    https://caltech.app.box.com/s/i4dk3iwdg1ufmryyblg9pcszl5b08gsc/folder/266758165968 RMEJLBASBMP_2024 > Fig1_objects
        adata.h5ad | 167328e0906e215ebb2090300a0cf812
            https://caltech.app.box.com/s/i4dk3iwdg1ufmryyblg9pcszl5b08gsc/file/1542991684463
    '''
    with cd(ROOT):
        adata = scanpy.read_h5ad('adata.h5ad')
        X = adata.raw.X.toarray()
        y = numpy.array(list(adata.obs['leiden'].astype(numpy.int64)))
        return X, y
