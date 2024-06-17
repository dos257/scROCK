import numpy
import scipy.special

import torch
from tqdm import tqdm



unicode = str
def md5(s):
    import hashlib
    if type(s) == unicode:
        s = s.encode('utf-8')
    return hashlib.md5(s).hexdigest()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MLPWithLinearOutput(torch.nn.Module):
    '''
    Generic MLP
    @param n_features: int, number of features on input
    @param n_classes: int, number of classes in multiclass classification task
    @param layers: array, number of neurons in hidden layers
    @param nonlinearity: str|torch.nn.Module, activation function
    @param batch_norm: bool, to use BatchNorm layer after every linear except last one
    @param seed: int, torch seed to initialize NN weights
    '''
    def __init__(self, n_features, n_classes, layers = [32, 16], nonlinearity='relu', batch_norm=False, seed=0):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        layer_sizes = [self.n_features] + layers + [self.n_classes]
        self.layers = []

        torch.manual_seed(seed)

        for i, (layer_size_m, layer_size_n) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append(torch.nn.Linear(layer_size_m, layer_size_n))
            if i + 2 < len(layer_sizes):
                if batch_norm:
                    self.layers.append(torch.nn.BatchNorm1d(layer_size_n))

                if type(nonlinearity) == str:
                    if nonlinearity == 'gelu':
                        self.layers.append(torch.nn.GELU())
                    elif nonlinearity == 'relu':
                        self.layers.append(torch.nn.ReLU())
                    else:
                        assert False, f'{nonlinearity} is not supported'
                else:
                    self.layers.append(nonlinearity)

        self.layers = torch.nn.ParameterList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



def train_dnn(
    net, dataset, dataset_test=None,
    optimizer='nadam', optimizer_lr = 0.0001,
    n_epochs=100, n_batches=None,
    batch_size=32, batch_scheme='random-shuffle',
    verbose=False, plots=False, on_each_batch=None, seed=0
):
    '''
    Trains DNN on dataset
    @param net: torch.nn.Module, DNN to train
    @param dataset: dataset to use for train
    @param dataset_test: dataset|None, dataset to use for test
    @param optimizer: str|torch.optim.Optimizer, optimizer to use
    @param optimizer_lr: float, learning rate of optimizer
    @param n_epochs: int|None, how many epochs to train
    @param n_batches: int|None, how many batches to train (if set, n_epochs should be None)
    @param batch_size: int, size of batch
    @param batch_replace: bool, sample batch with replacement
    @param verbose: bool, show plot with losses
    @param on_each_batch: callable, to call every batch
    @param seed: int, torch random seed
    '''
    assert type(n_epochs) is not None and n_batches is None or n_epochs is None and type(n_batches) is not None, 'Only one of n_epochs, n_batches should be not None'
    assert batch_scheme in ['random-shuffle']

    # TOFIX: where it should be?
    torch.manual_seed(seed)
    net.to(device)
    if verbose:
        print(net)

    if type(optimizer) == str:
        if optimizer == 'nadam':
            optimizer = torch.optim.NAdam(net.parameters(), lr=optimizer_lr)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
        elif optimizer == 'adamax':
            optimizer = torch.optim.Adamax(net.parameters(), lr=optimizer_lr)
        elif optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(net.parameters(), lr=optimizer_lr)
        else:
            assert False, f'{optimizer} is not supported'

    lossfn = torch.nn.CrossEntropyLoss()

    losses_train = []
    losses_test = []

    n_epoch_batches = dataset.n_samples // batch_size
    if n_batches is not None:
        n_epochs = (n_batches + n_epoch_batches - 1) // n_epoch_batches
    ibatch = 0

    rnd = torch.Generator()
    rnd.manual_seed(seed)
    if batch_scheme == 'random-shuffle':
        sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset, generator=rnd), batch_size=batch_size, drop_last=False)

    for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
        losses_epoch = []
        for i_epoch_batch, batch_idx in enumerate(sampler):
            optimizer.zero_grad()
            X_batch, y_batch = dataset[batch_idx]
            out_train = net(torch.Tensor(X_batch).to(device))
            loss_train = lossfn(out_train, torch.Tensor(y_batch).to(torch.long).to(device))

            if on_each_batch is not None:
                on_each_batch(ibatch, batch_idx, net, out_train, out_train.cpu().detach().numpy(), loss_train.cpu().detach().numpy())

            loss_train.backward()
            optimizer.step()
            losses_epoch.append(loss_train.cpu().detach().numpy())

            ibatch += 1
            if n_batches is not None and ibatch >= n_batches:
                break

        losses_train.append(numpy.mean(losses_epoch))
        loss_test = numpy.nan
        if dataset_test is not None:
            # TODO: batchwise
            out_test = net(torch.Tensor(dataset_test.X).to(device))
            loss_test = lossfn(out_test, torch.Tensor(dataset_test.y).to(torch.long).to(device)).cpu().detach().numpy()
        losses_test.append(loss_test)

        if verbose and plots:
            try:
                import IPython
                IPython.display.clear_output(wait=True)
            except:
                pass
            from matplotlib.pyplot import title, plot, legend, show, xlabel, ylabel
            if loss_test:
                title(f'last test loss = {losses_test[-1]:.6f}')
            else:
                title(f'last train loss = {losses_train[-1]:.6f}')
            xlabel('epoch')
            ylabel('cross-entropy')
            plot(losses_train, label='train')
            if loss_test:
                plot(losses_test, label='test')
                legend()
            show()

    return net, list(map(float, losses_train)), list(map(float, losses_test))



class DatasetWithMutableLabels(torch.utils.data.Dataset):
    def __init__(self, X, y, D=1.0, seed=0):
        '''
        @param X: array-like (n_samples, n_features), feature values for samples
        @param y: array-like (n_samples,), classes of samples, 0..n_classes-1
        @param D: float, probability that label of sample is correct
        @param seed: int, numpy random seed
        '''
        super().__init__()
        self.X = X
        self.y = y
        self.y_mislabel = y.copy()
        self.mislabel_idx = []
        self.y_new = y.copy()
        self.n_samples, self.n_features = X.shape
        self.n_classes = max(y) + 1
        self.D = D
        self.p = self.smoothed_probabilities(self.n_samples, self.n_classes, self.y, self.D)
        self.rnd = numpy.random.RandomState(seed)

    def __getitem__(self, index):
        return self.X[index, :], self.y_new[index]

    def __len__(self):
        return self.n_samples

    def smoothed_probabilities(self, n_samples, n_classes, y, D):
        p = numpy.zeros((n_samples, n_classes))
        D_others = (1 - D) / (n_classes - 1)
        for i in range(n_samples):
            p[i, :] = D_others
            p[i, y[i]] = D
        return p

    def update_class_probabilities(self, new_p, batch_idx=None):
        '''
        @param new_p: array-like (batch_size, n_classes), new values of class probabilities
        @param batch_idx: array-like (batch_size,)|None, indices of batch or None if all probabilities are updating
        '''
        # TODO: if i % label_update == 0 and i > first_update:
        if batch_idx is None:
            self.p[:] = new_p[:]
            self.y_new[:] = numpy.argmax(new_p, axis=1)
        else:
            self.p[batch_idx, :] = new_p
            self.y_new[batch_idx] = numpy.argmax(new_p, axis=1)

    def mislabel(self, n, strategy='uniform', seed=0, compat_consume_random_twice=True):
        y_mislabel, mislabel_idx = mislabel(self.y_new, n, strategy, seed, compat_consume_random_twice)
        self.y_mislabel = y_mislabel
        self.mislabel_idx = mislabel_idx
        self.y_new = self.y_mislabel.copy()
        self.p = self.smoothed_probabilities(self.n_samples, self.n_classes, self.y_new, self.D)



def mislabel(y, n, strategy='uniform', seed=0, compat_consume_random_twice=True):
    '''
    Changes class labels to others (not equal ones); uniformly or proportionally to class frequency
    @param y: array-like (n_classes,), class labels
    @param n: int, count of class labels to change
    @param strategy: str 'uniform'|'proportional', how to pick new class label
    @param seed: int, numpy random seed
    @return: tuple (y, mislabel_idx), new changed class labels, indices of changed class labels
    '''
    assert strategy in ['uniform', 'proportional'], f'strategy {strategy} is not supported'

    n_samples = y.shape[0]
    n_classes = max(y) + 1

    rnd = numpy.random.RandomState(seed)
    mislabel_idx = rnd.choice(n_samples, n, replace=False)

    if compat_consume_random_twice:
        # TOFIX: to reproduce
        rand_idx = rnd.choice(n_samples, n, replace=False)

    p = numpy.zeros(n_classes)
    if strategy == 'uniform':
        pass #p[:] = 1.0 / n_classes
    elif strategy == 'proportional':
        unique, counts = numpy.unique(y, return_counts=True)
        p[unique] = counts.astype(numpy.float64) / n_samples

    y_result = numpy.zeros_like(y)
    y_result[:] = y[:]
    for i in mislabel_idx:
        while True:
            if strategy == 'uniform':
                new_class = rnd.choice(n_classes)
            elif strategy == 'proportional':
                new_class = rnd.choice(n_classes, p=p)
            if new_class != y[i]:
                y_result[i] = new_class
                break

    return y_result, mislabel_idx



def print_accuracy_vs_mislabeling(y_true, y, mislabel_idx=[], y_mislabel=None, prints=True, returns=False):
    '''
    Prints comparison table of label equality, separately for all and mislabelled indices
    @param y_true: array-like (n_samples,), ground truth labels
    @param y: array-like (n_samples,), predicted labels
    @param mislabel_idx: array-like (n_mislabel), indices of mislabelled samples
    @param y_mislabel: array-like (n_samples,), class labels after mislabelling if it was performed, None otherwise
    @param prints: bool, to print report
    @param returns: bool, to return report data as dict
    @return: dict, report data, see result; None if returns=False
    '''
    import sklearn.metrics

    ok = numpy.sum(y == y_true)
    fail = numpy.sum(y != y_true)
    ok_mislabels = numpy.sum(y[mislabel_idx] == y_true[mislabel_idx])
    fail_mislabels = numpy.sum(y[mislabel_idx] != y_true[mislabel_idx])
    if y_mislabel is not None:
        still_fail_mislabels = numpy.sum((y[mislabel_idx] != y_true[mislabel_idx]) & (y[mislabel_idx] != y_mislabel[mislabel_idx]))
        total_changed = numpy.sum(y != y_mislabel)
    else:
        still_fail_mislabels = None
        total_changed = numpy.sum(y != y_true)
    ok_truelabels = ok - ok_mislabels
    fail_truelabels = fail - fail_mislabels
    n_mislabels = ok_mislabels + fail_mislabels
    n_truelabels = ok_truelabels + fail_truelabels
    assert n_mislabels == len(mislabel_idx)

    accuracy_table = [
        [ok_truelabels, ok_mislabels],
        [fail_truelabels, fail_mislabels],
    ]
    (tn, tp), (fp, fn) = accuracy_table
    f1_score = 2.0 * tp / (2 * tp + fp + fn)
    accuracy_score = float(tn + tp) / (tn + tp + fn + fp)
    y_pred_md5 = md5(''.join(map(str, y)))

    result = {
        'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y),
        'table': accuracy_table,
        'accuracy': accuracy_score,
        'f1': f1_score,
        'adjusted_rand': sklearn.metrics.adjusted_rand_score(y_true, y),
        'mislabels_changed_but_still_fail': still_fail_mislabels,
        'changed_total': total_changed,
        'y_pred_md5': y_pred_md5,
    }

    if prints:
        print(result['confusion_matrix'])
        print()
        print( '      True labels  Mislabels')
        print( '      -----------  ---------')
        print(f'OK    {ok_truelabels:-11d}  {ok_mislabels:-9d}  {ok}')
        print(f'Fail  {fail_truelabels:-11d}  {fail_mislabels:-9d}  {fail}')
        print(f'      {n_truelabels:-11d}  {n_mislabels:-9d}')
        if y_mislabel is not None:
            print(f'Mislabels changed but still fail = {still_fail_mislabels}')
        print(f'Changed total = {result["changed_total"]}')
        print(f'Accuracy = {result["accuracy"]}')
        print(f'F1 score = {result["f1"]}')
        print(f'Adj. Rand score = {result["adjusted_rand"]}')
        print()
        print(f'MD5 of y_pred = {result["y_pred_md5"]}')

    if returns:
        return result



class ADEBatchCallback(object):
    def __init__(self):
        pass
    def __call__(self, ibatch, batch_idx, net, output_device, output_cpu, loss):
        pass
    def __or__(self, rhs):
        pass



class ADELabelUpdaterAllSamples(ADEBatchCallback):
    '''
    Updates dataset class probabilities according to ADE scheme
    as in Zeng, Xinchuan & Martinez, Tony. (2001). An algorithm for correcting mislabeled data. Intell. Data Anal.. 5. 491-502. doi:10.3233/IDA-2001-5605
    @param dataset: dataset that able to update probabilities
    @param l_p: float, learning rate
    '''
    def __init__(self, dataset, first_update, label_update, l_p, start_update_U_after_first_update=False, collect_train_process = False, prints=True):
        super().__init__()
        self.dataset = dataset
        self.first_update = first_update
        self.label_update = label_update
        self.l_p = l_p
        self.start_update_U_after_first_update = start_update_U_after_first_update
        self.collect_train_process = collect_train_process
        self.prints = prints

        #self.scd_p = self.dataset.smoothed_probabilities(dataset.n_samples, dataset.n_classes, dataset.y, dataset.D)
        self.scd_U = numpy.log(self.dataset.p)

        self.train_process = []

    def __call__(self, ibatch, batch_idx, net, output_device, output_cpu, loss):
        import scipy.special
        import sklearn.metrics

        probs = torch.nn.Softmax(dim=1)(output_device).cpu().detach().numpy()

        batch_v = self.dataset.p[batch_idx]
        batch_u = self.scd_U[batch_idx]

        batch_u2 = batch_u + self.l_p * (probs - batch_v)

        if self.collect_train_process:
            loss = sklearn.metrics.log_loss(
                self.dataset.y_new[batch_idx],
                scipy.special.softmax(output_cpu, axis=1),
                labels=range(self.dataset.n_classes)
            )
            cur = {
                'batch': ibatch,
                'batch_idx': batch_idx.copy(),
                'batch_y': self.dataset.y_new[batch_idx].copy(),
                'parameters': [param.detach().numpy().copy() for param in net.parameters()],
                'logits': output_cpu.copy(),
                'U': batch_u.copy(),
                'probs': probs.copy(),
                'V': batch_v.copy(),
                'Unew': batch_u2.copy(),
                'allU': self.scd_U.copy(),
                'allV': self.dataset.p.copy(),
                'allP': self.dataset.p.copy(),
                'loss': loss,
            }

            #for k in cur:
            #    refdebug_check(ibatch, k, refdebug[ibatch][k], cur[k])
            self.train_process.append(cur)

        if not self.start_update_U_after_first_update or ibatch > self.first_update:
            batch_u = batch_u2

            # Update V and p
            batch_v = scipy.special.softmax(batch_u, axis=1)
            self.dataset.p[batch_idx] = batch_v
            self.scd_U[batch_idx] = batch_u

        # update labels
        if ibatch % self.label_update == 0 and ibatch > self.first_update:
            if self.prints:
                changed = numpy.where(self.dataset.y_new != numpy.argmax(self.dataset.p, axis=1))[0]
                if len(changed):
                    print(f'batch #{ibatch}', 'changes:', changed)
            self.dataset.update_class_probabilities(self.dataset.p)



def voting_scheme_max_votes(P, y_original, original_if_tie=False):
    n_voters, n_samples, n_classes = P.shape
    y_votes = numpy.argmax(P, axis=2)
    result = numpy.zeros((n_samples), dtype=numpy.uint32)
    for i in range(n_samples):
        bc = numpy.bincount(y_votes[:, i], minlength=n_classes)
        best = numpy.argmax(bc)
        best_votes = numpy.sum(y_votes[:, i] == best)
        if numpy.sum(bc == best_votes) > 1 and (original_if_tie or bc[y_original[i]] == best_votes):
            result[i] == y_original[i]
        else:
            result[i] = best
    return result

def voting_scheme_max_votes_original_if_tie(P, y_original, *args):
    return voting_scheme_max_votes(P, y_original, original_if_tie=True)

def voting_scheme_max_p_product(P, *args):
    return numpy.argmax(numpy.sum(numpy.log(P), axis=0), axis=1)



def scrock(
    X, y,
    D = 0.9,
    l_ps = [0.5, 0.75, 1.0, 1.25, 1.5],
    net_factory = None,
    label_update_factory = None,
    optimizer = 'adam', optimizer_lr = 0.0001, n_epochs = 40, n_batches = None, batch_size = 32, batch_scheme = 'random-shuffle',
    voting_scheme = voting_scheme_max_votes_original_if_tie,
    verbose = False,
    check_data = True,
    seed = 0,
):
    '''
    Runs scROCK ensemble algorithm
    @param X: array-like (n_samples, n_features), feature values for samples
    @param y: array-like (n_samples,), classes of samples, 0..n_classes-1
    @param D: float, probability that label of sample is correct
    @param l_ps: array like of floats, L_p ADE parameters for ensemble
    @param net_factory: callable, accepts (ialgo) for seed, should return torch.nn.Module
    @param label_update_factory: callable, accepts (l_p) for L_p, should return callable to be called every batch
    @param optimizer: see train_dnn
    @param optimizer_lr: see train_dnn
    @param n_epochs: see train_dnn
    @param n_batches: see train_dnn
    @param batch_size: see train_dnn
    @param batch_scheme: see train_dnn
    @param voting_scheme: callable, accepts (P, y_original), where P is array-like (n_algos, n_samples, n_classes) of probability outputs of ensemble algorithms; should return array-like (n_samples) with class labels
    @param verbose: bool, show verbose process output
    @param check_data: bool, to check if input data looks like log1p gene expression levels (non-negative, float, less than 20)
    @param seed: int, seed + ialgo is used as a seed for all parts of scROCK ensemble item
    @return array-like (n_samples,), proposed by scROCK classes of samples
    '''
    if check_data:
        assert numpy.all(X >= 0.0), 'All values in log-normalized X should be non-negative'
        assert numpy.sum((X > 0.0) & (X < 1.0)) > 0, 'At least one value in log-normalized X should be in (0,1)'
        assert numpy.sum(X > 20.0) == 0, f'No value in log-normalized X should be larger than 20, found {numpy.max(X)}'

    assert set(y) <= set(range(100)), 'y should contain integer class numbers in range [0,100['
    assert X.shape[0] == y.shape[0], f'X should contain the same number of samples as y, but X.shape = {X.shape}, y.shape = {y.shape}'

    n_algos = len(l_ps)
    y_proba = None
    for ialgo, l_p in enumerate(l_ps):
        seed_ialgo = seed + ialgo
        if verbose:
            import time
            t0 = time.time()
            print(f'Run ADE with L_p = {l_p}')

        dataset = DatasetWithMutableLabels(X, y, D = D, seed = seed_ialgo)

        if y_proba is None:
            y_proba = numpy.zeros((n_algos, dataset.n_samples, dataset.n_classes))

        if net_factory is None:
            net = MLPWithLinearOutput(
                dataset.n_features, dataset.n_classes,
                layers = [32, 16], nonlinearity = 'relu', batch_norm = False,
                seed = seed_ialgo
            )
        else:
            net = net_factory(seed_ialgo)

        if label_update_factory is None:
            label_update = ADELabelUpdaterAllSamples(
                dataset,
                first_update = 300, label_update = 1,
                l_p = l_p,
                start_update_U_after_first_update = True,
                prints = verbose,
            )
        else:
            label_update = label_update_factory(l_p)

        train_dnn(
            net,
            dataset, dataset_test = None,
            optimizer = optimizer, optimizer_lr = optimizer_lr,
            n_epochs = n_epochs, n_batches = n_batches,
            batch_size = batch_size, batch_scheme = batch_scheme,
            verbose = verbose,
            on_each_batch = label_update,
            seed = seed_ialgo,
        )

        y_proba[ialgo, :, :] = dataset.p

        if verbose:
            print(f'Changed {numpy.sum(dataset.y_new != dataset.y)} class labels')
            print(f'Run ADE with L_p = {l_p} done in {time.time() - t0:.3f} s')
            print()
    
    return voting_scheme(y_proba, y)



if __name__ == '__main__':
    pass #main()
