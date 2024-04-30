import numpy
import scipy.special

import torch
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MLPWithLinearOutput(torch.nn.Module):
    '''
    Generic MLP
    @param n_features: int, number of features on input
    @param n_classes: int, number of classes in multiclass classification task
    @param layers: array, number of neurons in hidden layers
    @param nonlinearity: str|torch.nn.Module, activation function
    @param batch_norm: bool, to use BatchNorm layer after every linear except last one
    '''
    def __init__(self, n_features, n_classes, layers = [32, 16], nonlinearity='relu', batch_norm=False, seed=0):
        super(MLPWithLinearOutput, self).__init__()
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
    batch_size=32, batch_replace=False,
    verbose=False, on_each_batch=None, seed=0
):
    '''
    Trains DNN on dataset
    @param net: torch.Module, DNN to train
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
    assert type(n_epochs) == int and n_batches is None or n_epochs is None and type(n_batches) == int, 'Only one of n_epochs, n_batches should be not None'

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

    for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
        losses_epoch = []
        for i_epoch_batch in range(n_epoch_batches):
            optimizer.zero_grad()
            batch_idx, X_batch, y_batch = dataset.batch(batch_size, replace=batch_replace)
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

        if verbose:
            import IPython
            from matplotlib.pyplot import title, plot, legend, show, xlabel, ylabel
            IPython.display.clear_output(wait=True)
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



# TODO: torch.utils.data.Dataset
class DatasetWithMutableLabels(object):
    def __init__(self, X, y, D=1.0, seed=0):
        '''
        @param X: array-like (n_samples, n_features), feature values for samples
        @param y: array-like (n_samples,), classes of samples, 0..n_classes-1
        @param D: float, probability that label of sample is correct
        @param seed: int, numpy random seed
        '''
        self.X = X
        self.y = y
        self.y_new = y.copy()
        self.n_samples, self.n_features = X.shape
        self.n_classes = max(y) + 1
        self.D = D
        self.p = self.smoothed_probabilities(self.n_samples, self.n_classes, self.y, self.D)
        self.rnd = numpy.random.RandomState(seed)

    def smoothed_probabilities(self, n_samples, n_classes, y, D):
        p = numpy.zeros((n_samples, n_classes))
        D_others = (1 - D) / (n_classes - 1)
        for i in range(n_samples):
            p[i, :] = D_others
            p[i, y[i]] = D
        return p

    def batch(self, batch_size, replace=False):
        '''
        @param batch_size: int, size of batch
        @param replace: bool, choice batch with replacement (bootstrap) or no (usual NN batches)
        @return: tuple (indices of batch, X_batch, y_batch)
        '''
        batch_idx = self.rnd.choice(self.n_samples, batch_size, replace=replace)
        return batch_idx, self.X[batch_idx, :], self.y_new[batch_idx]
    
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



def mislabel(y, n, strategy='uniform', seed=0):
    '''
    Changes class labels to others (not equal ones); uniformly or proportionally to class frequency
    @param y: array-like (n_classes,), class labels
    @param n: int, count of class labels to change
    @param strategy: str 'uniform'|'proportional', how to pick new class label
    @param seed: int, numpy random seed
    @return: tuple (y, mislabel_idx), new changed class labels, indices of changed class labels
    '''
    assert strategy in ['uniform', 'proportional']

    n_samples = y.shape[0]
    n_classes = max(y) + 1

    rnd = numpy.random.RandomState(seed)
    mislabel_idx = rnd.choice(n_samples, n, replace=False)

    # TOFIX: to reproduce
    rand_idx = rnd.choice(n_samples, n, replace=False)

    p = numpy.zeros(n_classes)
    if strategy == 'uniform':
        pass #p[:] = 1.0 / n_classes
    elif strategy == 'proportional':
        unique, counts = numpy.unique(y, return_counts=True)
        p[unique] = counts.astype(numpy.float) / n_samples
    else:
        assert False, f'strategy {strategy} is not supported'

    y_result = numpy.zeros_like(y)
    y_result[:] = y[:]
    for i in mislabel_idx:
        while True:
            if strategy == 'uniform':
                new_class = rnd.choice(n_classes)
            elif strategy == 'proportional':
                new_class = rnd.choice(n_classes, p=p)
            else:
                assert False
            if new_class != y[i]:
                y_result[i] = new_class
                break

    return y_result, mislabel_idx



def print_accuracy_vs_mislabeling(y_true, y, mislabel_idx):
    '''
    Prints comparison table of label equality, separately for all and mislabelled indices
    @param y_true: array-like (n_samples,), ground truth labels
    @param y: array-like (n_samples,), predicted labels
    @param mislabel_idx: array-like (n_mislabel), indices of mislabelled samples
    '''
    ok = numpy.sum(y == y_true)
    fail = numpy.sum(y != y_true)
    ok_mislabels = numpy.sum(y[mislabel_idx] == y_true[mislabel_idx])
    fail_mislabels = numpy.sum(y[mislabel_idx] != y_true[mislabel_idx])
    ok_truelabels = ok - ok_mislabels
    fail_truelabels = fail - fail_mislabels
    n_mislabels = ok_mislabels + fail_mislabels
    n_truelabels = ok_truelabels + fail_truelabels
    assert n_mislabels == len(mislabel_idx)

    print( '      True labels  Mislabels')
    print( '      -----------  ---------')
    print(f'OK    {ok_truelabels:-11d}  {ok_mislabels:-9d}  {ok}')
    print(f'Fail  {fail_truelabels:-11d}  {fail_mislabels:-9d}  {fail}')
    print(f'      {n_truelabels:-11d}  {n_mislabels:-9d}')



class ADELabelUpdater(object):
    '''
    Updates dataset class probabilities according to ADE scheme
    as in Zeng, Xinchuan & Martinez, Tony. (2001). An algorithm for correcting mislabeled data. Intell. Data Anal.. 5. 491-502. doi:10.3233/IDA-2001-5605
    @param dataset: dataset that able to update probabilities
    @param lr: float, learning rate
    '''
    def __init__(self, dataset, lr=0.02):
        self.dataset = dataset
        self.lr = lr
    
    def __call__(self, ibatch, batch_idx, net, output_device, output_cpu, loss):
        # Update class probabilities
        probs = scipy.special.softmax(output_cpu)
        # TOFIX
        #batch_u = scipy.special.logit(self.dataset.p[batch_idx])
        batch_u = numpy.log(self.dataset.p[batch_idx])
        batch_u += self.lr * (probs - self.dataset.p[batch_idx])
        p = scipy.special.softmax(batch_u)
        self.dataset.update_class_probabilities(p, batch_idx)



def run_ade(
    dataset, n_mislabel = 300, mislabel_seed=0, #batch_seed=0,
    #first_update = 100, label_update = 10, l_p = 1,
    net = None,
    label_update = None,
    #layers = [32, 16], nonlineariry = 'relu', batch_norm = False, mlp_seed = 0,
    optimizer = 'adam', optimizer_lr = 0.0001, n_epochs = 40, n_batches = None, batch_size = 32, batch_replace = False, train_dnn_seed = 0,
    verbose=True,
):
    '''
    Runs all ADE pipeline
    @param X: array-like (n_samples, n_features), input data features
    @param y: array-like (n_samples,), input data class labels
    @param D: float, see `DatasetWithMutableLabels`
    @param n_mislabel: int, see `n` in `mislabel`
    @param mislabel_seed: int, see `seed` in `mislabel`
    @param batch_seed: int, see `seed` in `DatasetWithMutableLabels`
    @param first_update: int, see `ADELabelUpdater`
    @param label_update: int, see `ADELabelUpdater`
    @param l_p: float, see `lr` in `ADELabelUpdater`
    @param layers: array-like (n_layers,), see `MLPWithLinearOutput`
    @param nonlinearity: str|torch.nn.Module, see `MLPWithLinearOutput`
    @param batch_norm: bool, see `MLPWithLinearOutput`
    @param mlp_seed: int, see `seed` in `MLPWithLinearOutput`
    @param optimizer: str|torch.optim.Optimizer, see `train_dnn`
    @param optimizer_lr: see `train_dnn`
    @param n_epochs: int|None, see `train_dnn`
    @param n_batches: int|None, see `train_dnn`
    @param batch_size: int, see `train_dnn`
    @param batch_replace: bool, see `train_dnn`
    @param train_dnn_seed: int, see `seed` in `train_dnn`
    @param verbose: bool, see `train_dnn`, also outputs mislabelling stats
    @return: tuple (y_new, mislabel_idx), new sample class labels after ADE, indices of initially mislabelled class labels
    '''
    assert type(n_epochs) == int and n_batches is None or n_epochs is None and type(n_batches) == int, 'Only one of n_epochs, n_batches should be not None'

    y_mislabel, mislabel_idx = mislabel(dataset.y, n=n_mislabel, seed=mislabel_seed)
    if verbose:
        print_accuracy_vs_mislabeling(dataset.y, y_mislabel, mislabel_idx)
    dataset.y = y_mislabel
    dataset.p = dataset.smoothed_probabilities(dataset.n_samples, dataset.n_classes, dataset.y, dataset.D)

    # TOFIX: first_update label_update
    #label_update = ADELabelUpdater(dataset, lr=l_p)

    '''
    net = MLPWithLinearOutput(
        dataset.n_features, dataset.n_classes,
        layers = layers, nonlinearity = nonlineariry, batch_norm = batch_norm,
        seed = mlp_seed,
    )
    '''

    train_dnn(
        net,
        dataset, dataset_test = None,
        optimizer = optimizer, optimizer_lr = optimizer_lr,
        n_epochs = n_epochs, n_batches = n_batches,
        batch_size = batch_size,
        batch_replace = batch_replace,
        verbose = verbose,
        on_each_batch = label_update,
        seed = train_dnn_seed,
    )

    return dataset.y_new, mislabel_idx



if __name__ == '__main__':
    pass #main()
