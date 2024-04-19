import numpy
import scipy.special

import torch
from tqdm import tqdm



class DNNWithLogitOutput(torch.nn.Module):
    '''
    Generic MLP
    @param n_features int number of features on input
    @param n_classes int number of classes in multiclass classification task
    @param layers array number of neurons in hidden layers
    @param nonlinearity str activation function
    @param batch_norm bool to use BatchNorm layer after every linear except last one
    '''
    def __init__(self, n_features, n_classes, layers = [32, 16], nonlinearity='gelu', batch_norm=True):
        super(DNNWithLogitOutput, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        layer_sizes = [self.n_features] + layers + [self.n_classes]
        self.layers = []
        for i, (layer_size_m, layer_size_n) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append(torch.nn.Linear(layer_size_m, layer_size_n))
            if i + 2 < len(layer_sizes):
                if batch_norm:
                    self.layers.append(torch.nn.BatchNorm1d(layer_size_n))
                if nonlinearity == 'gelu':
                    self.layers.append(torch.nn.GELU())
                elif nonlinearity == 'relu':
                    self.layers.append(torch.nn.ReLU())
                else:
                    assert False, f'{nonlinearity} is not supported'
        self.layers = torch.nn.ParameterList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



def train_dnn(net, dataset, dataset_test=None, optimizer='nadam', n_epochs=100, batch_size=32, verbose=False, on_each_batch=None, seed=0):
    '''
    Trains DNN on dataset
    @param net torch.Module DNN to train
    @param dataset 
    '''
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if verbose:
        print(net)
    if optimizer == 'nadam':
        optimizer = torch.optim.NAdam(net.parameters(), lr=0.0001)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    elif optimizer == 'adamax':
        optimizer = torch.optim.Adamax(net.parameters(), lr=0.0001)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0001)
    else:
        assert False, f'{optimizer} is not supported'
    lossfn = torch.nn.CrossEntropyLoss()

    losses_train = []
    losses_test = []
    n_batches = dataset.n_samples // batch_size
    for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
        losses_epoch = []
        for ibatch in range(n_batches):
            optimizer.zero_grad()
            batch_idx, X_batch, y_batch = dataset.batch(batch_size)
            logits_train = net(torch.Tensor(X_batch).to(device))
            loss_train = lossfn(logits_train, torch.Tensor(y_batch).to(torch.long).to(device))

            if on_each_batch is not None:
                on_each_batch(net, batch_idx, logits_train.cpu().detach().numpy(), loss_train.cpu().detach().numpy())

            loss_train.backward()
            optimizer.step()
            losses_epoch.append(loss_train.cpu().detach().numpy())

        losses_train.append(numpy.mean(losses_epoch))
        loss_test = numpy.nan
        if dataset_test is not None:
            # TODO: batchwise
            logits_test = net(torch.Tensor(dataset_test.X).to(device))
            loss_test = lossfn(logits_test, torch.Tensor(dataset_test.y).to(torch.long).to(device))
        losses_test.append(loss_test.cpu().detach().numpy())

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
        @param X array-like (n_samples, n_features) feature values for samples
        @param y array-like (n_samples,) Classes of samples, 0..n_classes-1
        @param D float probability that label of sample is correct
        @param seed int random seed
        '''
        self.X = X
        self.y = y
        self.seed = seed
        numpy.random.seed(self.seed)
        self.n_samples, self.n_features = X.shape
        self.n_classes = max(y) + 1

        self.p = numpy.zeros((self.n_samples, self.n_classes))
        D_others = (1 - D) / (self.n_classes - 1)
        for i in range(self.n_samples):
            self.p[i, :] = D_others
            self.p[i, self.y[i]] = D

    def batch(self, batch_size):
        '''
        @param batch_size int size of batch
        @return tuple (indices of batch, X_batch, y_batch)
        '''
        batch_idx = numpy.random.choice(self.n_samples, batch_size, replace=False)
        return batch_idx, self.X[batch_idx, :], self.y[batch_idx]
    
    def update_class_probabilities(self, batch_idx, new_p):
        '''
        @param batch_idx indices of batch
        @param new_p new values of class probabilities
        '''
        # TODO: if i % label_update == 0 and i > first_update:
        self.p[batch_idx, :] = new_p
        self.y[batch_idx] = numpy.argmax(new_p)



def mislabel(y, n, strategy='uniform', seed=0):
    '''
    Changes class labels to others (not equal ones); uniformly or proportionally to class frequency
    @param y array-like (n_classes,) class labels
    @param n int count of class labels to change
    @param strategy str 'uniform'|'proportional', how to pick new class label
    @param seed int random seed
    @return tuple (y, rand_idx) new changed class labels, indices of changed class labels
    '''
    assert strategy in ['uniform', 'proportional']

    n_samples = y.shape[0]
    n_classes = max(y) + 1

    numpy.random.seed(seed)
    mislabel_idx = numpy.random.choice(n_samples, n, replace=False)

    p = numpy.zeros(n_classes)
    if strategy == 'uniform':
        p[:] = 1.0 / n_classes
    elif strategy == 'proportional':
        unique, counts = numpy.unique(y, return_counts=True)
        p[unique] = counts.astype(numpy.float) / n_samples
    else:
        assert False, f'strategy {strategy} is not supported'

    y_result = numpy.zeros_like(y)
    for i in mislabel_idx:
        while True:
            new_class = numpy.random.choice(n_classes, p=p)
            if new_class != y_result[i]:
                y_result[i] = new_class
                break

    return y_result, mislabel_idx



def print_accuracy_vs_mislabeling(y_true, y, mislabel_idx):
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
    def __init__(self, dataset, lr=0.02):
        self.dataset = dataset
        self.lr = lr
    
    def __call__(self, net, batch_idx, logits, loss):
        # Update class probabilities
        probs = scipy.special.softmax(logits)
        batch_u = scipy.special.logit(self.dataset.p[batch_idx])
        batch_u += self.lr * (probs - self.dataset.p[batch_idx])
        self.dataset.update_class_probabilities(batch_idx, scipy.special.softmax(batch_u))



def run_ade(
    dataset, n_mislabeled = 300, first_update = 100, label_update = 10,
    l_p = 1, n_epochs = 40, batch_size = 32, lr = 0.001, layers = [32, 16], 
    verbose=True, seed=0
):
    y_mislabel, mislabel_idx = mislabel(dataset.y, n=n_mislabeled)
    if verbose:
        print_accuracy_vs_mislabeling(dataset.y, y_mislabel, mislabel_idx)

    label_update = ADELabelUpdater(dataset)

    train_dnn(
        DNNWithLogitOutput(dataset.n_features, dataset.n_classes, batch_norm=False),
        dataset,
        n_epochs = n_epochs,
        batch_size = batch_size,
        verbose = verbose,
        on_each_batch = label_update,
        seed = seed,
    )

    return dataset.y



if __name__ == '__main__':
    pass #main()
