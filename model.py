from torch import tensor as T
import torch
import numpy as np


class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
    '''

    def __init__(self, nvertex, out_dim, ndocs, nlabels, lr=0.05):
        '''
        Parameters specs:
            nvertex : total number of vertices in the graph
            out_dim : node vector dimension
            ndocs : number of documents in the corpus
            nlabels : number of labels
            lr : learning rate.
        '''
        eps = np.sqrt(1.0 / float(out_dim))
        self.w = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                            dtype=torch.Float32)
        self.d = np.asarray(np.random.uniform(low=-eps, high=eps, size=(ndocs, out_dim)),
                            dtype=torch.Float32)
        self.l = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nlabels, out_dim)),
                            dtype=torch.Float32)
        self.W = torch.shared(self.w, name='W', borrow=True)
        self.D = torch.shared(self.d, name='D', borrow=True)
        self.L = torch.shared(self.l, name='L', borrow=True)
        self.lr = lr

    def ww_model(self):
        '''
        Performs SGD update (pre-training on ww graph).
        '''
        indm = torch.scalar_tensor()
        indc = torch.scalar_tensor()
        indr = torch.tensor()  # vector of 5 negative edge samples
        Uj = self.W[indm, :]  # one row of W
        Ui = self.W[indc, :]  # one row of W
        Ui_Set = self.W[indr, :]  # rows for negative edge sampling
        cost_ww = torch.log(torch.sigmoid(torch.dot(Ui, Uj)))
        cost_ww -= torch.log(torch.sum(torch.sigmoid(torch.sum(Uj * Ui_Set, axis=1))))

        cost = -cost_ww
        # gradient w.r.t 3 variables
        grad_ww = torch.grad(cost, [Uj, Ui, Ui_Set])
        deltaW = torch.sub(self.W[indm, :], - (self.lr) * grad_ww[0])
        deltaW = torch.sub(deltaW[indc, :], - (self.lr) * grad_ww[1])
        deltaW = torch.sub(deltaW[indr, :], - (self.lr) * grad_ww[2])
        updateD = [(self.W, deltaW)]
        self.train_ww = torch.function(
            inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_ww(self, indm, indc, indr):
        return self.train_ww(indm, indc, indr)

    def wd_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        indm = torch.scalar_tensor()
        indc = torch.scalar_tensor()
        indr = torch.tensor()  # vector of 5 negative edge samples
        Uj = self.D[indm, :]  # one row of D
        Ui = self.W[indc, :]  # one row of W
        Ui_Set = self.W[indr, :]  # rows of W for negative edge sampling
        cost_wd = torch.log(torch.sigmoid(torch.dot(Ui, Uj)))
        cost_wd -= torch.log(torch.sum(torch.sigmoid(torch.sum(Uj * Ui_Set, axis=1))))
        cost = -cost_wd
        # gradient w.r.t 3 variables
        grad_wd = torch.gradient(cost, [Uj, Ui, Ui_Set])

        deltaD = torch.sub(self.D[indm, :], grad_wd[0], alpha=(self.lr))
        deltaW = torch.sub(self.W[indc, :], grad_wd[1], alpha=(self.lr))
        deltaW = torch.sub(deltaW[indr, :], grad_wd[2], alpha=(self.lr))
        updateD = [(self.W, deltaW), (self.D, deltaD)]
        self.train_wd = torch.function(
            inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_wd(self, indm, indc, indr):
        return self.train_wd(indm, indc, indr)

    def wl_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        indm = torch.scalar_tensor()
        indc = torch.scalar_tensor()
        indr = torch.tensor()  # vector of 5 negative edge samples
        Uj = self.L[indm, :]  # one row of L
        Ui = self.W[indc, :]  # one row of W
        Ui_Set = self.W[indr, :]  # rows of W for negative edge sampling
        cost_wl = torch.log(torch.sigmoid(torch.dot(Ui, Uj)))
        cost_wl -= torch.log(torch.sum(torch.sigmoid(torch.sum(Uj * Ui_Set, axis=1))))
        cost = -cost_wl
        # gradient w.r.t 3 variables
        grad_wl = torch.gradient(cost, [Uj, Ui, Ui_Set])

        deltaL = torch.sub(self.L[indm, :], grad_wl[0], alpha=(self.lr))
        deltaW = torch.sub(self.W[indc, :], grad_wl[1], alpha=(self.lr))
        deltaW = torch.sub(deltaW[indr, :], grad_wl[2], alpha=(self.lr))
        updateD = [(self.W, deltaW), (self.L, deltaL)]
        self.train_wl = torch.function(
            inputs=[indm, indc, indr], outputs=cost, updates=updateD)

    def pretraining_wl(self, indm, indc, indr):
        return self.train_wl(indm, indc, indr)

    def save_model(self, file_name):
        W = self.W.get_value()
        np.save(file_name, W)
