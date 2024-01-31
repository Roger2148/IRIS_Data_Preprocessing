import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
# from numpy import linalg


class ESN:
    def __init__(self, input_size, output_size, population_size=10, sparsity=0.9,
                 spectral_radius=1.0, leaking_rate=None, input_scaling=1):

        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size

        # adaptation speed (if leaking rate is given, use leaking rate instead)
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        # weight initialization
        self.weight_range_in = input_scaling  # U[-x, +x] # input scaling???
        self.weight_range = 2.0  # U[-0.5weight_range, +0.5weight_range]
        self.W_in = torch.rand(self.population_size, self.input_size)*self.weight_range_in - 0.5*self.weight_range_in
        # self.W_in = torch.rand(self.population_size, self.input_size)*weight_range_in

        self.W_pop = torch.rand(self.population_size, self.population_size)*self.weight_range - 0.5*self.weight_range
        # self.W_pop = torch.mul(self.W_pop, ~torch.eye(self.population_size, dtype=torch.bool))

        # sparsity
        self.sparsity = sparsity
        # if torch.cuda.is_available():
        #     self.W_pop =  self.W_pop * \
        #                   (torch.rand(self.W_pop.shape, device='cuda:0') < (1 - self.sparsity))
        # else:
        self.W_pop =  self.W_pop * \
                      (torch.rand(self.W_pop.shape) < (1 - self.sparsity)) #bug

        # ensure fading memory
        self.W_pop = self.W_pop.cpu()
        rhoW = max(abs(linalg.eig(self.W_pop)[0]))
        self.W_pop = self.W_pop * (self.spectral_radius / rhoW)

        if torch.cuda.is_available():
            self.W_in = self.W_in.cuda()
            self.W_pop = self.W_pop.cuda()

        # other
        self.ctrl_population_size = 0

    def forward(self, u_in, x_current_state):
        x_next = self.linear(u_in, x_current_state, W_pop=None)
        x_next = torch.tanh(x_next)
        x_next = (1-self.leaking_rate[:, None]) * x_current_state + self.leaking_rate[:, None] * x_next
        return x_next

    def linear(self, u_in, x_state, W_pop=None):
        if W_pop is None:
            W_pop = self.W_pop
        x_state_next = torch.matmul(self.W_in, u_in) + torch.matmul(W_pop, x_state) # bug
        return x_state_next

    # def widow_hoff(self, x_current, x_previous):
    #     Ins_pre = x_previous
    #     a = x_current
    #     Ins_cur = Ins_pre + self.adaptation_speed * (a - Ins_pre)
    #     return Ins_cur

    def create_sub_network(self, n=5*2, sparsity=None, leaking_rate=None, spectral_radius=None):
        sub_size = int(self.population_size/n)

        if sparsity is None:
            sparsity = self.sparsity
        elif isinstance(sparsity, int):
            pass
        elif isinstance(sparsity, list):
            # sparsity = np.array(sparsity)
            sparsity = np.diag(sparsity)
            sparsity = np.repeat(sparsity, sub_size, axis=1).repeat(sub_size, axis=0)

        if leaking_rate is None:
            pass
        elif isinstance(leaking_rate, int):
            self.leaking_rate = leaking_rate
        elif isinstance(leaking_rate, list):
            # leaking_rate = np.diag(leaking_rate)
            # leaking_rate = np.repeat(leaking_rate, sub_size, axis=1).repeat(sub_size, axis=0)
            leaking_rate = np.array(leaking_rate).repeat(sub_size)
            self.leaking_rate = torch.tensor(leaking_rate, device='cuda:0' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)

        if spectral_radius is None:
            pass
        else:
            self.spectral_radius = spectral_radius
        # elif isinstance(spectral_radius, int):
        #     self.spectral_radius = spectral_radius
        # elif isinstance(spectral_radius, list):
        #     spectral_radius = np.diag(spectral_radius)
        #     self.spectral_radius = np.repeat(spectral_radius, sub_size, axis=1).repeat(sub_size, axis=0)

        W = np.random.rand(self.population_size, self.population_size)*self.weight_range - 0.5*self.weight_range
        W_mask = np.kron(np.eye(n), np.ones((sub_size, sub_size)))
        W = W * W_mask
        W =  W * (np.random.rand(*W.shape) < (1 - sparsity))

        # ensure fading memory
        for i in range(n):
            start = sub_size * i
            ends = sub_size * (i+1)
            sub_W = W[start:ends, start:ends] + 0
            rhoW = max(abs(linalg.eig(sub_W)[0]))
            if isinstance(self.spectral_radius, list):
                sub_W = sub_W * (self.spectral_radius[i] / rhoW)
            else:
                sub_W = sub_W * (self.spectral_radius / rhoW)
            W[start:ends, start:ends] = sub_W

        self.W_pop = torch.tensor(W, device='cuda:0' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)


        # temp_state_vector = np.repeat(temp_state_vector, 2, axis=1).repeat(2, axis=0)
        # create a 3-by-3 array
        # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # # create a 30-by-30 diagonal array with ten copies of a on the diagonal
        # b = np.kron(np.eye(10), a)

    # def multi_leaking_rate(self, lr=):

    @staticmethod
    def PCA_svd(X, k, center=True):
        '''

        :param X: shape(number of sample, number of feature)
        :param k: first k PCs
        :param center:
        :return:
        '''
        X = torch.tensor(X, dtype=torch.float32)
        # X = X.t()
        n = X.size()[0]
        ones = torch.ones(n).view([n, 1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n, n])
        H = torch.eye(n) - h
        if torch.cuda.is_available():
            H = H.cuda()
            X = X.cuda()
        X_center = torch.mm(H.float(), X)
        u, s, v = torch.svd(X_center)
        components = v[:, :k].t()
        return components
