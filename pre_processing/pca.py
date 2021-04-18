import os
import numpy as np
import torch
import torch.nn.functional as F

class PCA():
    def __init__(self, n_components=1024, whitening=True,
                 parameters_path='models/pca_params_vcdb997090_resnet50_rmac_3840.npz'):
        self.n_components = n_components
        self.whitening = whitening
        self.parameters_path = parameters_path

    def train(self, x):
        '''training pca.
        Args:
            x: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        '''

        x = x.t()
        nPoints = x.size(1)
        nDims = x.size(0)

        # x = x.double()
        mu = x.mean(1).unsqueeze(1)
        x = x - mu

        if (nDims <= nPoints):
            doDual = False
            x2 = torch.matmul(x, x.t()) / (nPoints - 1)
        else:
            doDual = True
            x2 = torch.matmul(x.t(), x) / (nPoints - 1)

        L, U = torch.symeig(x2, eigenvectors=True)
        if (self.n_components < x2.size(0)):
            k_indices = torch.argsort(L, descending=True)[:self.n_components]
            L = torch.index_select(L, 0, k_indices)
            U = torch.index_select(U, 1, k_indices)

        lams = L
        lams[lams < 1e-9] = 1e-9

        if (doDual):
            U = torch.matmul(x, torch.matmul(U, torch.diag(1. / torch.sqrt(lams)) / np.sqrt(nPoints - 1)))

        Utmu = torch.matmul(U.t(), mu)

        U, lams, mu, Utmu = U.numpy(), lams.numpy(), mu.numpy(), Utmu.numpy()

        print('================= PCA RESULT ==================')
        print('U: {}'.format(U.shape))
        print('lams: {}'.format(lams.shape))
        print('mu: {}'.format(mu.shape))
        print('Utmu: {}'.format(Utmu.shape))
        print('===============================================')

        # save features, labels to h5 file.
        filename = os.path.join(self.parameters_path)
        np.savez(filename, U=U, lams=lams, mu=mu, Utmu=Utmu)

    def load(self):
        print('loading PCA parameters...')
        pca = np.load(self.parameters_path)
        U = pca['U'][...][:, :self.n_components]
        lams = pca['lams'][...][:self.n_components]
        mu = pca['mu'][...]
        Utmu = pca['Utmu'][...]

        if (self.whitening):
            U = np.matmul(U, np.diag(1./np.sqrt(lams)))
        Utmu = np.matmul(U.T, mu)

        self.weight = torch.from_numpy(U.T).view(self.n_components, -1, 1, 1).float()
        self.bias = torch.from_numpy(-Utmu).view(-1).float()

    def infer(self, data):
        '''apply PCA/Whitening to data.
        Args:
            data: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        Returns:
            output: [N, output_dim] FloatTensor with output of PCA/Whitening operation.
        '''

        N, D = data.size()
        data = data.view(N, D, 1, 1)
        if torch.cuda.is_available():
            output = F.conv2d(data, self.weight.cuda(), bias=self.bias.cuda(), stride=1, padding=0).view(N, -1)
        else:
            output = F.conv2d(data, self.weight, bias=self.bias, stride=1, padding=0).view(N, -1)

        output = F.normalize(output, p=2, dim=-1) # IMPORTANT!
        assert (output.size(1) == self.n_components)
        return output