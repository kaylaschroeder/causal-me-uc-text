# Steps following the pseudocode and the code base
# The file trainer.py in miv/models/MerrorKIV contains overarching process & steps, but not all necessary functions
import torch
import numpy as np
from scipy.spatial.distance import cdist

# Generate some data to be used as our variables
# Following sigmoid_design.py in the data folder

# NOTE: can try increasing the dimensions later to see if it can handle larger dimensions already

def f(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(16 * x - 8) + 1) * np.sign(x - 0.5)

# Train data set
mu = np.zeros((3,))
# mu = torch.zeros((3,))
sigma = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
# sigma = torch.tensor([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
# torch.distributions.MultivariateNormal(torch.zeros((3,)),\
#                                        torch.tensor([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])).sample()
from numpy.random import default_rng
# rng = default_rng(seed=rand_seed)
rng = default_rng(seed=1234)
data_size = 50
utw = rng.multivariate_normal(mu, sigma, size=data_size*N.shape[1])
u = utw[:, 0:1]
from scipy import stats
z = stats.norm.cdf(utw[0:data_size, 2])[:, np.newaxis]
Z=z
x = stats.norm.cdf(utw[:, 1] + utw[:, 2] / np.sqrt(2))[:, np.newaxis]
# x = z + rng.normal(0, 0.01, data_size)[:, np.newaxis]
X_hidden=x.reshape(-1, N.shape[1])
structural = f(x)
Y_struct=structural
outcome = f(x) + u
Y=outcome
# Let's use gaussian error
data_size = X_hidden.shape[0]
std_X = np.std(X_hidden)
# Select scale_m and scale_n
scale_m = 0.25
scale_n = 1
std_M, std_N = std_X * scale_m, std_X * scale_n
M = X_hidden + std_M * np.random.normal(0, 1, data_size)[:, np.newaxis]
N = X_hidden + std_N * np.random.normal(0, 1, data_size)[:, np.newaxis]
covariate = None
X_obs = None
# Test data set
x_test = np.linspace(0, 1, 100)
y_test = f(x_test)
X_all_test = x_test[:, np.newaxis]
Y_struct_test = y_test[:, np.newaxis]
covariate_test = None


# Define training parameters per the model of choice
# Training parameters per the yaml MerrorKIV_sigmoid.yaml
train_params = {'split_ratio': 0.5,
                'lambda_mn': [0, -10],
                'lambda_n': [0, -10],
                'xi': [0, -10],
                'lambda_x': None,
                'n_chi': 500,
                'Chi_lim': [-0.5, 0.5],
                'label_cutoff': 1.0,
                'reg_param': 0.,
                'batch_size': 64,
                'lr': 0.1,
                'num_epochs': 5}

# ***BEGIN STAGE 1***
# Split training into 2 data sets
from sklearn.model_selection import train_test_split
trainset1_idx, trainset2_idx = train_test_split(np.arange(X_hidden.shape[0]),
                                                test_size = train_params['split_ratio'],
                                                random_state = 1234)
Z_trainset1 = Z[trainset1_idx] ; Z_trainset2 = Z[trainset2_idx]
M_trainset1 = M[trainset1_idx] ; M_trainset2 = M[trainset2_idx]
N_trainset1 = N[trainset1_idx] ; N_trainset2 = N[trainset2_idx]
X_hidden_trainset1 = X_hidden[trainset1_idx] ; X_hidden_trainset2 = X_hidden[trainset2_idx]
Y_trainset1 = Y[trainset1_idx] ; Y_trainset2 = Y[trainset2_idx]

# 2: Obtain lambda and gamma via stage1_tuning function (trainer.py file)
# Initialize lambda mn and lambda n
# Evenly spaced values (50 of them) across the range of lambda
lambda_n = np.exp(np.linspace(train_params['lambda_n'][0], train_params['lambda_n'][1], 50))
lambda_mn = np.exp(np.linspace(train_params['lambda_mn'][0], train_params['lambda_mn'][1], 50))

# Obtain MN (concatenate along second axis)
MN_trainset1 = np.c_[M_trainset1, N_trainset1] ; MN_trainset2 = np.c_[M_trainset2, N_trainset2]

# Preliminaries
sigmaN = np.median(cdist(N_trainset1, N_trainset1, "sqeuclidean"))
sigmaMN = np.median(cdist(MN_trainset1, MN_trainset1, "sqeuclidean"))
sigmaZ = np.median(cdist(Z_trainset1, Z_trainset1, "sqeuclidean"))

# Compute RBF Kernels (similarity between vectors/points)
KZ1Z1 = np.exp(-cdist(Z_trainset1, Z_trainset1, "sqeuclidean") / 2 / float(sigmaZ))
# torch: KZ1Z1 = torch.exp(-torch.cdist(Z1, Z1, "sqeuclidean") / 2 / float(sigmaZ))
KZ1Z2 = np.exp(-cdist(Z_trainset1, Z_trainset2, "sqeuclidean") / 2 / float(sigmaZ))
KN1N1 = np.exp(-cdist(N_trainset1, N_trainset1, "sqeuclidean") / 2 / float(sigmaN))
KN1N2 = np.exp(-cdist(N_trainset1, N_trainset2, "sqeuclidean") / 2 / float(sigmaN))
KMN1MN1 = np.exp(-cdist(MN_trainset1, MN_trainset1, "sqeuclidean") / 2 / float(sigmaMN))
KMN1MN2 = np.exp(-cdist(MN_trainset1, MN_trainset2, "sqeuclidean") / 2 / float(sigmaMN))

# Calculation
n = Z_trainset1.shape[0]
# N
gamma_list = [np.linalg.solve(KZ1Z1 + n * lam1 * np.eye(n), KZ1Z2) for lam1 in lambda_n]
score = [np.trace(gamma.T.dot(KN1N1.dot(gamma)) - 2 * KN1N2.T.dot(gamma)) for gamma in gamma_list]
lambda_n = lambda_n[np.argmin(score)] # Ridge regression hyperparameter
gamma_n = gamma_list[np.argmin(score)] # Eq 22 in paper
# MN
gamma_list = [np.linalg.solve(KZ1Z1 + n * lam1 * np.eye(n), KZ1Z2) for lam1 in lambda_mn]
score = [np.trace(gamma.T.dot(KMN1MN1.dot(gamma)) - 2 * KMN1MN2.T.dot(gamma)) for gamma in gamma_list]
lambda_mn = lambda_mn[np.argmin(score)] # Ridge regression hyperparameter
gamma_mn = gamma_list[np.argmin(score)] # Eq 24 in paper

# ***END STAGE 1***



# ***BEGIN Merror STAGE***

# **1**
# The code uses the below function which we instead replace with its entire functionality
# stageM_data = create_stage_M_raw_data(self.n_chi, N1, M1, Z2, gamma_n, gamma_mn, sigmaN, KZ1Z2)

# The below calculations to get the `raw_labels` are used to select indices from Chi and Z
# Labels are used for comparison in the loss functionality to compare to the predicted labels (wX) and obtain MSE
# Thus, labels are wMN(alpha, Z) in equation 20
# The labels are basically considered to be ground truth
Chi_n = np.random.normal(0, 1, train_params['n_chi']* (N_trainset1.shape[1]))
Chi_n = Chi_n / 2 / np.pi / sigmaN ** 0.5 # because the computed sigmaN is actually sigma^2N
Chi_n = Chi_n.reshape(train_params['n_chi'],N_trainset1.shape[1])
n, m = KZ1Z2.shape
# Columns of Chi are repeated to account for the data size of the variable
cos_term = np.cos(Chi_n @ N_trainset1.T)  # shape: Chi.shape[0] x args.train.N.shape[0]
sin_term = np.sin(Chi_n @ N_trainset1.T)
# Real (cos) and imaginary (sin) parts; dot products with gamma N
denom = cos_term.dot(gamma_n) + sin_term.dot(gamma_n) * 1j
# Component shape: Chi.shape[0] x args.dev.Z.shape[0]
m_gamma_numer = sum([gamma_mn * M_trainset1[i].to_numpy().reshape(-1,1) for i in range(M_trainset1.shape[1])])
numer = cos_term.dot(m_gamma_numer) + sin_term.dot(m_gamma_numer) * 1j
raw_labels = (numer.to_numpy()/denom.to_numpy()).flatten().reshape(-1,1)
raw_Chi = np.repeat(Chi_n, m).reshape(-1, N_trainset1.shape[1])
# Unseen Z values
raw_Z = np.repeat(Z_trainset2[np.newaxis, :, :], train_params['n_chi'], axis=0).reshape(-1, Z_trainset2.shape[1])
raw_dict = {'labels':raw_labels, 'Chi':raw_Chi, 'Z':raw_Z}

# **2**
# The code uses the below function which we instead replace with its entire functionality
# stageM_data = prepare_stage_M_data(raw_data2=stageM_data, rand_seed=rand_seed)

real_label = np.real(raw_dict['labels']).flatten()
imag_label = np.imag(raw_dict['labels']).flatten()
# Select only values within 1 standard deviation of the real portion of the value and the imaginary portion of the value for our sample
idx_select = (real_label < np.mean(real_label) + 1. * np.std(real_label)) * (
            real_label > np.mean(real_label) - 1. * np.std(real_label)) \
                 * (imag_label < np.mean(imag_label) + 1. * np.std(imag_label)) * (
                         imag_label > np.mean(imag_label) - 1. * np.std(imag_label))
raw_labels = raw_dict['labels'][idx_select]
raw_Chi = raw_dict['Chi'][idx_select] # alpha j sample
raw_Z = raw_dict['Z'][idx_select] # zj sample
shuffle_idx = np.arange(raw_Z.shape[0])
default_rng(seed=1234).shuffle(shuffle_idx)
for key in raw_dict.keys():
    raw_dict[key][shuffle_idx]
# The below code line just converts the data to torch if needed then adds new values to the class
# Values added to class and converted to tensors
# Pretty sure this isn't needed though because all components are manually converted to tensors in the code
# StageMDataSetTorch.from_numpy(raw_data2)
stageM_data = {'labels':raw_labels, 'Chi':raw_Chi, 'Z':raw_Z}
stage1_MNZ = {'M': M_trainset1.to_numpy(), 'N': N_trainset1.to_numpy(), 'Z': Z_trainset1, 'sigmaZ': sigmaZ}

# **3**

# stage_m_out = self.stage_M_main(stageM_data=stageM_data, stage1_MNZ=stage1_MNZ, train_params=self.train_params)
class model_class(torch.nn.Module):
    def __init__(self, stageM_data: stageM_data, train_params: train_params, stage1_MNZ: stage1_MNZ,
                 gpu_flg: bool = False):
        super().__init__()
        self.stageM_data = stageM_data
        self.stage1_MNZ = stage1_MNZ
        self.reg_param = train_params['reg_param']
        # We are attempting to uncover a 1 dimensional X and thus initialize with the row averages of M and N
#       self.x_initialiser = (torch.tensor(stage1_MNZ['M']).mean(axis=1) + torch.tensor(stage1_MNZ['N']).mean(axis=1)) / 2
        # Multidimensional X with the same dimensions as N
        self.x_initialiser = (torch.tensor(stage1_MNZ['M'][:,0:stage1_MNZ['N'].shape[1]]) + torch.tensor(stage1_MNZ['N'])) / 2

        if not train_params['lambda_x']:
#             self.params = torch.nn.Parameter(self.x_initialiser.flatten())
#             self.x = self.params.reshape(-1,1)
            self.x = torch.nn.Parameter(self.x_initialiser)
            self.lambda_x = self.x
        else:
#             self.params = torch.nn.Parameter(self.x_initialiser.flatten())
#             self.x = self.params.reshape(-1,1)
            self.x = torch.nn.Parameter(self.x_initialiser)
            self.lambda_x = train_params['lambda_x']
        self.train_params = train_params
        self.KZ1Z1 = torch.tensor(np.exp(-cdist(stage1_MNZ['Z'], stage1_MNZ['Z'], "sqeuclidean") / 2 / float(stage1_MNZ['sigmaZ'])))
    def forward(self, idx):
    # Obtain wX(alpha, Z) in equation 19

        n = self.stage1_MNZ['Z'].shape[0]
        z = self.stageM_data['Z'][idx]
        K_Z1z = torch.tensor(np.exp(-cdist(stage1_MNZ['Z'], z, "sqeuclidean") / 2 / float(stage1_MNZ['sigmaZ'])))
        # gamma = self.cme_X.brac_inv.matmul(K_Zz)
        if not self.train_params["lambda_x"]:
            gamma_x_I_lambda = sum([torch.eye(n) * torch.exp(self.lambda_x[:,i].reshape(-1,1)) for i in range(self.lambda_x.shape[1])])
            gamma_x = torch.linalg.solve(self.KZ1Z1 + n * gamma_x_I_lambda, K_Z1z)
#             gamma_x = torch.linalg.solve(self.KZ1Z1 + n * torch.exp(self.lambda_x) * torch.eye(n), K_Z1z)
            # gamma_x = torch.linalg.solve(self.KZ1Z1 + n * self.lambda_x * torch.eye(n), K_Z1z)
        else:
            gamma_x_I_lambda = sum([torch.eye(n) * torch.exp(self.lambda_x[:,i].reshape(-1,1)) for i in range(self.lambda_x.shape[1])])
            gamma_x = torch.linalg.solve(self.KZ1Z1 + n * gamma_x_I_lambda, K_Z1z)
#             gamma_x = torch.linalg.solve(self.KZ1Z1 + n * self.lambda_x * torch.eye(n), K_Z1z)

        ### decompose e^{i\mathcal{X}n_i} ###
        cos_term = [torch.cos(torch.matmul(torch.tensor(self.stageM_data['Chi'][:,i].reshape(-1,1)[idx]).float(),\
                                          self.x[:,i].reshape(1, -1))) for i in range(self.x.shape[1])]
        sin_term = [torch.sin(torch.matmul(torch.tensor(self.stageM_data['Chi'][:,i].reshape(-1,1)[idx]).float(),\
                                          self.x[:,i].reshape(1, -1))) for i in range(self.x.shape[1])]
#         cos_term = torch.cos(torch.matmul(torch.tensor(self.stageM_data['Chi'].reshape(-1,1)[idx]).float(),\
#                                           self.x.reshape(1, -1)))
#         sin_term = torch.sin(torch.matmul(torch.tensor(self.stageM_data['Chi'].reshape(-1,1)[idx]).float(),\
#                                           self.x.reshape(1, -1)))

        denom = {}
        # using gamma to evaluate the charasteristic function value at a bunch of curly_x's
        denom['cos_weighted'] = sum([torch.sum(cos_term[i] * gamma_x.t(), dim=-1).reshape(-1, 1)\
                                     for i in range(self.x.shape[1])])
        denom['sin_weighted'] = sum([torch.sum(sin_term[i] * gamma_x.t(), dim=-1).reshape(-1, 1)\
                                     for i in range(self.x.shape[1])])
#         denom['cos_weighted'] = torch.sum(cos_term * gamma_x.t(), dim=-1).reshape(-1, 1)
#         denom['sin_weighted'] = torch.sum(sin_term * gamma_x.t(), dim=-1).reshape(-1, 1)
        denom['value'] = denom['cos_weighted'] + denom['sin_weighted'] * 1j

        numer = {}
        numer['cos_weighted'] = sum([torch.sum(cos_term[i] * gamma_x.t() * self.x[:,i].reshape(1, -1),\
                                               dim=-1).reshape(-1, 1) for i in range(self.x.shape[1])])
        numer['sin_weighted'] = sum([torch.sum(sin_term[i] * gamma_x.t() * self.x[:,i].reshape(1, -1),\
                                          dim=-1).reshape(-1, 1) for i in range(self.x.shape[1])])
#         numer['cos_weighted'] = torch.sum(cos_term * gamma_x.t() * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
#         numer['sin_weighted'] = torch.sum(sin_term * gamma_x.t() * self.x.reshape(1, -1), dim=-1).reshape(-1, 1)
        numer['value'] = numer['cos_weighted'] + numer['sin_weighted'] * 1j

        return numer['value'] / denom['value']

model = model_class(stageM_data=stageM_data, train_params=train_params, stage1_MNZ=stage1_MNZ)

model.train() # tells your model that you are training the model, not evaluating it
optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])

losses = []
early_stop = False
step = 0
for ep in range(train_params['num_epochs']):
    if early_stop:
        break
    running_loss = 0.0
    batches_idxes = []
    idxes = np.arange((stageM_data['Chi']).shape[0])
    np.random.shuffle(idxes)
    batch_i = 0
    while True:
        batches_idxes.append(torch.tensor(idxes[batch_i * train_params['batch_size']: (batch_i + 1) * train_params['batch_size']]))
        batch_i += 1
        if batch_i * train_params['batch_size'] >= (stageM_data['Chi']).shape[0]:
            break
    for i, batch_idx in enumerate(batches_idxes):
        preds = model(batch_idx)
        # Compute mse to obtain loss
        labels = stageM_data['labels'][batch_idx]
        dim_label = labels.shape[-1]
        num_label = labels.shape[0]
        preds_as_real = torch.view_as_real(preds)
        labels_as_real = torch.view_as_real(torch.tensor(labels))
        mse = torch.sum((labels_as_real - preds_as_real) ** 2) / num_label / dim_label
        reg = torch.sum((model.x - (torch.tensor(stage1_MNZ['M'].mean(axis=1) + stage1_MNZ['N'].mean(axis=1)) / 2)) ** 2)
        loss = mse + train_params['reg_param'] * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 1 == 0:
            print('[epoch %d, batch %5d] loss: %.5f, mse: %.5f, reg: %.5f' % (
            ep + 1, i + 1, running_loss / 1, mse / 1, train_params['reg_param'] * reg / 1))
            running_loss = 0.0

        losses.append(loss.item())
        if step > 8000: # Max of 8000 iterations
            break
        if (step > 2) and np.abs(losses[-1] - losses[-2]) < 1e-7: # Convergence considered to be < 1e-7
            early_stop = True
            break
        step += 1


# Convert model to numpy after training
fitted_x = model.x.detach().numpy()
#  assert stage_M_out.fitted_x.shape[0] == stage1_MNZ.Z.shape[0]
if not train_params['lambda_x']:
    lambda_x = np.exp(model.lambda_x.detach().numpy())  # syntax?
else:
    lambda_x = model.lambda_x

# **4**
# if X_obs is not None:
#     fitted_x = np.concatenate([fitted_x, X_obs], axis=-1)
# if covariate is not None:
#     fitted_x = np.concatenate([fitted_x, covariate], axis=-1)

gamma_x = sum([np.linalg.solve(KZ1Z1 + n * lambda_x[:,i] * np.eye(n), KZ1Z2) for i in range(lambda_x.shape[1])])
#gamma_x = np.linalg.solve(KZ1Z1 + n * lambda_x * np.eye(n), KZ1Z2)

# sigmaX = get_median(fitted_X)
sigmaX = np.median(cdist(fitted_x, fitted_x, "sqeuclidean"))
KfittedX = np.exp(-cdist(fitted_x, fitted_x, "sqeuclidean") / 2 / float(sigmaX))
W = KfittedX.dot(gamma_x)


# ***END Merror STAGE***


# ***BEGIN STAGE 2***

# Obtain beta hat (equation 27 in paper)
xi = train_params['xi']
if isinstance(xi, list):
    xi = np.exp(np.linspace(xi[0], xi[1], 50))
    M = W.shape[1]
    b = W.dot(Y_trainset2)
    A = W.dot(W.T) # W.T is transpose of W
    alpha_list = [np.linalg.solve(A + M * lam2 * KfittedX, b) for lam2 in xi]
    score = [np.linalg.norm(Y_trainset1 - KfittedX.dot(alpha)) for alpha in alpha_list]
    alpha = alpha_list[np.argmin(score)]
    xi = xi[np.argmin(score)]
else:
    alpha = np.linalg.solve(W.dot(W.T) + m * self.xi * KfittedX, W.dot(Y_trainset2))

# ***END STAGE 2***

# ***OBTAIN FINAL OUTPUT***
# Concatenate the covariate with the test data if there is a covariate
if covariate_test is not None:
    X_all_test = np.concatenate([X_all_test, covariate_test], axis=-1)
# Obtain predictions
Kx = np.exp(-cdist(X_all_test, fitted_x, "sqeuclidean") / 2 / float(sigmaX))
preds = np.dot(Kx, alpha) # Obtain predictions from equation 26 in paper
# Evaluate the model
mse = np.mean((Y_struct_test - preds)**2)

