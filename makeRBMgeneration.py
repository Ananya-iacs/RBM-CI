#####################################################
# Gnereation of near exact probabilitydistribution #
# Using RBM #
#####################################################

import numpy as np
np.bool = np.bool_

import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset , DataLoader, random_split
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.datasets
import torchvision.models

from random import shuffle
import random

from torch.nn import L1Loss, BCELoss,NLLLoss,CrossEntropyLoss,MSELoss,MarginRankingLoss, KLDivLoss
import torchvision.transforms

from bitstring import Bits, BitStream, BitArray, ConstBitStream

from newGeneration import mutationiConected, allConnectedDeterminants

from setup import readInput

model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

#=================================================================
# preparing the Dataset
#=================================================================

class CSVDataset(Dataset):

    def __init__(self, path):

#        df = np.genfromtxt(path,delimiter=',', dtype=int)
        df = np.genfromtxt(path, delimiter=',', dtype=np.float32)

        self.X = df[:,:-1]
        self.Y = df[:,-1]

        self.X = self.X.astype('float32')
        self.Y = self.Y.astype('float32')

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]

#=============================================================
#definig the RBM model
#=============================================================

class RBM(nn.Module):

    def __init__(self, n_vis=784, n_hin=500, k=5):

        super(RBM, self).__init__()

        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)

        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))

        self.k = k

#    def sample_from_p(self, p):
#
#        return torch.bernoulli(p)

    def sample_from_p(self, p):

        if torch.isnan(p).any():
            print("\n====================")
            print("NaN detected in probability tensor")
            print("min =", torch.nanmin(p))
            print("max =", torch.nanmax(p))
            print("====================\n")
            raise RuntimeError("NaN probability")

        if (p < 0).any() or (p > 1).any():

            print("\n====================")
            print("Probability outside [0,1]")
            print("min =", p.min().item())
            print("max =", p.max().item())
            print("====================\n")
            raise RuntimeError("Invalid probability")

        return torch.bernoulli(p)




    def v_to_h(self,v):

#        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))

        pre_h = F.linear(v,self.W,self.h_bias)

        if torch.isnan(pre_h).any():

            print("\nNaN in pre_h")
            print("W min/max =", self.W.min().item(), self.W.max().item())
            print("h_bias min/max =", self.h_bias.min().item(), self.h_bias.max().item())
            raise RuntimeError

        p_h = torch.sigmoid(pre_h)

        sample_h = self.sample_from_p(p_h)

        return p_h,sample_h

    def h_to_v(self,h):

#        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))

        pre_v = F.linear(h, self.W.t(), self.v_bias)

        if torch.isnan(pre_v).any():
            print("\nNaN in pre_v")
            print("W min/max =", self.W.min().item(), self.W.max().item())
            print("v_bias min/max =", self.v_bias.min().item(), self.v_bias.max().item())
            raise RuntimeError

        if torch.isinf(pre_v).any():
            print("\nInf in pre_v")
            raise RuntimeError

        p_v = torch.sigmoid(pre_v)


        sample_v = self.sample_from_p(p_v)

        return p_v,sample_v

    def forward(self,v):

        pre_h1, h1 = self.v_to_h(v)

        h_ = h1

        for _ in range(self.k):

            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)

        return v,v_

    def free_energy(self,v):

        vbias_term = v.mv(self.v_bias)

        wx_b = F.linear(v,self.W,self.h_bias)

        if torch.isnan(wx_b).any():
            print("NaN in wx_b")
            raise RuntimeError

        if torch.isinf(wx_b).any():
            print("Inf in wx_b")
            raise RuntimeError

        hidden_term = wx_b.exp().add(1).log().sum(1)

        return (-hidden_term - vbias_term).mean()

#=============================================================
#Preparing the train and test data
#=============================================================

def prepare_data(path1):

    train = CSVDataset(path1)

    print("\nDataset:", path1)
    print("Shape:", train.X.shape)
    print("Unique values in X:", np.unique(train.X))
    print("Y min/max:", np.min(train.Y), np.max(train.Y))

    train_dl = DataLoader(
        train,
        batch_size = trainBatchSize,
        shuffle = True,
        drop_last = False
    )

    return train_dl

#==============================================================
# Calling the RBM module
#==============================================================

n = nSite

n_vis = n

n_hin = hidNode * n

# IMPORTANT:
# keeping user input values instead of hardcoding

n_epoch = epoch

#==============================================================
# Training The RBM model
#==============================================================

def train_RBM(sample_trainData_file, sample_testData_file, k, ite):

    start = time.time()

    rbm = RBM(n_vis = n_vis, n_hin=n_hin , k=k)

    train_op = optim.Adam(
        rbm.parameters(),
        lr = lr,
        betas=(0.9, 0.999),
        eps = 1e-8
    )

    KLDiv_sklearn = nn.KLDivLoss(reduction="batchmean", log_target= True)

    train_dl = prepare_data(sample_trainData_file)
    test_dl = prepare_data(sample_testData_file)

    f1_open = open(outputfile+".after_training_"+str(ite)+".dat","w")

    f_open = open(outputfile+".losses_"+str(ite)+".dat", "w")

    f_open.write("#Epoch"+"\t"+"Train_Free_energy"+"\t"+"Test_Free_energy"+"\t"+"Train_KL_sklearn"+"\t"+"Test_KL_sklearn")
    f_open.write("\n")

    pbar = tqdm(total=n_epoch, desc="RBM training")

    for epoch_idx in range(n_epoch):

        train_f_ene = 0.0
        train_kl_sk = 0.0

        test_f_ene = 0.0
        test_kl_sk = 0.0

        rbm.train()

        for _, (data,target) in enumerate(train_dl):

            data = Variable(data.view( -1, n))

            sample_data = data

            if torch.isnan(sample_data).any():
                print("\nNaN found in sample_data")
                print("batch =", _)
                raise RuntimeError

            v,v1 = rbm(sample_data)

            f_ene = (rbm.free_energy(v) - rbm.free_energy(v1))

            if torch.isnan(f_ene):
                print("\n====================")
                print("NaN free energy during TRAIN")
                print("epoch =", epoch_idx)
                print("====================\n")
                raise RuntimeError("NaN free energy")

            kl_sk = KLDiv_sklearn(v,v1)

            train_f_ene += f_ene.item()
            train_kl_sk += kl_sk.item()

            train_op.zero_grad()

            f_ene.backward()

            train_op.step()

            for name, param in rbm.named_parameters():

                if torch.isnan(param).any():

                    print("\n====================")
                    print("NaN detected in parameter:", name)
                    print("====================\n")

                    raise RuntimeError("NaN parameter")

            if epoch_idx == n_epoch-1:

                lines = [",".join(str(int(x.item())) for x in row) + "\n" for row in v1]

                f1_open.writelines(lines)

                torch.save(rbm.state_dict(), outputfile+".model.pth")

        epoch_train_f_ene = train_f_ene/len(train_dl)
        epoch_train_kl_sk = train_kl_sk/len(train_dl)

        rbm.eval()

        for _, (data,target) in enumerate(test_dl):

            data = Variable(data.view( -1, n))

            sample_data = data

            v,v1 = rbm(sample_data)

            f_ene = (rbm.free_energy(v) - rbm.free_energy(v1))

            if torch.isnan(f_ene):
                print("\n====================")
                print("NaN free energy during TEST")
                print("epoch =", epoch_idx)
                print("====================\n")
                raise RuntimeError("NaN free energy")

            kl_sk = KLDiv_sklearn(v,v1)

            test_f_ene += f_ene.item()
            test_kl_sk += kl_sk.item()

        epoch_test_f_ene = test_f_ene/len(test_dl)
        epoch_test_kl_sk = test_kl_sk/len(test_dl)

        f_open.write(
            str(epoch_idx+1)+"\t"+
            str(epoch_train_f_ene)+"\t"+
            str(epoch_test_f_ene)+"\t"+
            str(epoch_train_kl_sk)+"\t"+
            str(epoch_test_kl_sk)
        )

        f_open.write("\n")

        pbar.update(1)

    pbar.close()

    f1_open.close()
    f_open.close()

    end = time.time()

#=====================================================================
#saving the optimized Weights and biases
#=====================================================================

def gibbs_sampling(rbm, v, k=50):

    for _ in range(k):

        _, h = rbm.v_to_h(v)

        _, v = rbm.h_to_v(h)

    return v

#=====================================================================
# RBM GENERATION
#=====================================================================

def RBM_generation(n_generation, allDet, allCicoef, k, ite):

    start = time.time()

    rbm = RBM(n_vis = n_vis, n_hin = n_hin , k = k)

    rbm.load_state_dict(torch.load(outputfile+".model.pth"))

    gibbs_lattice = []

    unique_sample = set()

    pbar = tqdm(total=n_generation, desc="RBM data generation")

    Ngenerated = n_generation

    for i in range(Ngenerated):

        if Ms == 0:

            randomlattice = random.sample(
                [0]*int(n/2) + [1]*int(n/2),
                n
            )

        elif Ms == 1:

            randomlattice = random.sample(
                [0]*int(n/2-1) + [1]*int(n/2+1),
                n
            )

        randomlattice = torch.tensor(randomlattice, dtype=torch.float32)

        lattice1 = gibbs_sampling(rbm, randomlattice, k)

        if Ms == 0:

            if sum(lattice1) == n/2:

                lattice_tuple = tuple(int(x.item()) for x in lattice1)

                gibbs_lattice.append(lattice_tuple)

                unique_sample.add(lattice_tuple)

                pbar.update(1)

        elif Ms == 1:

            if sum(lattice1) == (n/2)+1:

                lattice_tuple = tuple(int(x.item()) for x in lattice1)

                gibbs_lattice.append(lattice_tuple)

                unique_sample.add(lattice_tuple)

                pbar.update(1)

    pbar.close()

    with open(outputfile+".generated_"+str(ite)+".dat", "w") as fout:

        lines = [",".join(str(int(x)) for x in list(row)) + "\n" for row in gibbs_lattice]

        fout.writelines(lines)

    end = time.time()

    from collections import Counter

    counter = Counter(gibbs_lattice)

    distinct_count = len(counter)

    common = counter.most_common()

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    print("len of unique gen", len(unique_sample))
    print("unique/generated ratio =",
           len(unique_sample)/len(gibbs_lattice))
    
    novel = 0
    already_known = 0

    for elem in unique_sample:

#        b = BitArray(elem)
        b = BitArray(bin=''.join(str(x) for x in elem))

        if b not in allDet:
            novel += 1
        else:
            already_known += 1

    print("Novel determinants =", novel)
    print("Already known determinants =", already_known)

    if len(unique_sample) > 0:

        novelty_fraction = novel / len(unique_sample)

        print("Novelty fraction =", novelty_fraction)
        print("Novel/Old ratio =", novel/max(1,already_known))

    # ============================================================
    # ADDING NEW DETERMINANTS
    # ============================================================

    old_ci_map = {}

    for detx, cix in zip(allDet, allCicoef):

        old_ci_map[detx.bin] = abs(float(cix))

    old_det_set = set(detx.bin for detx in allDet)

    scored_candidates = []

    for lattice_tuple, freq in common:

        elem = BitArray(bin=''.join(str(x) for x in lattice_tuple))

        if elem.bin in old_det_set:

            continue

        neighbours = allConnectedDeterminants(elem)

        connection_score = 0.0

        connected = False

        for nbor in neighbours:

            if nbor.bin in old_ci_map:

                connection_score += old_ci_map[nbor.bin]

                connected = True

        if connected:

            score = np.sqrt(freq) * connection_score

            scored_candidates.append(
                (score, freq, connection_score, elem)
            )

    scored_candidates.sort(
        key = lambda x: x[0],
        reverse = True
    )

    print("Physics-ranked RBM candidates =", len(scored_candidates))

    if len(scored_candidates) > 0:

        print("Top RBM candidate score =", scored_candidates[0][0])

        print("Top RBM candidate freq =", scored_candidates[0][1])

        print("Top RBM connection score =", scored_candidates[0][2])

    truncate_unique_sample = []

    for score, freq, connection_score, elem in scored_candidates:

        if len(allDet) > subSpace:

            break

        if elem.bin not in old_det_set:

            allDet.append(elem)

            old_det_set.add(elem.bin)

            truncate_unique_sample.append(elem)

            if Ms == 0:

                comp = ~elem

                if comp.bin not in old_det_set:

                    allDet.append(comp)

                    old_det_set.add(comp.bin)

                    truncate_unique_sample.append(comp)



# ==========================================
# Exact connectivity diagnostic
# ==========================================

    reachable = 0

    for detx in truncate_unique_sample:

        neighbours = allConnectedDeterminants(detx)

        found = False

        for nbor in neighbours:

            if nbor.bin in old_det_set:
                found = True
                break

        if found:
            reachable += 1

    print("RBM determinants added =", len(truncate_unique_sample))
    print("Connected to old space =", reachable)

    if len(truncate_unique_sample) > 0:
        print("Connected fraction =",
              reachable / len(truncate_unique_sample))


    # ============================================================
    # FINAL DIAGNOSTICS
    # ============================================================

    print("New determinants actually added =", len(truncate_unique_sample))
    print("Final determinant space size =", len(allDet))

    with open(outputfile+".generated_unique_"+str(ite)+".dat", "w") as fout:

        lines = [",".join(str(int(x)) for x in list(row)) + "\n" for row in truncate_unique_sample]

        fout.writelines(lines)

    return truncate_unique_sample, len(truncate_unique_sample), allDet, len(allDet)




