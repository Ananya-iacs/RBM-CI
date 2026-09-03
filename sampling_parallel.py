import torch
import numpy as np
import math
import time
import random
from tqdm import tqdm
from collections import Counter
from bitstring import Bits, BitStream, BitArray, ConstBitStream
from newGeneration import allConnectedDeterminants

def RBM_sample(det, ci, sample_size, ci_pow, dataFile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frontier_per_det = 3
    ci_cut_frontier = 1.0e-3

    ci_tensor = torch.tensor(ci, dtype=torch.float32, device=device)
    p_tensor = torch.abs(ci_tensor)**ci_pow
    max_ci2 = torch.max(p_tensor).item()

    det_list = [d.bin for d in det]

    print("bitstring length =", len(det_list[0]))
    print("first determinant =", det_list[0])

    sample_dets = []
    sample_ci = []

    train_dets = []
    train_ci_list = []

    c_train = 0
    pbar = tqdm(total=sample_size, desc="Sampling data")

    while c_train < sample_size:
        i = np.random.randint(0, len(det))
        j = np.random.uniform(0, max_ci2 + 0.1)

        if j <= p_tensor[i].item():

            parent_det_str = det_list[i]
            parent_ci = ci_tensor[i].item()

            sample_dets.append(parent_det_str)
            sample_ci.append(parent_ci)

            train_dets.append(parent_det_str)
            train_ci_list.append(parent_ci)

            c_train += 1
            pbar.update(1)

    pbar.close()

    # ============================================================
    # TARGETED FRONTIER TRAINING DATA
    # ============================================================

    frontier_added = 0
    frontier_unique = set()

    for det_str, ci_val in zip(det_list, ci):

        if abs(float(ci_val)) < ci_cut_frontier:
            continue

        parent_bit = BitArray(bin=det_str)
        neighbours = allConnectedDeterminants(parent_bit)

        if len(neighbours) == 0:
            continue

        random.shuffle(neighbours)

        added_here = 0

        for nbor in neighbours:

            train_dets.append(nbor.bin)
            train_ci_list.append(float(ci_val))

            frontier_unique.add(nbor.bin)
            frontier_added += 1
            added_here += 1

            if added_here >= frontier_per_det:
                break

    print("targeted_frontier_per_det =", frontier_per_det)
    print("ci_cut_frontier =", ci_cut_frontier)
    print("targeted frontier rows added =", frontier_added)
    print("targeted frontier unique determinants =", len(frontier_unique))

    print("INPUT DETS =", len(det))
    print("INPUT UNIQUE DETS =", len(set(d.bin for d in det)))

    ci_counter = Counter(ci)
    duplicate_ci = sum(1 for v in ci_counter.values() if v > 1)
    print("Number of duplicated CI values =", duplicate_ci)

    unique_map = {}

    for det_str, ci_val in zip(sample_dets, sample_ci):
        if det_str not in unique_map:
            unique_map[det_str] = ci_val

    print("dets in sample", len(unique_map))

    allDet = []
    allCicoef = []

    for det_str, ci_val in unique_map.items():
        allDet.append(BitArray(bin=det_str))
        allCicoef.append(float(ci_val))

    total_weight = np.sum(np.array(ci, dtype=float)**2)
    sampled_weight = np.sum(np.array(allCicoef, dtype=float)**2)

    print("Unique-sampled CI weight fraction =",
          sampled_weight/max(total_weight,1.0e-16))

    print("\nSampling diagnostics")
    print("sample_ci min =", min(sample_ci))
    print("sample_ci max =", max(sample_ci))
    print("number unique CI =", len(set(sample_ci)))
    print("training rows total =", len(train_dets))
    print("training unique determinants =", len(set(train_dets)))
    print()

    with open(dataFile, 'w') as fout:
        for det_str, ci_val in zip(train_dets, train_ci_list):
            conf = list(det_str)
            line = ','.join(conf) + "," + str(ci_val) + '\n'
            fout.write(line)

    print("OUTPUT DETS =", len(allDet))
    print("OUTPUT UNIQUE DETS =", len(set(d.bin for d in allDet)))

    loss_fraction = 1.0 - len(allDet)/max(1, len(det))
    print("Fraction of determinants lost =", loss_fraction)

    print("Unique determinants retained =",
          len(allDet)/max(1,len(det)))

    return allDet, allCicoef, dataFile



