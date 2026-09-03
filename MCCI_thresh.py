import sys
import numpy as np
from numpy import linalg as LA
from bitstring import Bits, BitStream, BitArray, ConstBitStream
import random
import statistics as stat
import math
import os
import numba
import itertools as it
from numba import jit, njit,config, threading_layer, prange, cuda, int64
from numba.typed import List

#from HeisenHam import Hamiltonian
import net_nstates
from newGeneration import mutation, reflection, makeNewGeneration 
from newConvergence import checkConvergence, checkFinalConv, makeFitGeneration, convInitializer, update, updateDeterminatList
from spinCalculator import spinCalculator, stateFinder
from setup import readInput
from makeRBMgeneration import train_RBM, RBM_generation
from sampling_parallel import RBM_sample
import torch
from collections import Counter

model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

if model == 'HB':
    from HeisenHam import Hamiltonian
if model == 'GM':
    from GhoshMajumHam import Hamiltonian


def performMCCI():    
    convReach = False   
    subBasis = []
    
    if restart:
        with open(saveBasis,"r") as fsaveB:
            for i in range(subSpace):
                line = fsaveB.readline()
                det0 = BitArray(bin=line.strip())
                subBasis.append(det0)
    
    if (restart == False):
        
        det0 = det
        #random.shuffle(det0)
        subBasis.append(det0)
        print("det0", det0)
        if Ms == 0 and ~det0 not in subBasis:
            subBasis.append(~det0)
            detCopy0 = BitArray()

        #while len(subBasis) < int(0.01*subSpace) :
        while len(subBasis) < 100 :
            detCopy0 = subBasis[0].copy()
            random.shuffle(detCopy0)
            if detCopy0 not in list(subBasis):
                subBasis.append(detCopy0)
                if Ms == 0 and ~detCopy0 not in subBasis:
                    if ~detCopy0 not in list(subBasis):
                        subBasis.append(~detCopy0)
    
    
    subHam = Hamiltonian(subBasis)
    lenSB = len(subBasis)
    
    energy = np.zeros(lenSB)
    ciCoef = np.zeros((nStates * lenSB))
    net_nstates.diagonalization(hamil = subHam, n = lenSB, n1 = 3 * lenSB, n2 = nStates, ehamil = energy, vec = ciCoef)
    energyMin = energy[ 0 ]
    ciCoefMin = ciCoef[ 0 : lenSB]

    #energy, ciCoef = LA.eigh(subHam) 
    #energyMin = energy[ 0 ]
    #ciCoefMin = ciCoef[ :, 0 ]
    
    targetState = [0,0]
    s2ValDiff = [0.0,0.0]
    s2ValMin = 100


    ## to store all the det and their CI coef to a data file
    allDet = subBasis
    allCicoef = ciCoefMin
    
    for i in range(maxItr):
        print("**********iteration*********", i)
        # creation of new generation
        if (i <= mlStart):
            print("iter entry: lenallDet, lenallCi, len(setallDet)", len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))
            allDet, lenallDet = makeNewGeneration(allDet)
            print("after new gener: lenallDet, lenallCi, len(setallDet) ", len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))

        if (i == mlStart+1):
            newline = ("\nStarting Active-Learning Protocal \n")
            with open(outputfile, "a") as fout:
                fout.write(newline)

#        rbm_added = set(b.bin for b in newGen)                
        if (i > mlStart):
            trainDataFile = outputfile + ".TrainData_subSpace_sample_"+str(i)+".csv"
            testDataFile = outputfile + ".TestData_subSpace_sample_"+str(i)+".csv"
            
            allDet, allCicoef, sampleTrainDataFile = RBM_sample(allDet, allCicoef, trainSampleSize, ciPow, trainDataFile)
            allDet_test, allCicoef_test, sampleTestDataFile = RBM_sample(allDet, allCicoef, testSampleSize, ciPow, testDataFile)

            train_RBM(sampleTrainDataFile, sampleTestDataFile, gibStepTrain, i)
            energy_before_rbm = energyMin
            newGen, lenNewGen, allDet, lenallDet = RBM_generation(newGenerationSize, allDet, allCicoef, gibStepGen, i)
            rbm_added = set(b.bin for b in newGen)
            print("len of unique gen", lenNewGen)
            print("after new gener: lenallDet, lenallCi, len(setallDet) ", len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))

        newGenHam = Hamiltonian(allDet)
        energy = np.zeros( lenallDet)
        allCicoef = np.zeros((nStates * lenallDet))
            
        net_nstates.diagonalization(hamil = newGenHam, n= lenallDet, n1 = 3 * lenallDet, n2 = nStates, ehamil = energy , vec = allCicoef)
        #print("i, lenallDet, lenallCi", i, lenallDet, len(allCicoef))
        #print("i, lenallDet, lenallCi, len(setallDet)", i, len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))

        s2ValList =  spinCalculator(allDet, energy[ 0 : nStates ], allCicoef, lenallDet, convReach)
        #print("s2ValList", s2ValList)

        if (i == startSpinTargetItr+1): # for smooth transition from non spin target to spin target cacluations
            newline = ("\nStarting Optimization W.R.T Spin, Target State Spin Value -> %f \n\n")%(s2Target)
            with open(outputfile, "a") as fout:
                fout.write(newline)

        if (i >= startSpinTargetItr):
            targetState[1], s2ValDiff[1] = stateFinder(s2ValList,s2Target,targetState[0])  # for first state of a particular spin

        
        ciCoefNew = allCicoef[(lenallDet) * targetState[1] : (lenallDet) * (targetState[1] +1)]

        if (i > mlStart):
            important_rbm = 0
            for detx,cix in zip(allDet, ciCoefNew):
                if detx.bin in rbm_added:
                    if abs(float(cix)) > ciThresh:
                        important_rbm += 1
            print("Important RBM determinants =", important_rbm)

            rbm_weight = 0.0
            total_weight = 0.0
            for detx,cix in zip(allDet, ciCoefNew):
                w = float(cix)**2
                total_weight += w
                if detx.bin in rbm_added:
                    rbm_weight += w
            print("RBM weight fraction =", rbm_weight/total_weight)

            largest_rbm_ci = 0.0

            for detx,cix in zip(allDet, ciCoefNew):
                if detx.bin in rbm_added:
                    largest_rbm_ci = max(
                        largest_rbm_ci,
                        abs(float(cix))
                    )

            print("Largest RBM |CI| =", largest_rbm_ci)

        energyNew = energy[ targetState [ 1 ] ]

        if (i > mlStart):
            print("RBM energy improvement =",
                  energy_before_rbm - energyNew)

        s2ValNew = s2ValList [targetState [ 1 ]]
   
        energyChange = energyMin - energyNew
        spinChange = s2ValMin - s2ValNew

        allDet, allCicoef, energyMin, s2ValDiff, s2ValMin, energyUpdate = checkConvergence(energyMin, energyNew, s2ValMin, s2ValNew, allDet, ciCoefNew, targetState, s2ValDiff, i)
        convReach = checkFinalConv( energyChange, spinChange, convReach) 
        print("before trunc: lenallDet, lenallCi, len(setallDet) ", len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))
        
        if (i <= mlStart):
            absCicoef = [abs(x) for x in allCicoef]
            sortedCicoef = sorted(absCicoef, reverse=True)
            netAllDet = []
            netAllCi = []
            for elem in sortedCicoef[:1000]:
                #if elem == 0.0:
                #    continue
                #else:
                ind = list(absCicoef).index(elem)
                if allDet[ind] not in netAllDet:
                    netAllDet.append(allDet[ind])
                    netAllCi.append(allCicoef[ind])

            allDet = netAllDet
            allCicoef = netAllCi
        
        if (i > mlStart):
            net_allDet = []
            net_allCi = []
            for idy in range(len(allCicoef)):
                if abs(float(allCicoef[idy])) >= ciThresh:
                    net_allCi.append(float(allCicoef[idy]))
                    net_allDet.append(allDet[idy])

            allDet = net_allDet
            allCicoef = net_allCi

            survive = 0
            for detx in allDet:
                if detx.bin in rbm_added:
                    survive += 1
            print("RBM determinants surviving truncation =", survive)

            fOut = open("generated_after_truncation_"+str(i)+".dat", "w")
            for idx in range(len(allDet)):
                fOut.write(str(allDet[idx])+","+str(allCicoef[idx]))
                fOut.write("\n")
            fOut.close()

        print("after trunc: lenallDet, lenallCi, len(setallDet) ", len(allDet), len(allCicoef), len(set(b.bin for b in allDet)))

        if (convReach == True) or (i == maxItr - 1):
            if convReach:
                newline = ("\nIteration Converged.\n")
                with open(outputfile, "a") as fout:
                    fout.write(newline)
                break
            else:
                newline = ("\nReach Max Iteration Number.\n")
                with open(outputfile, "a") as fout:
                    fout.write(newline)   
                convReach = True
                break

    # Final Calculation
    energyFinal, basisFinal, ciFinal = update( energy[0 : nStates], allDet, allCicoef)
    
    with open( (str(outputfile) + '.basis'), "w") as fbasis:
        for element in basisFinal:
            fbasis.write(element.bin +'\n')

    with open( (str(outputfile) + '.ci'), "w") as fci:
        for element in ciFinal:
            fci.write(str(round(float(element),6)) +'\n')

