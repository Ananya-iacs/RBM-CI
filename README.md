
# RBM-CI

![Project Image](project-image-url)

RBM-CI is a protocol to generate important configurations for strongly correlated systems and to calculate ground and excited state energies along with targeting the spin state to calculate singlet-triplet gap for such systems with multireference character.

---

### Table of Contents
- [Description](#description)
- [How To Use](#how-to-use)
- [Output Files](#output-files)
- [References](#references)
- [Contributors](#contributors)
- [Author Info](#author-info)
- [License](#license)

---

## Description
Monte Carlo (MC) is a numerical technique where a problem is solved using the help of a random number. In Monte Carlo Configuration Interaction (MCCI), a system's electronic structure is solved by randomely searching over the Hilbert space. Though MCCI helps to study the electronic structure of a system, which was otherwise impossible to do, it suffers from slow convergence due to its stochastic nature. We devised a protocol called RBM-CI, where direct generation of important configuration by generative ML model, restricted Boltzmann machine (RBM) makes the convergence manyfold faster and can also optimize Hilbert space for a particular target state. 

Initially, MCCI steps update the sub-Hilbert space and build the training data set. RBM model learns from the MCCI data distribution and, using that information predicts the relative importance of unlabelled configurations. The preliminary information about the configurations helps build a better Hilbert space, leading to faster convergence.


---

## How To Use

### Prerequisites
1. CUDA Device
2. CUDA Toolkit
3. Python3.6 +
4. PyTorch
5. Numba
6. f2py3

### Installation
After geting the code the net_nstates.f file, which is a Fortran code, need to converted into a Python excutable file. f2py3 a Fortran to Python interface generator used at this point.


    f2py3 -L/usr/lib. -llapack -c net_nstates.f -m net_states
    
This command generate a file - net_nstates.cpython-xxxxxx-gnu.so. Rename this file to net_nstates.so.

    mv  net_nstates.cpython-xxxxxx-gnu.so  net_states.so


### Setup of Input File
In the input file, arguments are given in "P,Q,R" format, where P is the keyword and Q, R are values associated with the keyword.

     ***startSetup***
        model,HB
        nSite,18
        subSpace,15200
        mlStart,5
        ciPow,1
        trainSampleSize,100000
        testSampleSize,50000
        newGenerationSize,100000
        gibStepTrain,10
        gibStepGen,80
        ciThresh,1e-5
        hidNode,2
        trainBatchSize,1000
        lr,0.01
        epoch,200
        nStates,10
        Ms,1,0
        s2Target,0
        maxItr,50
        startSpinTargetItr,5
        energyTola,0.0001
        spinTola,0.01
        jValue,1
        beta,38.61
        bondOrder,bondorder.dat
        restart,False
    ***endSetup***


### Performing a Calculation

Once the input file is constructed and all the files put into the same directory, the user can perform the RBM-CI calculation by using the below command-

```bash
python exe.py input_file.in &
```

---

## Output Files
There is a total 10 output files generated after successful calculations-
The main files are

    1) input_file.in.out            # Main output file, which contains information on subspace size, energy, and spin value with each iteration. 
    2) input_file.in.out.basis      # Configurations of final sub-Hilbert space
    3) input_file.in.out.ci         # CI coefficienet corresponding to configurations
    4) input_file.in.out.model.pth  # Final optimized RBM model
    5) input_file.in.out.losses_{iteration number}.dat  # Train and test free energy and KL-divergence at each RBM-CI iteration
    6) input_file.in.out.TrainData_subSpace_sample_{iteration number}.csv # Train data set generated during each iteration
    7) input_file.in.out.TestData_subSpace_sample_{iteration number}.csv # Test data set generated during each iteration
    8) input_file.in.out.generated_{iteration number}.dat  #RBM generated total configurations at each iteration
    9) input_file.in.out.generated_unique_{iteration number}.dat #RBM generated unique configurations at each iteration
    

---

## References
Active Learning Assisted MCCI to Target Spin States https://pubs.acs.org/doi/10.1021/acs.jctc.2c00935


---

## Contributors
- Pritam Bhattacharyya
- Ananya Sinha
- Debashree Ghosh



---

## License
Pick a license (MIT is common and permissive) — GitHub can auto-generate this when creating the repo, or you can add a `LICENSE` file separately.
