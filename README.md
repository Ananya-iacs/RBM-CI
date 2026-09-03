
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
```bash
After geting the code the net_nstates.f file, which is a Fortran code, need to converted into a Python excutable file. f2py3 a Fortran to Python interface generator used at this point.
# Any build steps, compilation commands, etc.
    f2py3 -L/usr/lib. -llapack -c net_nstates.f -m net_states
This command generate a file - net_nstates.cpython-xxxxxx-gnu.so. Rename this file to net_nstates.so.
    mv  net_nstates.cpython-xxxxxx-gnu.so  net_states.so
```

### Setup of Input File
Explain the input format, with an annotated example block like in AL-MCCI's README — this is very useful for users.

### Performing a Calculation
```bash
python your_script.py input_file.in &
```

---

## Output Files
List and briefly describe each output file the user should expect.

---

## References
Link to your paper(s), with a paraphrased one-line description of what it covers (avoid quoting abstracts directly — just describe it in your own words).

---

## Contributors
- Your name
- Any collaborators

## Author Info
- Contact info / lab website / social links

---

## License
Pick a license (MIT is common and permissive) — GitHub can auto-generate this when creating the repo, or you can add a `LICENSE` file separately.
