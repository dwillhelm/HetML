# HetML  

HetML is a framework to develop and deploy machine learning models that predict material properties of van der Waals heterostructures.  Please cite the following reference if you use this repo:  

>“**Predicting Van der Waals Heterostructures by a Combined Machine Learning and Density Functional Theory Approach**”, Daniel Willhelm, Nathan Wilson, Raymundo Arroyave, Xiaoning Qian, Tahir Cagin, Ruth Pachter, and Xiaofeng Qian, *ACS Applied Materials & Interfaces (2022)*.  https://pubs.acs.org/doi/10.1021/acsami.2c04403

<!-- ![alt text](https://github.com/dwillhelm/HetML/blob/master/docs/figs/figure_1_new_DW_XQ_v3_highres.jpg?raw=true) -->

## Framework  
Target Properties: 
* Band Gap Energy (eV) 
* Ionization Energy (eV) 
* Electron Affiniity (eV) 
* Interlayer Distance (Angstrom)  
* Interlayer Binding Energy (meV/Angstrom^2)   
* (Coming Soon!) Charge Transfer (via Bader Analysis) 
* (Coming Soon!) Dipole Moment
* (Coming Soon!) In-plance lattice constant  


Some Deep learning models were also tested and can be found at this [repo](https://github.com/dwillhelm/DeepHetML)


<!-- ![alt text](https://github.com/dwillhelm/HetML/blob/master/docs/figs/figure_6.svg?raw=true) -->


<!-- https://pubs.acs.org/doi/10.1021/acsami.2c04403 -->


## Setup: 
### Python Environment
`conda env create -f environment.yml`
or 
`conda env create -f docs/envs/environment_full.yml` (this YAML lists all dependencies and subdependencies) 
or 
`pip install -r requirements.txt`

## Install Project Source Code
installs `hetml` Python package from setup.py 
`pip install .` 
or
 `pip install -e .` for a dev install


## ToDo: 
- [ ] Get dataset 
- [ ] Structure library --> compressed.  
- [ ] src code and setup into local package 
- [ ] figures and interactive figures (in notebook?)

- [x] Setup public repo to accompany the publication 
