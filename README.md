# HetML  

Framework to develop and deploy machine learning models that predict material properties of van der Waal heterostructures. Acompaning repo for **"Predicting Van der Waals Heterostructures by a Combined Machine Learning and Density Functional Theory Approach"** published in ACS Appl. Mater. Interfaces. https://pubs.acs.org/doi/10.1021/acsami.2c04403

Authors: D.Willhelm, N.Wilson, R. Arroyave, Xiaoning Qian, T.Cagin, R.Pachter, Xiaofeng Qian

![alt text](https://github.com/dwillhelm/HetML/blob/master/docs/figs/figure_1_new_DW_XQ_v3_highres.jpg?raw=true)

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


Some Deep learing models were also tested and can be found at this [repo](https://github.com/dwillhelm/DeepHetML)



<!-- ![alt text](https://github.com/dwillhelm/HetML/blob/master/docs/figs/figure_6.svg?raw=true) -->


## Manuscript  
**Abstract**:  
Van der Waals (vdW) heterostructures are constructed by different two-dimensional (2D) monolayers vertically stacked and weakly coupled by van der Waals interactions. VdW heterostructures often possess rich physical and chemical properties that are unique to their constituent monolayers. As many 2D materials have been recently identified, the combinatorial configuration space of vdW-stacked heterostructures grows exceedingly large, making it difficult to explore through traditional experimental or computational approaches in a trial-and-error manner. Here, we present a computational framework that combines first-principles electronic structure calculations, 2D material database, and supervised machine learning methods to construct efficient data-driven models capable of predicting electronic and structural properties of vdW heterostructures from their constituent monolayer properties. We apply this approach to predict the band gap, band edges, interlayer distance, and interlayer binding energy of vdW heterostructures. Our data-driven model will open avenues for efficient screening and discovery of low-dimensional vdW heterostructures and moiré superlattices with desired electronic and optical properties for targeted device applications.  

<!-- https://pubs.acs.org/doi/10.1021/acsami.2c04403 -->


## ToDo: 
- [ ] Get dataset 
- [ ] Structure library --> compressed.  
- [ ] src code and setup into local package 
- [ ] figures and interactive figures (in notebook?)

- [x] Setup public repo to accompany publication 
