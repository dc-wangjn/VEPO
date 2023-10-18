# Value Enhancement of Reinforcement Learning via Efficient and Robust Trust Region Optimization


# Author Contributions Checklist Form

## Data

The OhioT1DM dataset(2018) contains glucose monitoring, insulin, physiological sensor, and self-reported life-event data for six patents with type 1 diabetes over eight weeks. You need to apply it via [URL](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).

By the way, we generate synthetic data to mimic this real dataset and apply our method to the synthetic dataset, the data generator can be found in [./real_data/simulation/Ohio_Simulator.py](./real_data/simulation/Ohio_Simulator.py)

## Code

The code for VEPO calculations is consist of two parts, experiments in Section 5.1 of text and Supplement D.1 can be found in  fold ```toy``` , while experiments in Section 5.2 and Supplement D.2 can be found in fold ```real_data``` . The source code of manuscript and ACC form is attached in this repo for the convenience of reviewing. To reproduce our results in manuscript, the following instruction will be helpful.
### Installation
* Use `git clone` command to clone our repository to your machine.
* Conduct `pip install -r requirements.txt` to install the dependencies of our repo.

* Note that we  import the rpy2 library which is an interface to R running embedded in a Python process to use the R package(VL) named `infiniteHorizon`, which is implemented by Luckett in their [paper](https://www.tandfonline.com/doi/10.1080/01621459.2018.1537919) . In order to install this package, R should be installed in your machine  and the dependencies of `caret, kernlab, MASS, nnet` should also be installed in your device. After preparing all the needed materials, install the `./rworkspace/infiniteHorizon_010.tar.gz` using `install.packages` command in your R shell.

### Toy Module
Firstly enter the fold `toy`.
This sub-project implement the toy example in Section 5.1 of text to show the multiple robustness and desired value enhancement property.
There are two main version of our method:
* toy.py This script implement a class that all the nuisance functions are esimated via monte-carlo method.
* toy_est.py This script implement a class that the nuisance functions are parametrized to be lookup tables.

We then apply our algorithm using parallelization technique,
* multi_nmoptim_robust.py This script import toy.py to derive the corresponding value enhancement procedure results.
* multi_nmoptim_est.py This script import toy_est.py to derive the corresponding value enhancement procedure results.

It is worth noting that you can pass the needed parameters in command line to above two scripts in the following style:

    python multi_nmoptim_robust.py --sigma 2 --delta 0.05 --delta2 0.05 --rep 100 --batch 32 --suffix as_u_like

    python multi_nmoptim_est.py --sigma 2 --delta 0.05 --delta2 0.05 --rep 100 --batch 32 --suffix as_u_like

The result will be saved in fold `results`. We provide our results in that fold.
To reproduce the result in text, we can replace the delta value to be 0.1,0.2,0.05, which will derive the numerical  results showed in Figure 1, Figure S1, Figure S2 respectively.
Then, to generate the picture, the specific procedure is displayed in `plot.ipynb `

### Real Data Module
Firstly enter the fold `real_data`.
This subsection implement the experiments and simulations introduced in Section 5.2 of main text and Section D.2 of supplementary. 

We use the deep learning models to parametrize the nuisance functions, which can be found in `./_RL` and `./_density`:
* ./_RL/FQE.py Made to estimate the Q function using fitted Q-evaluation method.
* ./_density/omega_SASA_new.py Made to estimate the conditional density ratio.
* ./_density/guassian.py Made to estimate the underlying dynamics.

At the same time, several methods are implemented to provide the initial policy to our algorithm, we here only list the three methods used in the main text, i.e., FQI, CQL, VL:
* ./RL/FQI.py Implement the FQI method.
* ./RL/CQL.py Implement the CQL method.
* ./_base_poicy/vlearning/vl_simu.py Implement the VL method by applying `infiniteHorizon` R package. 



After preparing the needed environment,  our results could be reproduced by running the following scripts:
* experiments_ohio.py This script implement the experiment using generate synthetic data introduced in Section 5.1.
* experiments_real-data.py This script implement the experiment using real dataset introduced in Section 5.1.
* experiments_vl.py This script implement the experiment using simulation data introduced in D.2 of Supplementary.

You can pass the needed parameters in command line to above  scripts in the following style:

    
    python experiments_ohio.py --init_policy CQL  --delta 0.1 --gamma 0.9 --suffix as_u_like 

where `init_policy` specifies the intial policy used in this experiment, you can replace it with `FQI` only here. `gamma` denotes the $\gamma$ value and `delta` denotes the $\delta$ value. `suffix` is a tag for the uniqueness of file name. Parameters in  following command are similar:

    python experiments_real_data.py --init_policy CQL  --delta 0.1  --gamma 0.9 --suffix as_u_like 

    python experiments_vl.py --init_policy FQI  --delta 0.05  --gamma 0.9 --sigma 0.25 --size 1   --suffix as_u_like

Note that `init_policy` can be `FQI,CQL,VL` in script `experiments_vl.py`. `sigma` is specifies the standard error of $\epsilon$ in Supplementary D.2. `size`  be 1 represent $TN = 5000$, while 0 represent $TN = 3000$.

To derive Figure 2 , we should run `experiments_ohio.py` using the following parameter combination :

| Figure 2 | gamma=0.9 | gamma=0.95 |
| ------ | ------ | ------ |
| FQI| (FQI,0.9) | (FQI,0.95) |
| CQL | (CQL,0.9) | (CQL,0.95) |

To derive Figure 3 , we should run `experiments_real_data.py` using the following parameter combination :

| Figure 3 | gamma=0.9 | gamma=0.95 |
| ------ | ------ | ------ |
| FQI| (FQI,0.9) | (FQI,0.95) |
| CQL | (CQL,0.9) | (CQL,0.95) |

To derive Figure S4,S5,  run `experiments_vl.py` using the following parameter combination :

| Figure S4 | gamma=0.9 | gamma=0.95 |
| ------ | ------ | ------ |
| FQI| (FQI,0.9,size=1,sigma=1) | (FQI,0.95,size=1,sigma=1) |
| CQL | (CQL,0.9,size=1,sigma=1) | (CQL,0.95,size=1,sigma=1) |
| VL | (VL,0.9,size=1,sigma=1) | (VL,0.95,size=1,sigma=1) |

| Figure S5 | gamma=0.9 | gamma=0.95 |
| ------ | ------ | ------ |
| FQI| (FQI,0.9,size=0,sigma=1) | (FQI,0.95,size=0,sigma=1) |
| CQL | (CQL,0.9,size=0,sigma=1) | (CQL,0.95,size=0,sigma=1) |
| VL | (VL,0.9,size=0,sigma=1) | (VL,0.95,size=0,sigma=1) |


Above results will be saved in fold `results`. Then, to generate the picture, the specific procedure is displayed in final_plot.ipynb.
