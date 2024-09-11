# Setting up the environment 

1. Clone the repository and enter it: 
```shell
  git clone git@github.com:raphael-group/paste3.git
  cd paste3
```
2. Create and activate a conda environment using the provided `environment.yml` file: 
```shell
   conda env create --file environment.yml
   conda activate paste
```
The command prompt will change to indicate the new conda environment by prepending `(paste)`. 
3. When you are done using the package, deactivate the `paste` environment and return to `(base)` by entering the following command: 
```shell
conda deactivate
```

# Installation 

1. Enter the `paste3` repository cloned before setting up the environment
```shell
cd paste3
```
2. Install the package:
```shell
pip install . 
```