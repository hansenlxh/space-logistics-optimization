# Prerequisites

In your Linux home directory, create a new directory "opt" by

```shell
cd ~
mkdir -p opt
```

Also make sure you have conda:

```sh
conda --version
```

# Download solver

Download and extract solver package (adjust Gurobi version if needed)

```shell
cd opt
wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz
tar xvfz gurobi11.0.3_linux64.tar.gz
```

Alternatively, download Gurobi software from [download center](https://www.gurobi.com/downloads/gurobi-software/) and place it your `~/opt` directory.

# Environmental Variable Setup

Setup environmental variables.
Adjust Gurobi version and shell (`.zshrc` instead of `.bashrc`, etc)

```sh
echo `export GUROBI_HOME="/opt/gurobi1101/linux64"` >> ~/.bashrc
echo `export PATH="${PATH}:${GUROBI_HOME}/bin"` >> ~/.bashrc
echo `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"` >> ~/.bashrc
source ~/.bashrc
```

# License Setup

Create a new conda environment for Gurobi license setup, and install gurobi module.
**Make your gurobi module version is the same as the downloaded solver version**

```sh
conda create -n gurobi_setup python=3.11 -y
conda activate gurobi_setup
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi=11.0.3 -y
```

Then authenticate your licensee via `grbgetkey` command:

```sh
grbgetkey YOUR/GUROBI/KEY
```

Save your license file `gurobi.lic` in `~` (if placed somewhere else, configure the environemnt variable so it can be in the path).

# Test Installation

```shell
cd ~
gurobi_cl $GUROBI_HOME/examples/data/afiro.mps
```

if not working,

```sh
cd ~/opt/gurobi1103/linux64/examples/data/
gurobi_cl ./afiro.mps
```

if not working,

```sh
cd ~/opt/gurobi1003/linux64/examples/data/
gurobi_cl ./afiro.mps
```
