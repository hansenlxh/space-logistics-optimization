Mac specific read-me instructions

# Poetry installation:

On Mac, after following poetry_installation.md steps it may not be added to the PATH by default, so write `export PATH="$HOME/.local/bin:$PATH"` `$HOME/.local/bin`.

After doing `poetry install` in the slpy environment and running `pytest` expect tests to fail corresponding to Gurobi and IPOPT if those are not already installed system-wide.

# Gurobi installation troubleshooting:
Follow the instructions exactly as given in `gurobi_installation.md`, including using a separate conda environment from slpy. Download the license from Gurobi. After following the steps, the User Portal in Gurobi under the "Summary" tab should display something like: `version 13, installed on MacBook-Pro.local`.

# IPOPT installation troubleshooting:
Some of these instructions differ from the `ipopt_installation.md` file. Easier to refer to the following as a cross-reference: https://coin-or.github.io/Ipopt/INSTALL.html


## Install prerequisites
This step is slightly different:
Use homebrew to achieve the same things:

```sh 
brew update
brew install bash gcc
brew link --overwrite gcc
brew install pkg-config
```

Then call brew install ___ for all the other libraries listed from before as needed.

## Get `coinbrew`
This step is the same

## fetch, build, and install with coinbrew
On this step, may encounter some issues with the build line (not recognizing list, or random number generator). If command-line tools are not there, make sure to install them: 

```sh
xcode-select --install
```

Also, make sure to tell the compiler where the system root is (SDK Path)
```sh
export SDKROOT=$(xcrun --show-sdk-path)
./coinbrew build Ipopt --test --no-prompt --verbosity=3
```
## Add executable file to ``LD_LIBRARY_PATH``

Steps to follow for Mac: 
Replace `path/to` with the path to the COIN-OR where you installed ipopt.


```sh
export IPOPT_DIST="/path/to/COIN-OR/dist"
echo "export PATH=\"\$PATH:$IPOPT_DIST/bin\"" >> ~/.zshrc
echo "export DYLD_LIBRARY_PATH=\"\$DYLD_LIBRARY_PATH:$IPOPT_DIST/lib\"" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$IPOPT_DIST/lib\"" >> ~/.zshrc 
echo "export PKG_CONFIG_PATH=\"\$PKG_CONFIG_PATH:$IPOPT_DIST/lib/pkgconfig\"" >> ~/.zshrc
source ~/.zshrc
```