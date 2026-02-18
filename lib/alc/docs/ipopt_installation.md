# Install prerequisites

(most Linux distros already have them installed)

```sh
sudo apt install pkg-config gcc g++ gfortran libblas-dev liblapack-dev
```

# Install Ipopt with coinbrew

## Get `coinbrew`

For more `coinbrew` commands, refer to [COIN-OR User Guide](https://coin-or.github.io/user_introduction) and [coinbrew](https://coin-or.github.io/coinbrew/).
First, go to dir you wanna install ipopt

```sh
cd ~
mkdir COIN-OR && cd COIN-OR
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
```

## fetch, build, and install with coinbrew

```sh
./coinbrew fetch Ipopt --no-prompt
./coinbrew build Ipopt --test --no-prompt --verbosity=3
sudo ./coinbrew install Ipopt --no-prompt
```

# Add executable file to `LD_LIBRARY_PATH`

For example, with zsh:

```sh
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/COIN-OR/dist/lib"' >> ~/.zshrc
echo 'export PATH="$PATH:$HOME/COIN-OR/dist/bin"' >> ~/.zshrc
source ~/.zshrc
```

The actual path will be different.
The path that you need to add to the path should be printed when you finish installing ipopt successfully.
Find out duplicates:

```sh
find "$HOME" /usr/local /usr -name 'libipopt.so*' 2>/dev/null
```
