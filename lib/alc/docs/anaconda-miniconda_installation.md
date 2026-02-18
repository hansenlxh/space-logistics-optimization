Anaconda/Miniconda is used for environment management (packages/dependencies are separately managed via `poetry`) here.
Run the following to install miniconda on Linux (replace `zsh` to the shell of your choice):

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh # replace "zsh" if needed
```

then, `source ~/.bashrc` or `source ~/.zshrc` as always.
