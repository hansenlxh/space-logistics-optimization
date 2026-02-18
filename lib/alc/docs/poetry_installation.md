Assuming python is installed system-wide, install `poetry` by running:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

It should be installed to `$HOME/.local/bin` by default, so no need to set up $PATH. If `which poetry` does not work, try `source ~/.bashrc`, `source ~/.zshrc` or similar.
Alternatively, you can `pip install poetry` inside a virtual environment.
