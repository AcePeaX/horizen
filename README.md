# Horizen®
Horizen® is an LLM project that aims to help me train on LLMs and transformers.


## Getting started

### Setup python
> Note: you need to have python version >= 3.10 . Execute `python3 --version`to check the version

To start using/developping Horizen®, start by setting up a python virtual environment:
```shell
python3 -m venv .venv
source .venv/bin/activate
```

And then install the dependencies:

```shell
python3 -m pip install torch numpy regex datasets tiktoken
```
or simply 
```shell
python3 -m pip install -r requirements.txt
```

> Note: in the notebooks, don't forget to choose your virtual environment.

### Setup log folder
Quickly create a folder for logs : 
```shell
mkdir logs
```

## Useful commands
Here are some usefull commands:
- `make train` : executes the training script in `scripts/bin`, will use the hyperparameters specified, otherwise reverts to saved file in `saves` directory.
- `make test` : executes the testing script in `scripts/bin`, especially usefull for text generation.
- `make plot` : generates a PNG file containing the plots of the training. Don't forget to put the nave of the model in `plot.py` file in `scripts/bin`.