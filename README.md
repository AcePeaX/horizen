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