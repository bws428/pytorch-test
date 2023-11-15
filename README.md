# Installing PyTorch

Try to install PyTorch using `pip` in a `conda` environment, rather than globally. These instructions assume that Python3 has been installed on your system using [Anaconda](https://www.anaconda.com/download/).

First, create a new project:

```zsh
touch my-torch-project
cd my-torch-project
```

Create a Conda environment named `env_pytorch` (or whatever) using:

```zsh
conda create -n env_pytorch python=3.9
```

Activate the environment using:

```zsh
conda activate env_pytorch
```

Now install PyTorch using pip:

```zsh
pip install torchvision
```

Note: This will install both torch and torchvision.

## Test the installation

The file `torch-test.py` is just a simple script to test if PyTorch has installed correctly.

The output should be something similar to:

```zsh
tensor([[0.5807, 0.3010, 0.3725],
        [0.8028, 0.8234, 0.4467],
        [0.8980, 0.7338, 0.8621],
        [0.6313, 0.1638, 0.1598],
        [0.6147, 0.8680, 0.7573]])
```

## Getting started

Learn the basics of ML and PyTorch:

https://pytorch.org/tutorials/beginner/basics/intro.html

## Cleaning up

When you're done working in this environment, it can be deactivated with:

```zsh
conda deactivate
```
