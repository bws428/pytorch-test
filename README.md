## Machine Learning Development Setup for MacOS (Apple Silicon)

The following tutorial is designed to take you from "zero to hero" and get your Apple Silicon Mac setup for Machine Learning experimentation and development. The goal is to provide the instructions I wish I had had when I first got started with software development on the Mac.

The end result of this particular tutorial is the installation of the open-source [Fooocus](https://github.com/lllyasviel/Fooocus) text-to-image generator (similar to [Stable Diffusion](https://stablediffusionweb.com/) and [Midjourney](https://www.midjourney.com/)), but feel free to stop or diverge at any point to get the specific setup you're looking for.

# 1. Set up a coding environment

The following software will get you up and running for general software development on the Mac. Once you get the hang of them, you'll find these programs useful in day-to-day Mac operations, even when you're not writing code.

## Install VS Code

Visual Studio Code is a lightweight but powerful source code editor which runs on your desktop. It comes with built-in support for JavaScript, TypeScript and Node.js and has a rich ecosystem of extensions for other languages and runtimes (such as Python, PHP, Go, Rust, etc.).

https://code.visualstudio.com/

Begin your journey with VS Code with these [introductory videos](https://code.visualstudio.com/docs/introvideos/overview).

## Install iTerm2

iTerm2 is a replacement for the default Terminal app on MacOS. It works with macOS 10.14 or newer. iTerm2 brings the terminal into the modern age with features you never knew you always wanted.

https://iterm2.com/

This is the program you'll use to type in shell commands and run programs, so whenever you see something like, "Paste this command in a macOS Terminal or Linux shell prompt," this is the program you'll use to do it.

## Install Oh My Zsh

Oh My Zsh is a delightful, open source, community-driven framework for managing your terminal shell configuration. It's basically like hanging RGB lights in iTerm.

https://ohmyz.sh/

## Get comfortable with iTerm2 and basic shell commands

[Here's a good cheatsheet](https://www.lifewire.com/mac-terminal-commands-4774997) to help remembering some of the common terminal commands on Mac and Linux. (NOTE: Try NOT to use the `sudo` command unless it's specifically called out for some reason.)

Here are a few commands to get you going:

```sh
# Print the working (current) directory path
pwd
# List all the files and folders in the current directory
ls
# Long (detailed) listing, including hidden files/folders
ls -la
# Change directories/folders
cd <directory> (or </full/path/to/directory>)
# Go up one level
cd ..
# Go back to your home directory
cd ~
# Clear the screen
clear
# Create a new folder/directory in the current directory
mkdir <my_directory>
# Open the current directory in VS Code
code .
```

## Install Homebrew

Homebrew is the missing package manager for MacOS. It's like `apt-get` on Linux, but better.

https://brew.sh/

Homebrew will allow you to install all kinds of tools that your Mac should have come with, but didn't. For example:

```zsh
brew install fortune
brew install gh
```

If you want to find out more about a package before you install it, you can use `brew info`, for example:

```sh
brew info nmap
```

To list all of the packages that you have installed with `brew`, type:

```sh
brew list
```

For more help using Homebrew, you can [read the documentation](https://docs.brew.sh/) online, or just type `brew help` in your terminal.

# 2. Get Python & Conda for scientific computing

## Install Miniconda3

[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) is a free minimal installer for [Conda](https://docs.conda.io/en/latest/), the most popular package manager for scientific computing. It is a small bootstrap version of the more full-featured (maybe bloated?) [Anaconda](https://www.anaconda.com/) platform that includes only `conda`, Python, the packages they both depend on, and a small number of other useful packages (like `pip`, `zlib`, and a few others). If you need more packages, use the `conda install` command to install from thousands of packages available by default in Anaconda’s public repo.

In iTerm2, run the following commands, one after the other:

```bash
mkdir -p ~/.miniconda3
```

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/.miniconda3/miniconda.sh
```

```bash
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
```

```zsh
rm -rf ~/.miniconda3/miniconda.sh
```

```bash
~/.miniconda3/bin/conda init zsh
```

Finally, close iTerm2 and restart it, to re-initialize the shell.

## Managing Conda Environments

When you restart your terminal after installing `conda`, the command prompt should look a little different. Something like this perhaps:

```zsh
(base) ➜  ~
```

The word `(base)` indicates that a `conda` [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is activated. Think of environments as a sandboxed area where you can install specific versions of Python and other packages that you may not want to conflict with other coding projects, or with system functions on your Mac. Whenever you're starting a new Python project, you should create and/or activate a new (or existing) `conda` environment so that nothing you install will conflict with existing projects.

You can deactivate the `base` environment by typing:

```zsh
conda deactivate
```

Now your shell prompt should go back to normal. If you want to list all the available `conda` environments, you can type:

```zsh
conda env list
```

To start one of the listed environments (probably only `base` for now), you type:

```zsh
conda activate base
```

With a `conda` environment activated, you should be able to view the currently installed version of Python 3 by running:

```sh
python3 -V
```

# 3. Install PyTorch for Machine Learning

[PyTorch](https://pytorch.org/) is an open-source machine learning framework use for training deep neural networks, specifically deep learning models used in applications like image recognition and language processing. Written in Python, it's relatively easy for most machine learning developers to learn and use.

We'll use `conda` to install PyTorch and all of its required dependencies.

1.  First, we'll create a directory to hold our PyTorch projects. It can be anything you like, but as an example:

```sh
mkdir ~/Projects/pytorch
cd ~/Projects/pytorch
```

2. Next, we'll create and activate a `conda` environment for installing PyTorch:

```sh
conda create -n pytorch
conda activate pytorch
```

3.  Now we should see our command prompt change to:

```sh
(pytorch) ➜  pytorch
```

4.  This means our new `pytorch` environment is activated, and we can proceed to start installing PyTorch and its dependencies:

```sh
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

## Test PyTorch and Metal GPU Acceleration

PyTorch uses the new [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/) (MPS) backend for GPU training acceleration on Apple Silicon Macs.

Here we will write a short Python script to confirm that PyTorch was installed correctly, and that the MPS GPU acceleration is working on your Mac.

From your `~/Projects/pytorch` directory (or wherever you created your project), we'll use VS Code to create a new Python script:

```sh
code torch-test.py
```

This command should open VS Code and create a new file called `torch-test.py`. Inside this file, paste the following code:

```python
# A simple test to ensure that pytorch is installed in the
# currently active conda environment.
#
# Run this file with: `python3 torch-test.py`
import torch

# This part ensures that the Mac Silicon Metal Performance Shaders (MPS)
# are accessible by PyTorch. https://developer.apple.com/metal/pytorch/
#
# Output should be: `tensor([1.], device='mps:0')`
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

Save the file, then go back to the terminal and run the file with:

```sh
python3 torch-test.py
```

The result of running this file should be:

```sh
tensor([1.], device='mps:0')
```

If you get this, then PyTorch is installed and working correctly.

# 4. Install Fooocus

In this last section, we are going to clone the [Fooocus repository](https://github.com/lllyasviel/Fooocus) from GitHub.com, and then proceed to follow the installation instructions for Mac, as described in the Fooocus README file:

https://github.com/lllyasviel/Fooocus#mac

We are going to create and activate a NEW `conda` enviroment, install PyTorch again into the new environment (just like in Step 3), and then try to run Fooocus.

### Clone the Fooocus repository

Let's start by navigating to our `Projects` directory (or whatever you created in Step 3). We'll store our new Fooocus project there.

```sh
cd ~/Projects
```

Next, we can clone the Fooocus repo (use only ONE of the following two commands):

```sh
git clone https://github.com/lllyasviel/Fooocus.git
```

Alternatively, if you happened to earn extra credit in Step 1 by installing and configuring the [GitHub CLI](https://cli.github.com/) (`gh`), you could clone the repo with the following command (either one of these should work).

```sh
gh repo clone lllyasviel/Fooocus
```

Finally, after we've cloned the Fooocus repo to our Mac, we'll change into the project's directory to prepare for the next steps:

```sh
cd Fooocus
```

### Create a new Conda Environment

Now we'll create a new `conda` environment called `fooocus` using the included `environment.yaml` file, which will provide some baseline configuration for this project.

```sh
conda env create -f environment.yaml
```

Activate the environment:

```sh
conda activate fooocus
```

We should see the terminal command prompt change to `(fooocus)`, indicating that our new environment has been activated.

```sh
(fooocus) ➜  Fooocus
```

### Install PyTorch (again) in the Fooocus Environment

Do we really need to do this? I don't think the .enviroment.yaml does it...
