# A simple test to ensure that pytorch is installed in the
# currently active conda environment.
#
#
# First run: `conda activate pytorch` (or whatever the env name is)
#
# Run this file with: `python3 torch-test.py`

import torch

x = torch.rand(5,3)
print(x)

# This part ensures that the Mac Silicon Metal Performance Shaders (MPS)
# are accessible by PyTorch.  https://developer.apple.com/metal/pytorch/
#
# Output should be:  `tensor([1.], device='mps:0')`
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
