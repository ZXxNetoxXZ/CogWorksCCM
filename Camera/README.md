# Camera

This provides a simple interface, via OpenCV, for configuring and utilizing a camera device, using Python


## Installation Instructions
We will need to install OpenCV with the Python bindings so that we can access laptop cameras via our Python code. Follow the instructions for Windows and Mac.

### Windows Instructions (Python 3.{2-6})
Requires: Anaconda + Python 3.{2-6}, numpy, python-opencv

- Create a new conda environment:
    * `conda create --name myenv` replacing myenv with your environment's name.

- Switch to that environment using
    * `source activate myenv`

- Install anaconda in that new environment
    * `conda install anaconda`

#### Installing opencv (Recommended)
```shell
conda install -c conda-forge opencv
```

#### Installing opencv (Back-Up Plan)
Installing python-opencv:

 1. Download the appropriate (32-bit or 64-bit) wheel package from 
   - http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

 2. Navigate to the directory containing this .whl file, and simply run `pip install <whl-file-name>`
  > If you have multiple conda envs, make sure that the appropriate one is active when running the pip-install!

### Mac OS X Instructions (Python 3)
Requires: Anaconda + Python 3 (Tested on 3.{5-6}) + Homebrew

#### Installing opencv:
- Create a new conda environment:
    * `conda create --name myenv` replacing myenv with your environment's name.

- Switch to that environment using
    * `source activate myenv`

- Install anaconda in that new environment
    * `conda install anaconda`

- install [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels) in that environment
    * `conda install nb_conda_kernels`
    * Now, in jupyter notebooks, you can see which environments you are in! Make SURE you are in the new conda environment.
    * You can check what kernel you're in by clicking the new 'conda' tab and seeing which one is checked.
    * to switch environments in a notebook, go to kernel > change kernel > and then selecting your new environment.
    
- `conda install -c anaconda opencv`

Clone Camera, navigate to the resulting directory, and run

```shell
python setup.py develop
```

## Usage
Please see the camera tutorial notebookin this repo for details of how to configure your camera.

```python
%matplotlib notebook
from camera import take_picture
import matplotlib.pyplot as plt
img_array = take_picture()  # returns shape-(H, W, C) array

fig,ax = plt.subplots()
ax.imshow(img_array)
```
