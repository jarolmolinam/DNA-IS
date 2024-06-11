# DNA-IS
# DNA Ionic Strength transferable DNA-CG model
This code implements an ionic strength transferable DNA coarse-grained model using graph neural networks.


## Installation

We recommend creating a new enviroment

```bash
conda create -f env.yml
conda activate isnff
```

This package requires:
- numpy
- pytorch


Then clone and install this github repository on Linux

```bash
git clone https://github.com/avasquee/IS-NFF/
cd IS-NFF
pip install .
```


## Examples
Look in the folder notebooks for the files "...ipynb" or "....ipynb". Open any this notebooks and use the isnff environment.


## Usage
In python, a toy FFS is run as follows

```python
import isnff as nff



```



## Authors
This project was produced in the Laboratory of Molecular and Computational Genomics at University of Wisconsin- Madison

Alejandro Vasquez-Echeverri - vasquezechev@wisc.edu

Jarol Molina - jarolmolina@gmail.com

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgment

This code is based on T-NFF (url) and NeuralForceField (urls)
