
[Demo Video](https://youtu.be/DrdrvsZLmA8)

# Time-Series Analysis with Streaming Functional Data Analysis (FDA) using Incremental and Progressive Magnitude-Shape Plot

Implementation of incremental and progressive Magnitude-Shape plot for analyzing streaming time-series data


## About

- A Hardware System Monitoring Approach with Streaming [Functional Data Analysis](https://fda.readthedocs.io/en/latest/auto_examples/index.html)
- To perform FDA while addressing the computational problem, we introduce new incremental and progressive algorithms that promptly generate the magnitude-shape (MS) plot, which conveys both the functional magnitude and shape outlyingness of time series data. 
- Features
    - A python implementation of magnitude-shape oulyingness measures for univariate time-series. 
        - W. Dai and M. G. Genton, “Multivariate functional data visualization and outlier detection,” J. Comput. Graph. Stat., vol. 27, no. 4,pp. 923–934, 2018.
    - This implementation **incrementally** updates new incoming time points and **progressively** updates (with or without approximation) the new time-series (i.e., functions) for fast streaming input data.
    - For FPCA refer [here](https://cran.r-project.org/web/packages/fdapace/vignettes/fdapaceVig.html).
    
    
## Requirements
Python3, Numpy, Cython, cffi, scipy, scikit-fda

Note: Tested on macOS Monterey.


## Setup 

``` pip install -r requirements.txt ```

### To install incremental and progressive MS: 

` cd ./path-to-folder/ `

```python setup.py install```


### Usage:

#### With Python3

Import *inc_ms_fda* and *prog_ms_fda* from python (from inc_ms_fda import IncFDO and from prog_ms_fda import ProgressiveFDA). See sample.ipynb (jupyter notebook).


## How to Cite:
Shilpika, F., Fujiwara, T., Sakamoto, N., Nonaka, J., & Ma, K.-L. (2021). A Visual Analytics Approach for Hardware System Monitoring with Streaming Functional Data Analysis.
