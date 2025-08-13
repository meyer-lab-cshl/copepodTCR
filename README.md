[![Downloads](https://static.pepy.tech/badge/copepodTCR)](https://pepy.tech/project/copepodTCR)
[![PyPI version](https://img.shields.io/pypi/v/copepodTCR.svg)](https://pypi.org/project/copepodTCR/)
[![Conda Version](https://img.shields.io/conda/vn/vasilisa.kovaleva/copepodTCR?style=flat-square)](https://anaconda.org/vasilisa-kovaleva/copepodTCR)

# COmbinatorial PEptide POoling Design for TCR specificity

T cell receptor (TCR) repertoire diversity enables the antigen-specific immune responses against the vast space of possible pathogens. Identifying TCR-antigen binding pairs from the large TCR repertoire and antigen space is crucial for biomedical research.  Here, we introduce **copepodTCR**, an open-access tool to design and interpret high-throughput experimental TCR specificity assays.

copepodTCR implements a combinatorial peptide pooling scheme for efficient experimental testing of T cell responses against large overlapping peptide libraries, that can be used to identify the specificity of (or "deorphanize") TCRs. The scheme detects experimental errors and, coupled with a hierarchical Bayesian model for unbiased interpretation, identifies the response-eliciting peptide sequence for a TCR of interest out of hundreds of peptides tested using a simple experimental set-up.

Documentation: [copepodTCR.readthedocs](https://copepodtcr.readthedocs.io/en/latest/index.html).

Also you can use [copepodTCR app](https://copepodtcr.cshl.edu/).

## Cite as

Kovaleva V. A., et al. "copepodTCR: Identification of Antigen-Specific T Cell Receptors with combinatorial peptide pooling." bioRxiv (2023): 2023-11.

Or use the following BibTeX entry:

```
@article{
    kovaleva2023copepodtcr,
    title        = {copepodTCR: Identification of Antigen-Specific T Cell Receptors with combinatorial peptide pooling},
    author       = {Kovaleva, Vasilisa A and Pattinson, David J and Barton, Carl and Chapin, Sarah R and Minervina, Anastasia A and Richards, Katherine A and Sant, Andrea J and Thomas, Paul G and Pogorelyy, Mikhail V and Meyer, Hannah V},
    year         = 2023,
    journal      = {bioRxiv},
    publisher    = {Cold Spring Harbor Laboratory},
    pages        = {2023--11}
}
```

## Installation

Can be installed with pip:
```python
pip install copepodTCR
```

or conda: 
```python
conda install -c vasilisa.kovaleva copepodTCR
```

Then you need to install manifold3d, required for 3D modeling of masks. You can skip this step, if you don't plan to print masks for pooling step.

```python
pip install manifold3d
```

Alternative to manifold3d is Blender, it can be installed from [Blender official website](https://www.blender.org/) (version 4.5 and higher).

You can use :func:`cpp.pick_engine()` to check with engines are available in you environment.

### Requirements
Required packages should be installed simulataneously with the copepodTCR packages.

But if they were not, here is the list of requirements:
```python
    pip install "pandas>=1.5.3"
    pip install "numpy>=1.23.5"
    pip install "trimesh>=3.23.5"
    pip install "pymc>=5.9.2"
    pip install "arviz>=0.16.1"
    pip install "matplotlib>=3.10.5"
    pip install "seaborn>=0.13.2"
    pip install "plotly>=6.2.0"

```
