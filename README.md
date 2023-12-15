<span style="color:white"> </span>
## <span style="color:#015396">COmbinatorial PEptide POoling Design for TCR specificity</span>
<span style="color:#015396">CopepodTCR</span> is a tool for the design of combinatorial peptide pooling schemes for TCR speficity assays.

<span style="color:#015396">CopepodTCR</span> guides the user through all stages of the experiment design and interpetation:
- selection of parameters for the experiment (**Balance check**)
- examination of peptides (**Overlap check**)
- generation of pooling scheme (**Pooling scheme**)
- generation of punched cards of efficient peptide mixing (**STL files**)
- results interpetation using hierarchical Bayesian model (**Interpretation**)

# <span style="color:#015396">Cite as</span>

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

# <span style="color:#015396">Description</span>

Identification of a cognate peptide for TCR of interest is crucial for biomedical research. Current computational efforts for TCR specificity did not produce reliable tool, so testing of large peptide libraries against a T cell bearing TCR of interest remains the main approach in the field.

Testing each peptide against a TCR is reagent- and time-consuming. More efficient approach is peptide mixing in pools according to a combinatorial scheme. Each peptide is added to a unique subset of pools ("address"), which leads to matching activation patterns in T cells stimulated by combinatorial pools.

Efficient combinatorial peptide pooling (CPP) scheme must implement:
- use of overlapping peptide in the assay to cover the whole protein space;
- error detection.

Here, we present <span style="color:#015396">CopepodTCR</span> -- a tool for design of CPP schemes. CopepodTCR detects experimental errors and, coupled with a hierarchical Bayesian model for unbiased results interpretation, identifies the response-eliciting peptide for a TCR of interest out of hundreds of peptides tested using a simple experimental set-up.

Detailed instructions please see at [copepodTCR.readthedocs](https://copepodtcr.readthedocs.io/en/latest/index.html). Also you can use [copepodTCR app](https://copepodtcr.cshl.edu/).


# <span style="color:#015396">Usage</span>

The experimental setup starts with defining the protein/proteome of interest and obtaining synthetic peptides tiling its space.

This set of peptides, containing an overlap of a constant length, is entered into copepodTCR. It creates a peptide pooling scheme and, optionally, provides the pipetting scheme to generate the desired pools as either 384-well plate layouts or punch card models which could be further 3D printed and overlay the physical plate or pipette tip box.

Following this scheme, the peptides are mixed, and the resulting peptide pools tested in a T cell activation assay. The activation of T cells is measured for each peptide pool (experimental layout, activation assay, and experimental read out) with the assay of choice, such as flow cytometry- or microscopy-based activation assays detecting transcription and translation of a reporter gene.

The experimental measurements for each pool are entered back into copepodTCR which employs a Bayesian mixture model to identify activated pools.  Based on the activation patterns, it returns the set of overlapping peptides leading to T cell activation (Results interpretation).

# <span style="color:#015396">Branch-and-Bound algorithm</span>

For detailed description of the algorithm and its development refer to Kovaleva et al (2023).

The Branch-and-Bound part of copepodTCR generates a peptide mixing scheme by optimizing the peptide distribution into a predefined number of pools. The distribution of each peptide is encoded into an address (edges in the graph), which connect nodes in the graph (circles) that represent a union between two addresses. The peptide mixing scheme constitutes the path through these unions and connecting addresses that ensure a balanced pool design.

# <span style="color:#015396">Activation model</span>

For detailed description of the model, refer to Kovaleva et al (2023).

To accurately interpret results of T cell activation assay, copepodTCR utilizes a Bayesian mixture model.

The model considers the activation signal to be drawn from two distinct distributions arising from the activated and non-activated pools and provides the probabilities that the value was drawn from either distribution as a criterion for pool classification.

# <span style="color:#015396">CopepodTCR Python package</span>

Can be installed with
```python
pip install copepodTCR
```

or 
```python
conda install -c vasilisa.kovaleva copepodTCR
```

### <span style="color:#015396">Requirements</span>
Required packages should be installed simulataneously with the copepodTCR packages.

But if they were not, here is the list of requirements:
```python
    pip install "pandas>=1.5.3"
    pip install "numpy>=1.23.5"
    pip install "cvxpy>=1.3.2"
    pip install "trimesh>=3.23.5"
    pip install "trimesh>=3.23.5"
    pip install "pymc>=5.9.2"
    pip install "arviz>=0.16.1"
```
