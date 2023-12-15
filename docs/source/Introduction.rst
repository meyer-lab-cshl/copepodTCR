.. image:: ./logo.svg

COmbinatorial PEptide POoling Design for TCR specificity
==========================================================

CopepodTCR helps the user with all stages of the experiment design and interpetation:
- selection of parameters for the experiment (**Balance check**)
- examination of peptides (**Overlap check**)
- generation of pooling scheme (**Pooling scheme**)
- generation of punched cards of efficient peptide mixing (**STL files**)
- results interpetation using hierarchical Bayesian model (**Interpretation**)

Task
----------

Identification of a cognate peptide for TCR of interest is crucial for biomedical research. Current computational efforts for TCR specificity did not produce reliable tool, so testing of large peptide libraries against a T cell bearing TCR of interest remains the main approach in the field.

Testing each peptide against a TCR is reagent- and time-consuming. More efficient approach is peptide mixing in pools according to a combinatorial scheme. Each peptide is added to a unique subset of pools ("address"), which leads to matching activation patterns in T cells stimulated by combinatorial pools.

Efficient combinatorial peptide pooling (CPP) scheme must implement:
- use of overlapping peptide in the assay to cover the whole protein space;
- error detection.


Here, we present copepodTCR -- a tool for design of CPP schemes. CopepodTCR detects experimental errors and, coupled with a hierarchical Bayesian model for unbiased results interpretation, identifies the response-eliciting peptide for a TCR of interest out of hundreds of peptides tested using a simple experimental set-up.

For detailed description of the problem please refer to Kovaleva et al, 2023.

Usage
----------

The experimental setup starts with defining the protein/proteome of interest and obtaining synthetic peptides tiling its space.

This set of peptides, containing an overlap of a constant length, is entered into copepodTCR. It creates a peptide pooling scheme and, optionally, provides the pipetting scheme to generate the desired pools as either 384-well plate layouts or punch card models which could be further 3D printed and overlay the physical plate or pipette tip box.

Following this scheme, the peptides are mixed, and the resulting peptide pools tested in a T cell activation assay. The activation of T cells is measured for each peptide pool (experimental layout, activation assay, and experimental read out) with the assay of choice, such as flow cytometry- or microscopy-based activation assays detecting transcription and translation of a reporter gene.

The experimental measurements for each pool are entered back into copepodTCR which employs a Bayesian mixture model to identify activated pools.  Based on the activation patterns, it returns the set of overlapping peptides leading to T cell activation (Results interpretation).

Branch-and-Bound algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For detailed description of the algorithm and its development refer to Kovaleva et al (2023).

The Branch-and-Bound part of copepodTCR generates a peptide mixing scheme by optimizing the peptide distribution into a predefined number of pools. The distribution of each peptide is encoded into an address (edges in the graph), which connect nodes in the graph (circles) that represent a union between two addresses. The peptide mixing scheme constitutes the path through these unions and connecting addresses that ensure a balanced pool design.

Activation model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed description of the model, refer to Kovaleva et al (2023).

To accurately interpret results of T cell activation assay, copepodTCR utilizes a Bayesian mixture model.

The model considers the activation signal to be drawn from two distinct distributions arising from the activated and non-activated pools and provides the probabilities that the value was drawn from either distribution as a criterion for pool classification.

Algorithm
----------


We designed an algorithm that navigates the peptide space by seeking a Hamiltonian path in its corresponding graph to meet the given constraints. The package offers two versions of this algorithm:

1. A basic search for a Hamiltonian path of a given length, simultaneously checking for union and address uniqueness (**Hamiltonian search (trivial version)**).

2. A faster version based on the same principle, but it condences the path by considering both vertices and edges (**Hamiltonian search (advanced version)**).

Our initial inspiration came from the reflective binary code by Frank Gray. Thus, we have incorporated functions in the package for producing balanced Gray code and its flexible-length option. However, we currently advise against using these for address arrangement due to potential imbalances and non-unique unions.

Gray codes
~~~~~~~~~~~

The generation of different versions of Gray codes are included in the package. Generated arrangements with their help would not satisfy one of the requiremenets described above and in Kovaleva et al, 2023.The package includes:

- generation of Gray Codes
- generation of Balanced Gray Codes (based on `Counting sequences, Gray codes and lexicodes <https://repository.tudelft.nl/islandora/object/uuid%3A975a4a47-7935-4f76-9503-6d4e36b674a3>`_)
- generation of Balanced Gray Codes with flexible length (based on `Balanced Gray Codes With Flexible Lengths <https://ieeexplore.ieee.org/abstract/document/7329924>`_)

