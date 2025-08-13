.. image:: ./logo.svg

COmbinatorial PEptide POoling Design for TCR specificity
==========================================================

T cell receptor (TCR) repertoire diversity enables the antigen-specific immune responses against the vast space of possible pathogens. Identifying TCR-antigen binding pairs from the large TCR repertoire and antigen space is crucial for biomedical research.  Here, we introduce **copepodTCR**, an open-access tool to design and interpret high-throughput experimental TCR specificity assays.

copepodTCR implements a combinatorial peptide pooling scheme for efficient experimental testing of T cell responses against large overlapping peptide libraries, that can be used to identify the specificity of (or "deorphanize") TCRs. The scheme detects experimental errors and, coupled with a hierarchical Bayesian model for unbiased interpretation, identifies the response-eliciting peptide sequence for a TCR of interest out of hundreds of peptides tested using a simple experimental set-up. 


How to use
----------

The experimental setup starts with defining the protein/proteome of interest and obtaining synthetic peptides tiling its space. Peptide sequences can be generated in silico from a protein of interest and then checked using functions from **Peptides generation and assessment** section.

This set of peptides, containing an overlap of a constant length, is entered into copepodTCR. Parameters for CPP scheme can be selected using functions from **Peptide occurrence search**. It creates a peptide pooling scheme (functions from **Pooling**) and, optionally, provides the pipetting scheme to generate the desired pools as either 384-well mask models which could be further 3D printed and overlay the physical plate or pipette tip box (**3D models** section).

Following this scheme, the peptides are mixed, and the resulting peptide pools tested in a T cell activation assay. The activation of T cells is measured for each peptide pool (experimental layout, activation assay, and experimental read out) with the assay of choice, such as flow cytometry- or microscopy-based activation assays detecting transcription and translation of a reporter gene.

The experimental measurements for each pool are entered back into copepodTCR which employs a Bayesian mixture model to identify activated pools.  Based on the activation patterns, it returns the set of overlapping peptides leading to T cell activation (**Results interpretation with a Bayesian mixture model**). Also they can be displayed using functions from **Plotting results** section.

For more details, refer to "copepodTCR: Identification of Antigen-Specific T Cell Receptors with combinatorial peptide pooling" (`bioRxiv version <https://www.biorxiv.org/content/10.1101/2023.11.28.569052v2>`_).

Algorithm for CPP generation
----------------------------

Algorithms for CPP generation are described in "Unbiased and Error-Detecting Combinatorial Pooling Experiments with Balanced Constant-Weight Gray Codes for Consecutive Positives Detection" (`arXive version <https://arxiv.org/abs/2502.08214>`_). `CodePUB <https://codepub.readthedocs.io/en/latest/Introduction.html>`_ python package accompanies the paper and provides all functions required to use the algorithm.
