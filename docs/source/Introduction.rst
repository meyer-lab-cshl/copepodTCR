.. image:: ./logo.svg

COmbinatorial PEptide POoling Design for TCR specificity
==========================================================

T cell receptor (TCR) repertoire diversity enables antigen-specific immune responses against many possible pathogens. Identifying TCR-antigen binding pairs is important for biomedical research. Here, we introduce **copepodTCR**, an open-access tool for designing and interpreting high-throughput TCR specificity assays.

copepodTCR designs combinatorial peptide-pooling (CPP) schemes for testing large overlapping and non-overlapping peptide libraries. These schemes help identify the peptide (from a tested library) that activates a TCR of interest. The package also supports experimental error detection and helps with CPP results interpretation using Bayseian Mixture model and a decision-tree algorithm.


How to use
----------

The experimental setup starts with defining the protein/proteome of interest and obtaining overlapping synthetic peptides that tile the protein or proteome sequence. Peptide sequences can be generated in silico from a protein of interest and then checked using functions from **Peptides generation and assessment** section.

copepodTCR uses these peptides to generate a peptide-pooling scheme. It can also generate 3D-printable mask models that help mix these peptides from a plate into pools.

Following this scheme, peptides are mixed into pools and tested in a T cell activation assay. The activation of T cells is measured for each peptide pool with the assay of choice, such as flow cytometry- or microscopy-based activation assays detecting transcription and translation of a reporter gene.

The experimental measurements for each pool are entered back into copepodTCR which employs a Bayesian mixture model to identify activated pools. Based on the activation patterns, it returns the set of overlapping peptides leading to T cell activation (**Results interpretation with a Bayesian mixture model**). The results can also be visualized with the functions in Plotting results.

For more details, refer to «copepodTCR: Identification of Antigen-Specific T Cell Receptors with combinatorial peptide pooling» (`bioRxiv version <https://www.biorxiv.org/content/10.1101/2023.11.28.569052v3>`_).

Algorithm for CPP generation
----------------------------

Algorithms for CPP generation are described in "Unbiased and Error-Detecting Combinatorial Pooling Experiments with Balanced Constant-Weight Gray Codes for Consecutive Positives Detection" (`published in Bioinformatics <https://doi.org/10.1093/bioinformatics/btaf611>`_). `CodePUB <https://codepub.readthedocs.io/en/latest/Introduction.html>`_ python package accompanies the paper and provides all functions required to use the algorithm.
