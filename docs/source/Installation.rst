Installation
=============================

1. Install copepodTCR.

   With pip:

   .. code-block:: python

      pip install copepodTCR

   Or with conda:

   .. code-block:: python

      conda install -c vasilisa.kovaleva copepodTCR

2. Install dependencies for 3D mask modeling. You can skip this step if you do not plan to use functions from the **3D models** section.

   After installing copepodTCR either way, install manifold3d:

   .. code-block:: python

      pip install manifold3d

   Blender is an alternative to manifold3d. It can be installed from the `Blender official website <https://www.blender.org/>`_ (version 4.5 or higher).

   Use :func:`cpp.pick_engine()` to check which boolean-operation engines are available in your environment.


Requirements
------------

Except for manifold3d, the required packages should be installed automatically with copepodTCR. If they are missing, install them manually.

* pandas>=1.5.3

  .. code-block:: python

     pip install "pandas>=1.5.3"

* numpy>=1.23.5

  .. code-block:: python

     pip install "numpy>=1.23.5"

* codepub>=2.3

  .. code-block:: python

     pip install "codepub>=2.3"

* trimesh>=4.7.1

  .. code-block:: python

     pip install "trimesh>=4.7.1"

* PyMC>=5.9.2

  .. code-block:: python

     pip install "pymc>=5.9.2"

* Arviz>=0.16.1

  .. code-block:: python

     pip install "arviz>=0.16.1"

* matplotlib>=3.10.5

  .. code-block:: python

     pip install "matplotlib>=3.10.5"

* seaborn>=0.13.2

  .. code-block:: python

     pip install "seaborn>=0.13.2"

* plotly>=6.2.0

  .. code-block:: python

     pip install "plotly>=6.2.0"
