Installation
=============================

1. Install copepodTCR.

	With pip:

	.. code-block:: python

		pip install copepodTCR

	Or with conda:

	.. code-block:: python

		conda install -c vasilisa.kovaleva copepodTCR

2. Instal dependencies for 3D modeling of masks. You can skip this step, if you don't plan to use function from **3D model** section.
	
	After installing copepodTCR either way, install manifold3d:

	.. code-block:: python

		pip install manifold3d

	Alternative to manifold3d is Blender, it can be installed from `Blender official website <https://www.blender.org/>`_ (version 4.5 and higher).

	You can use :func:`cpp.pick_engine()` to check with engines are available in you environment.


Requirements
------------

Except for manifold3d, required packages should be installed simulataneously with the copepodTCR packages.

But if they were not, here is the list of requirements:

* pandas>=1.5.3

	.. code-block:: python

		pip install "pandas>=1.5.3"

* numpy>=1.23.5

	.. code-block:: python

		pip install "numpy>=1.23.5"

* trimesh>=3.23.5

	.. code-block:: python

		pip install "trimesh>=3.23.5"

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