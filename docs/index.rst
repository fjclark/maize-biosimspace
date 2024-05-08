.. maize-biosimspace documentation master file, created by
   sphinx-quickstart on Wed Mar  8 17:24:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

maize-biosimspace
=================
*maize* is a graph-based workflow manager for computational chemistry pipelines. This repository contains a namespace package providing BioSimSpace extensions for *maize*. You can find the core maize documentation `here <https://molecularai.github.io/maize>`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Setup

   installation
   configuration

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   production-md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   steps/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Core

   Steps <https://molecularai.github.io/maize/docs/steps>
   Maize <https://molecularai.github.io/maize>

Quick Start
-----------
Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/fjclark/maize-biosimspace.git
   cd maize-biosimspace
   mamba env create -f env-users.yml
   mamba activate maize-biosimspace
   pip install --no-deps ./

Ensure that the required AMBER, GROMACS, and NAMD executables are available in your environment if you plan to use the relevant nodes.

Many maize-biosimspace workflows will now be available through CLIs. Try typing `bss_` and hitting tab to list them. To see the options for each, type e.g.

.. code-block:: bash

   bss_parameterise -h

To run production 0.1 ns of production MD on a protein-ligand complex using GROMACS, dumping all of the intermediate files to the current directory:

.. code-block:: bash

   mkdir gmx_md_example
   cd gmx_md_example
   cp ../tests/data/complex.* .
   bss_production_gromacs --inp complex.prm7 complex.rst7 --runtime 0.1 --save_name gmx_md_out --dump_to .

This will run locally. To configure maize nodes to run through a scheduler such as slurm, see :doc:`configuration`.



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
