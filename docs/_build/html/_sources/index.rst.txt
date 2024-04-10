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

Installation
------------
To install, simply clone this repository and run:

.. code-block:: bash

   mamba env create -f env-users.yml
   mamba activate maize-biosimspace
   pip install --no-deps ./

If you want to keep up-to-date with the latest changes to the core, clone `maize <https://github.com/MolecularAI/maize>`_, switch to the directory, and run (in the same conda environment):

.. code-block:: bash

   pip install --no-deps ./

If you plan on developing, you should use ``env-dev.yml`` instead and use the ``-e`` flag for ``pip``.

Configuration
-------------
Each step documentation will contain information on how to setup and run the node, as well as install the required dependencies. Dependencies can be managed in several ways, depending on the node and workflow you are running:

* Through a ``module`` system:

  Specify a module providing an executable in the ``config.toml`` (see `Configuring workflows <https://molecularai.github.io/maize/docs/userguide.html#config-workflow>`_). This module will then be loaded in the process running the node.

* With a separate python environment:

  Some nodes will require custom python environments that are likely to be incompatible with the other environments. In those cases, the node process can be spawned in a custom environment. Note that this environment must still contain *maize*. Required custom environments can be found in the appropriate node directory.

* By specifying the executable location and possibly script interpreter. This can also be accomplished using ``config.toml`` (see `Configuring workflows <https://molecularai.github.io/maize/docs/userguide.html#config-workflow>`_).


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
