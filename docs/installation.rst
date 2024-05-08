Installation
------------
For a basic installation, simply clone this repository and run:

.. code-block:: bash

   git clone https://github.com/fjclark/maize-biosimspace.git
   cd maize-biosimspace
   mamba env create -f env-users.yml
   mamba activate maize-biosimspace
   pip install --no-deps ./

Ensure that the required AMBER, GROMACS, and NAMD executables are available in your environment if you plan to use the relevant nodes.

If you want to keep up-to-date with the latest changes to the core, clone `maize <https://github.com/MolecularAI/maize>`_, switch to the directory, and run (in the same conda environment):

.. code-block:: bash

   pip install --no-deps ./

If you plan on developing, you should use ``env-dev.yml`` instead and use the ``-e`` flag for ``pip``.

.. code-block:: bash

   git clone https://github.com/fjclark/maize-biosimspace.git
   cd maize-biosimspace
   mamba env create -f env-dev.yml
   mamba activate maize-biosimspace
   pip install -e --no-deps ./