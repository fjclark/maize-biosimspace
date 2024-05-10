CLI Tutorials
==============

maize-biosimspace provides a large number of nodes for performing common operations such as parameterisation, minimisation, equilibration, production molecular
dynamics, and alchemical free energy calculations. Nodes are generally engine-specific as maize requires that nodes have a list of ``required_callables`` whose presence
in the environment is checked before the node is run. To see a list of all available nodes, type ``bss_`` and hit tab to list them. To see the options for each, pass
pass the ``-h`` flag e.g.

.. code-block:: bash

   bss_parameterise -h

These tutorials give specific examples of using BioSimSpace maize nodes to run production molecular dynamics, to create equilibrated systems starting from unparameterised
input structures, and to run absolute binding free energy calculations starting from a protein pdb and an sdf file containing multiple ligands.

.. toctree::
   :maxdepth: 1

   tutorial_cli_production_md
   tutorial_cli_system_preparation
   tutorial_cli_abfe
   