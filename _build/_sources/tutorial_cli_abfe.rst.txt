Absolute Binding Free Energy Calculations
=========================================

Here, we'll run a quick absolute binding free energy calculation for benzene bound to T4 Lysozyme. For this, we'll
use the ``bss_abfe_multi_isomer`` workflow through its CLI, which requires only an SDF file containing all required
ligands, and the pdb of the protein. Check the options with

.. code-block:: bash

   bss_abfe_multi_isomer -h

Copy over the required input:

.. code-block:: bash

   mkdir bss_abfe_example
   cd bss_abfe_example
   cp ../tests/data/benzene.sdf .
   cp ../tests/data/t4l.pdb .

Now, let's run a relatively very short (but still fairly expensive)
ABFE calculation with 0.01 ns of sampling per production window.
Note that you currently need to provide the absolute path to your inputs:

.. code-block:: bash

   bss_abfe_multi_isomer --lig_sdfs_file <INSERT ABSOLUTE PATH TO benzene.sdf> \
                         --protein_pdb_path <INSERT ABSOLUTE PATH to t4l.pdb> \
                         --run_time_abfe 0.01 \
                         --dump_to . \

The results (see the log or ``results.json`` in the ``dump`` directory) should show
a binding free energy of around -5 kcal/mol.

Running through the command line with this many arguments is unweildy,
and some options aren't available through the CLI (for example, the lambda
spacing). It's likely a better option to write a quick script - using the
pre-made workflow directly in python - simply import the workflow factory
, customise the options, and run (all in a python script):

.. code-block:: python
   
   from maize.graphs.exs.biosimspace.abfe_single_node import get_abfe_multi_isomer_workflow
   workflow = getabfe_multi_isomer_workflow()

   # Set workflow options...

   # Run
   workflow.execute()
