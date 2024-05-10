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
ABFE calculation with 2 replicates:

.. code-block:: bash

   bss_abfe_multi_isomer --lig_sdfs_file benzene.sdf \
                         --protein_pdb benzene.t4l \
                         --ligand_force_field gaff2 \
                         --protein_force_field ff14SB \
                         --abfe_timestep 4 \
                         --abfe_n_replicates 2 \
                         --abfe_runtime 0.1 \
                         --abfe_runtime_generate_boresch_restraint 0.1 \
                         --prep_runtime_restrained_npt 0.05 \
                         --prep_runtime_unrestrained_npt 0.05 \
                         --abfe_estimator TI \
                         --results_file_name abfe_out \

The ``abfe_out`` file should show results around -4 kcal / mol.

Running through the command line with this many arguments is unweildy,
and some options aren't available through the CLI (for example, the lambda
spacing). It's likely a better option to write a quick script - using the
pre-made workflow directly in python - simply import the workflow factory
, customise the options, and run (all in a python script):

.. code-block:: python
   
   from maize.graphs.exs.biosimspace.afe import getabfe_multi_isomer_workflow
   workflow = getabfe_multi_isomer_workflow()

   # Set workflow options...

   # Run
   workflow.execute()
