Production Molecular Dynamics
=============================

Here, we'll run production molecular dynamics on a protein-ligand complex. To check the available production CLIs, type ``bss_production`` and hit tab:

.. code-block:: bash

   bss_production_gromacs     bss_production_pmemd       bss_production_pmemd_cuda  bss_production_sander      bss_production_somd     

We'll pick gromacs. To check the available arguments and defaults, run

.. code-block:: bash

   bss_production_gromacs -h

We'll run a quick 0.1 ns of production molecular dynamics on the protein-ligand complex included with `maize-biosimspace` for testing. We'll specify the output 
name to be ``gmx_md_out`` and we'll save all of the intermediate files (including input scripts, logs, and trajectory files) to a subdirectory in the current
working directory by specifying ``--dump_to .``

.. code-block:: bash

   mkdir gmx_md_example
   cd gmx_md_example
   cp ../tests/data/complex.* .
   bss_production_gromacs --inp complex.prm7 complex.rst7 --runtime 0.1 --save_name gmx_md_out --dump_to .

You should now have the final coordinate file, ``gmx_md_out.rst7``, a copy of the input topology file ``gmx_md_out.prm7``, and a sub-directory containing all of the
intermediate files. Note that despite running through GROMACS, we were able to pass in AMBER files as input. This is because BioSimSpace automatically converts
between file formats (using Sire under the hood).

