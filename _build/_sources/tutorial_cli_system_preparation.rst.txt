System Preparation
==================

Often, we have structure files for a protein and / or ligand and we would like to parameterise, solvate, minimise, heat, and equilibrate them to obtain systems
suitable for production molecular dynamics simulations or free energy calculations. While individual nodes are provided for all of these steps, 
`maize-biosimspace` also provides CLIs for two complete system preparation workflows: ``bss_system_prep_free``, which is designed to prepare a ligand in a box
of water (but can also be used for an apo protein) and ``bss_system_prep_bound``, which is designed to set up protein-ligand complexes. 

Here, we'll prepare the complex of T4 lysozyme L99A bound to benzene, a very common test system for absolute binding free energy calculations. First, we'll copy 
over the required input:

.. code-block:: bash

   mkdir sysprep_bound_example
   cd sysprep_bound_example
   cp ../tests/data/benzene.sdf .
   cp ../tests/data/t4l.pdb .

.. tip::

   This pdb has been sanitised and will work with BioSimSpace first time, but often pdbs will require some tweaking before they are accepted by ``tleap`` (which 
   BioSimSpace uses behind the scenes). The recommended workflow is

   * Clean your unsanitised pdb using pdb4amber, e.g. ``pdb4amber -i protein.pdb -o protein_sanitised.pdb``
   * Attempt to run the workflow below
   * If the workflow raises an error, attempt parameterisation directly with ``tleap`` to get more detailed error messages. E.g., type ``tleap``, then

   .. code-block:: bash

      source leaprc.protein.ff14SB
      source leaprc.water.tip3p
      # Loading an unsanitised pdb will likely raise an error
      prot = loadpdb protein_sanitised.pdb
      saveamberparm prot protein.parm7 protein.rst7
      savepdb prot protein_fully_sanitised.pdb

   * If the above fails, this is often due to residue/ atom names which do not match the templates. Read the errors to find out which residues / atoms are causing the issues, then check the expected names in library which was loaded after typing ``source leaprc.protein.ff14SB`` e.g. ``cat $AMBERHOME/dat/leap/lib/amino12.lib``.  Rename the offending atoms/ residues and repeat the above step.

   BioSimSpace is very fussy about parameterisation and will fail if tleap raises any warnings. To get round this, run the tleap script above and use the
   ``protein_full_sanitised.pdb`` file as your input, which will not raise any errors.

To run system preparation for our protein-ligand complex, we'll use the ``bss_system_prep_bound`` CLI, saving the output system to "t4l_benzene_complex_equilibrated"
and using the gaff2 ff14SB force fields. There are a large number of other parameters which can be modified (see ``bss_system_prep_bound -``) but we'll run the 
defaults for now.

.. note::

   Make sure that you have access to a GPU locally, or have configured `Maize` to submit to a queue with gpu access (see :doc:`configuration`)

.. code-block:: bash

   bss_system_prep_bound --inp benzene.sdf --protein_pdb t4l.pdb --ligand_force_field gaff2 --protein_force_field ff14SB --save_name t4l_benezene_complex_equilibrated

