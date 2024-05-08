Configuration
-------------
For a detailed guide to configuring Maize, please see the `maize documentation <https://molecularai.github.io/maize/docs/userguide.html#configuring-workflows>`_
and the `maize-contrib documentation <https://molecularai.github.io/maize-contrib/docking.html#Configuration>`_.

By default, Maize looks for `$XDG_CONFIG_HOME/maize.toml` (`~/.config/maize.toml`) for configuration information. To set up Maize to run through slurm, run

.. code-block:: bash

   export XDG_CONFIG_HOME=~/.config

Then create `~/.config/maize.toml` containing the following

.. code-block:: toml

  system = "slurm"  # Can be one of {'cobalt', 'flux', 'local', 'lsf', 'pbspro', 'rp', 'slurm'}
  max_jobs = 100  # The maximum number of jobs that can be submitted by a node at once
  queue = "gpu" #CHANGEME to your desired GPU queue
  launcher = "srun"  # The launcher to use for the command, usually one of {'srun', 'mpirun', 'mpiexec'}
  walltime = "24:00:00"  # Job walltime limit, shorter times may improve queueing times

Now when you execute a Maize workflow, it will submit jobs to the slurm scheduler.