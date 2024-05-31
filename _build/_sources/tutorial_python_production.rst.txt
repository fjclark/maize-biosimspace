Creating a Workflow with Production MD
=======================================

Here, we'll look at a basic example of a custom workflow which 
uses ``pmemd.cuda`` to run some production MD. In reality, you 
would want to add some extra steps before or afterwards:

.. code-block:: python

    """Run production Molecular Dynamics using PMEMD.CUDA through BioSimSpace."""

    from pathlib import Path

    from maize.core.workflow import Workflow
    from maize.steps.exs.biosimspace import ProductionPmemdCuda
    from maize.steps.io import LoadData, Return
    from maize.utilities.execution import JobResourceConfig

    # Build the graph
    flow = Workflow(name="Prod_BSS_AMBER_Test", cleanup_temp=False, level="debug")

    # Add the nodes
    load_sys = flow.add(LoadData[list[Path]])
    prod_pmemd = flow.add(
        ProductionPmemdCuda,
        name="Production_Amber",
        parameters={
            "runtime": 1.0, # ns
        },
    )
    retu = flow.add(Return[list[Path]])

    # Set parameters
    load_sys.data.set(
        [
            Path(
                "< path to complex.prm7>" # CHANGEME
            ),
            Path(
                "< path to complex.rst7>" # CHANGEME
            ),
        ]
    )

    # Connect the nodes
    flow.connect(load_sys.out, prod_pmemd.inp)
    flow.connect(prod_pmemd.out, retu.inp)

    # Check and run!
    flow.check()
    flow.visualize()
    flow.execute()

    mols = retu.get()

    # Load a BioSimSpace system from the returned paths
    import BioSimSpace as BSS

    sys = BSS.IO.readMolecules([str(mols[0]), str(mols[1])])
    print(40 * "#")
    print(sys)
    # In reality, you would do something here...

