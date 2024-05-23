"""The recommended workflow to use for ABFE calculations with BioSimSpace."""

from pathlib import Path

from maize.core.workflow import Workflow, expose
from maize.steps.exs.biosimspace._utils import SdfPathtoIsomerList
from maize.steps.exs.biosimspace.abfe import BssAbfe
from maize.steps.io import LoadData, Void


def get_abfe_multi_isomer_workflow() -> Workflow:
    """
    A workflow to perform absolute binding free energy calculations on a set
    of isomers.
    """

    flow = Workflow(name="absolute_binding_free_energy_multi_isomer", cleanup_temp=False)

    load_data = flow.add(LoadData[Path], name="LoadSdfPath")

    # Convert to isomers
    sdf_to_isomers = flow.add(SdfPathtoIsomerList, name="SdfPathtoIsomerList")
    abfe_multi_isomer = flow.add(BssAbfe, name="AbsoluteBindingFreeEnergy")
    void = flow.add(Void, name="Void")

    # Connect the nodes/ subgraphs
    flow.connect(load_data.out, sdf_to_isomers.inp)
    flow.connect(sdf_to_isomers.out, abfe_multi_isomer.inp)
    flow.connect(abfe_multi_isomer.out, void.inp)

    # Map the inputs/ parameters
    exclude_parameters = ["commands", "python", "scripts", "modules"]
    flow.map(
        *[
            value
            for parameter, value in abfe_multi_isomer.parameters.items()
            if parameter not in exclude_parameters
        ]
    )
    flow.combine_parameters(load_data.data, name="lig_sdfs_file")

    return flow


abfe_multi_isomer_exposed = expose(get_abfe_multi_isomer_workflow)
