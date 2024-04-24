"""BioSimSpace absolute binding free energy subgraphs and workflows."""

from abc import ABC
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Annotated, Callable, Literal

from maize.core.graph import Graph
from maize.core.interface import FileParameter, Flag, Input, Output, Parameter, Suffix
from maize.core.workflow import Workflow, expose
from maize.steps.exs.biosimspace import (
    AFEResult,
    AFESomd,
    CollectAFEResults,
    Combine,
    EquilibrateGromacs,
    GenerateBoreschRestraintGromacs,
    LegType,
    MinimiseGromacs,
    Parameterise,
    ProductionGromacs,
    SaveAFEResults,
    Solvate,
    StageType,
)
from maize.steps.exs.biosimspace._utils import IsomerToSDF
from maize.steps.io import LoadData, Return
from maize.steps.plumbing import Accumulate, Copy, Scatter
from maize.utilities.chem.chem import Isomer
from maize.utilities.macros import parallel

from .system_preparation import SystemPreparationBound, SystemPreparationFree

__all__ = [
    "AbsoluteBindingFreeEnergySingleRepeat",
    "AbsoluteBindingFreeEnergyMultiRepeat",
    "abfe_no_prep_workflow",
    "abfe_with_prep_workflow",
]


class AbsoluteBindingFreeEnergySingleRepeat(Graph):
    """
    A class for running a single absolute binding free energy calculation
    using SOMD through BioSimSpace. This requires parameterised and equilibrated
    input structures, and performs short production simulations followed by
    ABFE calculations.
    """

    # Class variables
    required_stages = {
        LegType.BOUND: [StageType.RESTRAIN, StageType.DISCHARGE, StageType.VANISH],
        LegType.FREE: [StageType.DISCHARGE, StageType.VANISH],
    }

    # inp_free: Input[list[Path]] = Input(optional=True)
    # """
    # Paths to equilibrated input files for the free leg. A
    # topology and a coordinate file are required. These can
    # be in any of the formats given by BSS.IO.fileFormats()
    # e.g.:

    # gro87, grotop, prm7, rst rst7
    # """

    inp_bound: Input[list[Path]] = Input(optional=True)
    """
    Paths to equilibrated input files for the bound leg. A
    topology and a coordinate file are required. These can
    be in any of the formats given by BSS.IO.fileFormats()
    e.g.:
    
    gro87, grotop, prm7, rst rst7
    """

    # Parameters
    lam_vals: Parameter[dict[LegType, [StageType, list[float]]]] = Parameter(
        default={
            LegType.BOUND: {
                StageType.RESTRAIN: [0.0, 1.0],
                StageType.DISCHARGE: [0.0, 0.291, 0.54, 0.776, 1.0],
                StageType.VANISH: [
                    0.0,
                    0.026,
                    0.054,
                    0.083,
                    0.111,
                    0.14,
                    0.173,
                    0.208,
                    0.247,
                    0.286,
                    0.329,
                    0.373,
                    0.417,
                    0.467,
                    0.514,
                    0.564,
                    0.623,
                    0.696,
                    0.833,
                    1.0,
                ],
            },
            LegType.FREE: {
                StageType.DISCHARGE: [0.0, 0.222, 0.447, 0.713, 1.0],
                StageType.VANISH: [
                    0.0,
                    0.026,
                    0.055,
                    0.09,
                    0.126,
                    0.164,
                    0.202,
                    0.239,
                    0.276,
                    0.314,
                    0.354,
                    0.396,
                    0.437,
                    0.478,
                    0.518,
                    0.559,
                    0.606,
                    0.668,
                    0.762,
                    1.0,
                ],
            },
        }
    )
    """A dictionary of lambda values for each stage of the calculation."""

    # Output
    out: Output[AFEResult] = Output()
    """The predicted free energy change."""

    def build(self) -> None:
        # Create a node to collect all the results
        collect_results = self.add(CollectAFEResults)
        afe_nodes = []

        for leg in LegType:
            # The leg multiplier is used to determine the direction of the perturbation
            leg_multiplier = -1 if leg == LegType.BOUND else 1

            # # Set Up
            # sys_prep = self.add(
            #     SystemPreparationBound if leg == LegType.BOUND else SystemPreparationFree,
            #     name=f"SystemPreparation{leg.leg_name}",
            # )

            # for param in sys_prep.parameters.values():
            #     self.combine_parameters(param, name=f"{leg.leg_name}_{param.name}")

            # Derive restraints or run some short production simulations to generate different configurations
            # for each repeat run
            node = GenerateBoreschRestraintGromacs if leg == LegType.BOUND else ProductionGromacs
            pre_production = self.add(node, name=f"{node.__name__}{leg.class_name}")

            # Map the inputs
            # TODO: Tidy this up and fix mapping to general parameters
            if leg == LegType.BOUND:
                name = "generate_boresch_restraint"
                param_names_to_map_local = ["force_constant", "runtime", "timestep"]
                param_names_to_map_general = []
            else:
                name = "pre_production"
                param_names_to_map_local = [
                    "runtime",
                    "timestep",
                ]
                param_names_to_map_general = [
                    "temperature",
                    "pressure",
                    "report_interval",
                    "restart_interval",
                    "thermostat_time_constant",
                ]
            for param_name in param_names_to_map_local:
                self.combine_parameters(
                    pre_production.parameters[param_name], name=f"{param_name}_{name}"
                )
            # for param_name in param_names_to_map_general:
            #     self.combine_parameters(pre_production.parameters[param_name], name=param_name)

            # Set up the stages required for each leg
            leg_stage_nodes = []
            for stage in self.required_stages[leg]:
                afe = self.add(
                    AFESomd,
                    name=f"AbsoluteBindingFreeEnergy{stage.class_name}{leg.class_name}",
                )
                leg_stage_nodes.append(afe)
                afe_nodes.append(afe)

                # Set options
                afe.perturbation_type.set(stage.perturbation_type)
                afe.dg_sign.set(leg_multiplier)
                afe.lam_vals.set(self.lam_vals.value[leg][stage])
                # afe.results_file_name.set(f"results_{leg.leg_name}_{stage.stage_name}")
                # If this is the bound vanish leg, add in the restraint correction
                if leg == LegType.BOUND and stage == StageType.VANISH:
                    afe.apply_restraint_correction.set(True)

            # Connect the nodes within the leg
            for node in leg_stage_nodes:
                self.connect(pre_production.out, node.inp)
                self.connect(node.out, collect_results.inp)
                # Make sure that we pass the restraints if this is the bound leg
                if leg == LegType.BOUND:
                    self.connect(pre_production.boresch_restraint, node.boresch_restraint)

            # Map the per-leg inputs
            # name = "inp_bound" if leg == LegType.BOUND else "inp_free"
            # self.combine_parameters(pre_production.inp, name=name)
            # Map ports
            if leg == LegType.BOUND:
                self.inp_bound = self.map_port(pre_production.inp, name="inp_bound")
            else:
                self.inp_free = self.map_port(pre_production.inp, name="inp_free")

        # Map the desired options for the alchemical nodes
        # WATCH TAU_T and THERMOSTAT TIME CONSTANT
        param_names_to_map = [
            "estimator",
            "pressure",
            "temperature",
            "report_interval",
            "restart_interval",
            "runtime",
            "tau_t",
            "thermostat_time_constant",
            "timestep",
        ]
        for param_name in param_names_to_map:
            self.combine_parameters(
                *[node.parameters[param_name] for node in afe_nodes], name=param_name
            )

        # Map the overall inputs
        self.out = self.map_port(collect_results.out)


class AbsoluteBindingFreeEnergyMultiRepeat(Graph):
    """
    A class for running multiple repeat absolute binding free energy calculations
    using SOMD through BioSimSpace. This requires unparameterised
    input structures and performs short production simulations followed by
    ABFE calculations.
    """

    n_repeats: Parameter[int] = Parameter(default=2)
    """The number of repeat calculations to perform."""

    def build(self) -> None:
        # We need a node to accumulate the results
        accumulate_results = self.add(
            Accumulate[AFEResult],
            name="AccumulateAFEResults",
            parameters={"n_packets": self.n_repeats.value},
        )

        # Add all the repeat ABFE nodes
        abfe_subgraph = self.add(
            parallel(
                AbsoluteBindingFreeEnergySingleRepeat,
                n_branches=self.n_repeats.value,
                inputs=[],
                constant_inputs=["inp_bound", "inp_free"],
                outputs=["out"],
            )
        )
        # Connect the outputs to the accumulator
        self.connect(abfe_subgraph.out, accumulate_results.inp)

        # Map the parameters from the abfe subgraph
        self.map(*abfe_subgraph.parameters.values())

        # Map all the inputs to graph inputs
        self.inp_free = self.map_port(abfe_subgraph.inp_free, name="inp_free")
        self.inp_bound = self.map_port(abfe_subgraph.inp_bound, name="inp_bound")
        self.out = self.map_port(accumulate_results.out, name="out")


class AbsoluteBindingFreeEnergyMultiWithPrep(Graph):
    """
    A class for running multiple repeat absolute binding free energy calculations
    from unparameterised input structures.
    """

    def build(self) -> None:

        # We need to copy input to the bound and free legs
        copy_inp = self.add(Copy[Path], name="CopyInput")

        # Run system preparation for each leg
        sys_prep_free = self.add(SystemPreparationFree, name="SystemPreparationFree")
        sys_prep_bound = self.add(SystemPreparationBound, name="SystemPreparationBound")

        # Run repeats of ABFE calculations
        abfe_calc = self.add(AbsoluteBindingFreeEnergyMultiRepeat, name="AbsoluteBindingFreeEnergy")

        # Save the ABFE results
        save_results = self.add(SaveAFEResults, name="SaveAFEResults")

        # Connect the nodes
        self.connect(copy_inp.out, sys_prep_free.inp)
        self.connect(copy_inp.out, sys_prep_bound.inp)
        self.connect(sys_prep_free.out, abfe_calc.inp_free)
        self.connect(sys_prep_bound.out, abfe_calc.inp_bound)
        self.connect(abfe_calc.out, save_results.inp)

        # Map inputs/ parameters for prep stages
        self.combine_parameters(sys_prep_bound.protein_pdb, name="protein_pdb_path")
        # Get a dict of all parameters for each system preparation stage where parameters with the same name
        # Share the same key in the dict
        params_prep = defaultdict(list)
        params_prep = {param.name: [param] for param in sys_prep_bound.parameters.values()}
        # Now add the free params, appending to the list if the key already exists
        for param in sys_prep_free.parameters.values():
            params_prep[param.name].append(param)

        for param_name, param_vals in params_prep.items():
            if param_name in ["protein_pdb"]:
                continue
            prefix = "prep_" if "force_field" not in param_name else ""
            self.combine_parameters(*param_vals, name=f"{prefix}{param_name}")

        # Map inputs/ parameters for the abfe stage
        for param in abfe_calc.parameters.values():
            self.combine_parameters(param, name=f"abfe_{param.name}")

        self.combine_parameters(save_results.file, name="results_file_name")

        # self.combine_parameters(copy_inp.inp, name="lig_sdf_path")
        self.inp = self.map_port(copy_inp.inp, name="inp")


class AbsoluteBindingFreeEnergyMultiIsomer(Graph):
    """
    A subgraph for running multiple repeat absolute binding free energy calculations
    on a set of isomers.
    """

    inp: Input[list[Isomer]]
    """A list of isomer collections to run ABFE calculations on."""

    # TODO: Actually add the scores
    # out: Output[list[Isomer]]
    # """Scored molecule output."""

    def build(self) -> None:

        # Decompose the input list of isomers into individual isomers
        scatter = self.add(Scatter[Isomer], name="ScatterIsomers")

        # Convert the input isomer paths to sdfs
        iso_to_sdf_paths = self.add(IsomerToSDF, name="IsomerToSDF", loop=False)

        # TODO: Figure out why adding loop=True causes nodes to run with no
        # input, even if
        # Run the ABFE calculations on each isomer
        abfe_calc = self.add(
            AbsoluteBindingFreeEnergyMultiWithPrep,
            name="AbsoluteBindingFreeEnergy",
            loop=False,
        )

        # Return the scored isomers
        # At the moment, we just return the input isomers
        # accumulate = self.add(Accumulate[Isomer], name="AccumulateIsomers")

        # Join things up
        self.connect(scatter.out, iso_to_sdf_paths.inp)
        self.connect(iso_to_sdf_paths.out, abfe_calc.inp)

        # Map the parameters
        self.map(*abfe_calc.parameters.values())

        # Map inputs and outputs
        self.inp = self.map_port(scatter.inp, name="inp")


def get_abfe_multi_isomer_workflow() -> Workflow:
    """
    A workflow to perform absolute binding free energy calculations on a set
    of isomers.
    """

    flow = Workflow(name="absolute_binding_free_energy_multi_isomer")

    load_data = flow.add(LoadData[list[Isomer]], name="LoadData")

    abfe_multi_isomer = flow.add(
        AbsoluteBindingFreeEnergyMultiIsomer, name="AbsoluteBindingFreeEnergy"
    )

    # Connect the nodes/ subgraphs
    flow.connect(load_data.out, abfe_multi_isomer.inp)

    # Map the inputs/ parameters
    flow.map(*abfe_multi_isomer.parameters.values())
    flow.combine_parameters(load_data.data, name="lig_sdfs_file")

    return flow


abfe_multi_isomer_exposed = expose(get_abfe_multi_isomer_workflow)


def get_abfe_no_prep_workflow() -> Workflow:
    """
    A workflow to perform absolute binding free energy calculations given
    parameterised and equilibrated input systems.
    """

    flow = Workflow(name="absolute_binding_free_energy_no_prep")

    abfe_calc = flow.add(AbsoluteBindingFreeEnergyMultiRepeat, name="AbsoluteBindingFreeEnergy")
    save_results = flow.add(SaveAFEResults, name="SaveAFEResults")

    # Connect the nodes/ subgraphs
    flow.connect(abfe_calc.out, save_results.inp)

    # Map the inputs/ parameters
    flow.combine_parameters(abfe_calc.inp_bound, name="inp_bound")
    flow.combine_parameters(abfe_calc.inp_free, name="inp_free")
    flow.map(*abfe_calc.parameters.values())
    # TODO: Figure out why supplying a default value for the save_results.file parameter
    # doesn't work (still needs to be supplied for CLI if no default is given in SaveAFEResults)
    flow.combine_parameters(save_results.file, name="results_file_name")

    return flow


abfe_no_prep_exposed = expose(get_abfe_no_prep_workflow)


def get_abfe_with_prep_workflow() -> Workflow:
    """
    A workflow which takes prepared but unparameterised input structures and
    runs 1) system preparation, 2) ABFE calculations.
    """
    flow = Workflow(name="absolute_binding_free_energy")

    # Run setup and repeats of ABFE calculations
    abfe_with_prep = flow.add(
        AbsoluteBindingFreeEnergyMultiWithPrep, name="AbsoluteBindingFreeEnergy"
    )

    # Map the inputs/ parameters
    flow.map(*abfe_with_prep.parameters.values())

    return flow


abfe_with_prep_exposed = expose(get_abfe_with_prep_workflow)
# abfe_with_prep_exposed = get_abfe_with_prep_workflow
