"""
Script to automatically update pyproject.toml with auto-generated
command-line functions in the project.scripts section.
"""

import os
from setuptools import setup, find_packages
import toml

CLI_MODULES = [
    # Steps
    "maize.steps.exs.biosimspace.parameterise",
    "maize.steps.exs.biosimspace.solvate",
    "maize.steps.exs.biosimspace.minimise",
    "maize.steps.exs.biosimspace.equilibrate",
    "maize.steps.exs.biosimspace.production",
    # Graphs
    "maize.graphs.exs.biosimspace.system_preparation",
    "maize.graphs.exs.biosimspace.abfe",
]

def get_cli_functions(cli_modules: list[str]) -> dict[str, str]:
    """Get a dictionary of command-line functions and their entry points."""

    cli_functions = {}
    for module_name_full in cli_modules:
        # Find all functions called "exposed" in the module
        module_name_short = module_name_full.split('.')[-1]
        module = __import__(module_name_full, fromlist=[module_name_short])
        for name in dir(module):
            if name.endswith("_exposed"):
                cli_name = f"bss_{name.replace('_exposed', '')}"
                cli_functions[cli_name] = f"{module_name_full}:{name}"

    return cli_functions


def update_pyproject_toml(cli_modules: list[str]) -> None:
    """Update pyproject.toml with auto-generated command-line functions."""

    pyproject_path = 'pyproject.toml'
    if os.path.exists(pyproject_path):
        pyproject_data = toml.load(pyproject_path)
    else:
        pyproject_data = {}

    # Define the command-line functions and their entry points
    cli_functions = get_cli_functions(cli_modules)

    # Update the pyproject.toml data in the project.scripts section
    pyproject_data.setdefault('project', {}).setdefault('scripts', {}).update(cli_functions)

    # Write the updated data back to pyproject.toml
    with open(pyproject_path, 'w') as toml_file:
        toml.dump(pyproject_data, toml_file)

if __name__ == "__main__":
    update_pyproject_toml(CLI_MODULES)
