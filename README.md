# maize-biosimsace

This is a [*namespace package*](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) for [BioSimSpace](https://biosimspace.openbiosim.org/) nodes and subgraphs for [maize](https://github.com/MolecularAI/maize).

Installation
------------

Clone this repository and run:
```bash
mamba env create -f env-users.yml
mamba activate maize
pip install --no-deps ./
```

Usage
-----

Import the relevant steps from the subpackage e.g.:

```python
from maize.steps.exs.biosimspace import ProductionGromacs
```

Development
-----------
Follow the development guidelines for [maize](https://molecularai.github.io/maize/development.html).


Status
------

In active development and not ready for general use.

Aknowledgements
---------------
This repo is based very closely on [maize-contrib](https://github.com/MolecularAI/maize-contrib) (a separate repo was created to allow the use of the the GPL-3.0 license required by BioSimSpace).