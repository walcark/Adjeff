Installation
============

Adjeff requires Python 3.11–3.12 and is managed via `pixi`_.

.. _pixi: https://prefix.dev/

CPU environment
---------------

.. code-block:: bash

   pixi run -e dev test

GPU environment (CUDA 12.6)
----------------------------

.. code-block:: bash

   pixi run -e dev-gpu test

Dependencies
------------

Core dependencies (managed automatically by pixi):

- `PyTorch <https://pytorch.org/>`_ ≥ 2.0
- `xarray <https://docs.xarray.dev/>`_ ≥ 2026.1
- `Smart-G <https://github.com/hygeos/smartg>`_ ≥ 1.1.3 (conda-forge only)
- `zarr <https://zarr.readthedocs.io/>`_ ≥ 3.0
- `pydantic <https://docs.pydantic.dev/>`_ ≥ 2.12
