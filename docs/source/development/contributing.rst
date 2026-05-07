Contributing
============

Environment setup
-----------------

Adjeff uses `pixi <https://prefix.dev/>`_ for environment management.

.. code-block:: bash

   # CPU dev environment
   pixi run -e dev fmt
   pixi run -e dev lint
   pixi run -e dev type-check
   pixi run -e dev test

Code style
----------

- **Formatter / linter**: ``ruff`` (line length 79, double quotes)
- **Type checker**: ``mypy --strict``
- **Docstrings**: NumPy convention

All checks must pass before opening a pull request.

Commit conventions
------------------

Commits follow the ``<type>: <description>`` format (feat, fix, doc, chore,
refactor, test).
