Documentation Guidelines
========================

Documenting a module/package
----------------------------
* To document a new module document each function, class, attribute usw. use Numpy Style Docstrings.
* Collect all Objects and functions you want to show up in the documentation of a module in eiter a ``__init__.py`` file
  (see e.g. :mod:`kron` and ``/source/kron/__init__.py``) or at the beginning of the document (see e.g. :mod:`kron.utils` and ``/source/kron/utils.py``).
* Add a ``.rst`` file in the ``/docs/source/modules/`` directory with the following content:

  | ``.. automodule:: <your module>``
  |       ``:no-members:``
  |       ``:no-inherited-members:``
  |       ``:no-special-members:``
    
* Add a eiter a new section if you'r adding a new package or just a new item if you'r adding a new module to a existing packege
  in the ``/docs/source/index.rst`` file.

Documenting private methods
----------------------------
If you want to have private methods (staring with an underscore) shown up in the documentation you can add
your method to the exceptions list in the ``/docs/source/_templates/autosummary/class.rst``.
Note that this will effect all classes in this documentation, but only if the class actually has a method of the given name.
Adding the method ``__call__`` to the exeption list will show the ``__call__`` method in the docs for all classes that implement 
``__call__``.

Building the docs
------------------
See ``/docs/README.rst``.