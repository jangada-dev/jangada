**************
``Persistable``
**************

.. currentmodule:: jangada.serialization

.. autoclass:: Persistable
    :show-inheritance:


Construction and Context Manager
================================

.. autosummary::
   :toctree: generated/

   Persistable.__init__
   Persistable.__enter__
   Persistable.__exit__

Persistence API
===============

.. autosummary::
   :toctree: generated/

   Persistable.save
   Persistable.load
   Persistable.save_serialized_data
   Persistable.load_serialized_data

HDF5 Tree Codec
===============

.. autosummary::
   :toctree: generated/

   Persistable._save_data_in_group
   Persistable._load_data_from_h5py_tree

Class Attributes
================

.. autosummary::
   :toctree: generated/

   Persistable.extension

Nested Types
============

.. autosummary::
   :toctree: generated/

   Persistable.ProxyDataset
