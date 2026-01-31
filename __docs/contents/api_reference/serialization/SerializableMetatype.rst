**********************
``SerializableMetatype``
**********************

.. currentmodule:: jangada.serialization

.. autoclass:: SerializableMetatype
    :show-inheritance:


Registry Lookup
===============

.. autosummary::
   :toctree: generated/

   SerializableMetatype.__getitem__
   SerializableMetatype.__contains__

Primitive Type Registry
=======================

.. autosummary::
   :toctree: generated/

   SerializableMetatype.register_primitive_type
   SerializableMetatype.remove_primitive_type
   SerializableMetatype.is_primitive_type

Dataset Type Registry
=====================

.. autosummary::
   :toctree: generated/

   SerializableMetatype.register_dataset_type
   SerializableMetatype.remove_dataset_type
   SerializableMetatype.is_dataset_type

Registry Introspection
======================

.. autosummary::
   :toctree: generated/

   SerializableMetatype.serializable_types
   SerializableMetatype.primitive_types
   SerializableMetatype.dataset_types
   SerializableMetatype.serializable_properties
   SerializableMetatype.copiable_properties
