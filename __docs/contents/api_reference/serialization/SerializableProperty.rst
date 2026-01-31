************************
``SerializableProperty``
************************

.. currentmodule:: jangada.serialization

.. autoclass:: SerializableProperty
    :show-inheritance:


Descriptor Interface
====================

.. autosummary::
   :toctree: generated/

   SerializableProperty.__set_name__
   SerializableProperty.__get__
   SerializableProperty.__set__
   SerializableProperty.__delete__

Property Protocol
=================

.. autosummary::
   :toctree: generated/

   SerializableProperty.getter
   SerializableProperty.setter
   SerializableProperty.deleter
   SerializableProperty.default
   SerializableProperty.parser
   SerializableProperty.observer

Properties
==========

.. autosummary::
   :toctree: generated/

   SerializableProperty.readonly
   SerializableProperty.writeonce
   SerializableProperty.copiable

Decorator
=========

.. autosummary::
   :toctree: generated/

   serializable_property