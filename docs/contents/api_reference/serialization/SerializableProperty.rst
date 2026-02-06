************************
``SerializableProperty``
************************

.. currentmodule:: jangada.serialization

.. autoclass:: SerializableProperty
    :show-inheritance:


Type Aliases
============

SerializableProperty uses type aliases to provide clear signatures for callback
functions. See :doc:`type_aliases` for detailed documentation.

- :data:`~jangada.serialization.T` - Type variable for property values
- :data:`~jangada.serialization.DefaultCallable` - Default value factory signature
- :data:`~jangada.serialization.Observer` - Change observer signature
- :data:`~jangada.serialization.Parser` - Value parser/validator signature
- :data:`~jangada.serialization.Postinitializer` - Post-initializer signature


Descriptor Protocol
===================

The descriptor protocol methods that define how SerializableProperty integrates
with Python's attribute access mechanism.

.. autosummary::
   :toctree: generated/

   SerializableProperty.__set_name__
   SerializableProperty.__get__
   SerializableProperty.__set__
   SerializableProperty.__delete__


Decorator Methods
=================

Methods for configuring property behavior using decorator syntax. These methods
return new descriptor instances rather than modifying the existing one.

.. autosummary::
   :toctree: generated/

   SerializableProperty.postinitializer
   SerializableProperty.default
   SerializableProperty.parser
   SerializableProperty.add_observer
   SerializableProperty.remove_observer


Read-Only Properties
====================

Attributes that provide information about the descriptor's configuration.

.. autosummary::
   :toctree: generated/

   SerializableProperty.writeonce
   SerializableProperty.copiable