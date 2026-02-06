************************
``SerializableMetatype``
************************

.. currentmodule:: jangada.serialization

.. autoclass:: SerializableMetatype
    :show-inheritance:


Metaclass Protocol
==================

Special methods that implement the metaclass protocol.

.. autosummary::
   :toctree: generated/

   SerializableMetatype.__new__
   SerializableMetatype.__getitem__
   SerializableMetatype.__contains__


Primitive Type Management
=========================

Methods for registering and managing primitive types that serialize as-is.

.. autosummary::
   :toctree: generated/

   SerializableMetatype.register_primitive_type
   SerializableMetatype.remove_primitive_type
   SerializableMetatype.is_primitive_type


Dataset Type Management
=======================

Methods for registering and managing types that require array conversion.

.. autosummary::
   :toctree: generated/

   SerializableMetatype.register_dataset_type
   SerializableMetatype.remove_dataset_type
   SerializableMetatype.is_dataset_type


Read-Only Properties
====================

Properties providing access to type and class registries.

.. autosummary::
   :toctree: generated/

   SerializableMetatype.serializable_types
   SerializableMetatype.primitive_types
   SerializableMetatype.dataset_types
   SerializableMetatype.serializable_properties
   SerializableMetatype.copiable_properties


Overview
========

``SerializableMetatype`` is the metaclass that powers the ``Serializable`` system.
It provides automatic registration, property discovery, and type management without
requiring manual configuration.

Key Features
------------

**Automatic Registration**
    Every class that inherits from ``Serializable`` is automatically registered
    in a global registry, indexed by its fully qualified name.

**Property Discovery**
    The metaclass walks the Method Resolution Order (MRO) to discover all
    ``SerializableProperty`` descriptors, including those inherited from base classes.

**Type Registries**
    Maintains separate registries for primitive types (serialized as-is) and
    dataset types (require conversion to/from arrays).

**Subscript Access**
    Classes can be retrieved by their qualified name using subscript notation:
    ``Serializable['module.ClassName']``

**Membership Testing**
    Check if a class is registered using the ``in`` operator:
    ``'module.ClassName' in Serializable``


How It Works
============

Class Creation
--------------

When a new ``Serializable`` subclass is defined, the metaclass:

1. Calls ``__new__`` to create the class object
2. Generates the fully qualified name
3. Registers the class in ``Serializable._subclasses``
4. Walks the MRO to collect ``SerializableProperty`` descriptors
5. Stores them in ``_serializable_properties``

Example::

    # When this class is defined...
    class MyClass(Serializable):
        prop1 = SerializableProperty(default=0)
        prop2 = SerializableProperty(default="")

    # ...the metaclass automatically:
    # 1. Registers it: Serializable._subclasses['__main__.MyClass'] = MyClass
    # 2. Collects properties: MyClass._serializable_properties = {'prop1': ..., 'prop2': ...}


Property Collection
-------------------

The metaclass collects properties from all base classes in the MRO::

    class Base(Serializable):
        base_prop = SerializableProperty(default=0)

    class Derived(Base):
        derived_prop = SerializableProperty(default=1)

    # Derived._serializable_properties includes both:
    # {'base_prop': ..., 'derived_prop': ...}


Class Registry
--------------

Access registered classes by name::

    # Define a class
    class MyClass(Serializable):
        pass

    # Access by qualified name
    qualname = 'mymodule.MyClass'
    retrieved_class = Serializable[qualname]

    # Check registration
    if qualname in Serializable:
        print("Class is registered")


Type System
===========

Primitive Types
---------------

Primitive types are serialized without transformation. By default, these types
are registered::

    - str
    - numbers.Number (int, float, complex, Decimal, etc.)
    - pathlib.Path

Register additional primitive types::

    from decimal import Decimal
    Serializable.register_primitive_type(Decimal)

    # Now Decimal values serialize as-is
    class FinancialData(Serializable):
        amount = SerializableProperty(default=Decimal('0.00'))


Dataset Types
-------------

Dataset types require conversion to/from NumPy arrays. These are typically used
for types that will be stored as HDF5 datasets rather than attributes.

Built-in dataset types::

    - numpy.ndarray
    - pandas.Timestamp
    - pandas.DatetimeIndex

Register custom dataset types::

    import numpy as np

    class CustomArray:
        def __init__(self, data):
            self.data = np.array(data)

    def disassemble(obj):
        """Convert object to (array, attributes) tuple."""
        return obj.data, {'dtype': str(obj.data.dtype)}

    def assemble(array, attrs):
        """Reconstruct object from array and attributes."""
        result = CustomArray(array)
        # Could use attrs to restore additional state
        return result

    Serializable.register_dataset_type(
        CustomArray,
        disassemble=disassemble,
        assemble=assemble
    )

The disassemble/assemble pattern allows complex types to be stored efficiently
in array-based storage systems like HDF5.


Properties Access
=================

All Properties
--------------

Get all SerializableProperty descriptors on a class::

    class MyClass(Serializable):
        prop1 = SerializableProperty(default=0)
        prop2 = SerializableProperty(default="")
        regular_attr = 42  # Not a SerializableProperty

    props = MyClass.serializable_properties
    # Returns: {'prop1': <SerializableProperty>, 'prop2': <SerializableProperty>}

This includes properties from base classes.


Copiable Properties
-------------------

Get only properties marked as copiable::

    class MyClass(Serializable):
        data = SerializableProperty(default=None, copiable=True)
        cache = SerializableProperty(default=None, copiable=False)

    copiable = MyClass.copiable_properties
    # Returns: {'data': <SerializableProperty>}

This is used internally by ``serialize(is_copy=True)`` and ``copy()``.


Advanced Usage
==============

Multiple Inheritance
--------------------

Properties from all base classes are collected::

    class Mixin1(Serializable):
        prop1 = SerializableProperty(default=0)

    class Mixin2(Serializable):
        prop2 = SerializableProperty(default=0)

    class Combined(Mixin1, Mixin2):
        prop3 = SerializableProperty(default=0)

    # Combined has all three properties
    assert 'prop1' in Combined.serializable_properties
    assert 'prop2' in Combined.serializable_properties
    assert 'prop3' in Combined.serializable_properties


Dynamic Class Creation
----------------------

The metaclass works with dynamically created classes::

    # Create a class at runtime
    DynamicClass = SerializableMetatype(
        'DynamicClass',
        (Serializable,),
        {
            'value': SerializableProperty(default=0)
        }
    )

    # It's automatically registered
    qualname = get_full_qualified_name(DynamicClass)
    assert qualname in Serializable


Introspection
-------------

Query the type registries::

    # Get all registered Serializable classes
    all_classes = Serializable.serializable_types

    # Get all primitive types
    primitives = Serializable.primitive_types
    assert str in primitives

    # Get all dataset types
    datasets = Serializable.dataset_types
    assert np.ndarray in datasets

    # Check specific types
    assert Serializable.is_primitive_type(str)
    assert Serializable.is_dataset_type(np.ndarray)


Design Rationale
================

Why a Metaclass?
----------------

The metaclass approach provides several benefits:

1. **Automatic registration**: No manual registration code needed
2. **Zero boilerplate**: Properties are discovered automatically
3. **Inheritance support**: Base class properties are inherited correctly
4. **Type safety**: All subclasses share the same serialization machinery
5. **Introspection**: Easy to query what classes and types are available

Alternative approaches (decorators, manual registration) would require more
code and be more error-prone.


Why Global Registries?
-----------------------

Global registries enable:

1. **Deserialization**: Reconstruct objects from just a class name string
2. **Cross-module serialization**: Serialize objects from any module
3. **Plugin architectures**: Dynamically loaded modules auto-register
4. **Introspection**: Query all available Serializable types

The registries are class-level (not instance-level) to minimize memory overhead.


Thread Safety Considerations
-----------------------------

The current implementation does not guarantee thread safety for:

- Concurrent class definition (rare in practice)
- Concurrent type registration/removal

For most applications, this is not an issue since:

- Classes are typically defined at module load time (single-threaded)
- Type registration happens during initialization (before threads spawn)

If you need thread safety, add locking around type registration operations.


See Also
========

* :doc:`serializable` - The base class using this metaclass
* :doc:`serializable_property` - Property descriptor documentation
* :doc:`helper_functions` - Helper functions like get_full_qualified_name