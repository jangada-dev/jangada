****************
``Serializable``
****************

.. currentmodule:: jangada.serialization

.. autoclass:: Serializable
    :show-inheritance:


Construction and Initialization
================================

Methods for creating and initializing Serializable objects.

.. autosummary::
   :toctree: generated/

   Serializable.__init__
   Serializable.copy


Serialization Methods
=====================

Static methods for converting objects to/from dictionary structures.

.. autosummary::
   :toctree: generated/

   Serializable.serialize
   Serializable.deserialize


Comparison and Copying
======================

Special methods for object comparison and copying protocol support.

.. autosummary::
   :toctree: generated/

   Serializable.__eq__
   Serializable.__copy__


Type Registration
=================

Class methods for managing the type registry system.

.. autosummary::
   :toctree: generated/

   Serializable.register_primitive_type
   Serializable.remove_primitive_type
   Serializable.is_primitive_type
   Serializable.register_dataset_type
   Serializable.remove_dataset_type
   Serializable.is_dataset_type


Read-Only Properties
====================

Properties that provide information about registered types and properties.

.. autosummary::
   :toctree: generated/

   Serializable.serializable_types
   Serializable.primitive_types
   Serializable.dataset_types
   Serializable.serializable_properties
   Serializable.copiable_properties


Examples
========

Basic Usage
-----------

Define a simple Serializable class::

    from jangada.serialization import Serializable, SerializableProperty

    class Experiment(Serializable):
        name = SerializableProperty(default="")
        temperature = SerializableProperty(default=293.15)
        data = SerializableProperty(default=None)

Create and serialize an object::

    exp = Experiment(name="Test1", temperature=373.15)
    data = Serializable.serialize(exp)

    # data is now a dictionary:
    # {
    #     '__class__': 'mymodule.Experiment',
    #     'name': 'Test1',
    #     'temperature': 373.15,
    #     'data': None
    # }

Deserialize back to an object::

    restored = Serializable.deserialize(data)
    print(restored.name)  # 'Test1'
    print(restored.temperature)  # 373.15


Nested Objects
--------------

Serialization handles nested Serializable objects automatically::

    class Trial(Serializable):
        trial_num = SerializableProperty(default=0)
        experiment = SerializableProperty(default=None)

    trial = Trial(trial_num=1, experiment=exp)
    data = Serializable.serialize(trial)
    restored = Serializable.deserialize(data)

    # Nested object is preserved
    print(restored.experiment.name)  # 'Test1'


Collections
-----------

Lists, dicts, tuples, and sets are handled recursively::

    experiments = [
        Experiment(name="Exp1", temperature=300.0),
        Experiment(name="Exp2", temperature=350.0),
        Experiment(name="Exp3", temperature=400.0)
    ]

    data = Serializable.serialize(experiments)
    restored = Serializable.deserialize(data)

    print(len(restored))  # 3
    print(restored[0].name)  # 'Exp1'


Copying Objects
---------------

Create independent copies with the copy() method::

    original = Experiment(name="Original", temperature=300.0)
    copied = original.copy()

    copied.name = "Modified"
    print(original.name)  # 'Original' (unchanged)
    print(copied.name)    # 'Modified'


Selective Copying
-----------------

Use the copiable flag to control what gets copied::

    class DataProcessor(Serializable):
        # This will be copied/serialized
        input_data = SerializableProperty(default=None, copiable=True)

        # This won't be copied (cache, temporary state)
        cached_result = SerializableProperty(default=None, copiable=False)

    proc = DataProcessor(input_data=[1, 2, 3], cached_result=[4, 5, 6])

    # Copy only includes copiable properties
    copied = proc.copy()
    print(copied.input_data)      # [1, 2, 3]
    print(copied.cached_result)   # None (default)


Custom Primitive Types
----------------------

Register custom types that serialize as-is::

    from decimal import Decimal

    Serializable.register_primitive_type(Decimal)

    class FinancialData(Serializable):
        amount = SerializableProperty(default=Decimal('0.00'))

    data = FinancialData(amount=Decimal('123.45'))
    serialized = Serializable.serialize(data)
    # Decimal values pass through unchanged


Custom Dataset Types
--------------------

Register types that need conversion to/from arrays::

    import numpy as np

    class CustomMatrix:
        def __init__(self, data):
            self.data = np.array(data)

    def disassemble(matrix):
        """Convert to array and metadata."""
        return matrix.data, {'shape': matrix.data.shape}

    def assemble(arr, attrs):
        """Reconstruct from array and metadata."""
        return CustomMatrix(arr)

    Serializable.register_dataset_type(
        CustomMatrix,
        disassemble=disassemble,
        assemble=assemble
    )


NumPy and Pandas Integration
-----------------------------

NumPy arrays and Pandas timestamps work automatically::

    import numpy as np
    import pandas as pd

    class Measurement(Serializable):
        timestamp = SerializableProperty(default=None)
        values = SerializableProperty(default=None)

    measurement = Measurement(
        timestamp=pd.Timestamp('2024-01-15 12:30:00', tz='UTC'),
        values=np.array([1.2, 3.4, 5.6])
    )

    data = Serializable.serialize(measurement)
    restored = Serializable.deserialize(data)

    # Timezone and array are preserved
    print(restored.timestamp.tz)  # <UTC>
    print(restored.values.shape)   # (3,)


Equality Comparison
-------------------

Objects compare based on their copiable properties::

    exp1 = Experiment(name="Test", temperature=300.0)
    exp2 = Experiment(name="Test", temperature=300.0)
    exp3 = Experiment(name="Other", temperature=300.0)

    print(exp1 == exp2)  # True (same values)
    print(exp1 == exp3)  # False (different name)


See Also
========

* :doc:`serializable_property` - Property descriptor for serializable attributes
* :doc:`serializable_metatype` - Metaclass documentation
* :doc:`type_aliases` - Type aliases used in the API
* :doc:`hdf5_persistence` - Using Serializable with HDF5 storage (if available)