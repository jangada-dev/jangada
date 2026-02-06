###########################
Serialization System
###########################

The Jangada serialization system provides a powerful framework for converting
Python objects to dictionary structures suitable for persistence in HDF5 files,
JSON, or other storage formats.

.. currentmodule:: jangada.serialization


Overview
========

The serialization system consists of three main components:

1. **SerializableProperty** - Descriptor for properties that can be serialized
2. **Serializable** - Base class for objects that can be serialized
3. **SerializableMetatype** - Metaclass that enables automatic registration

Together, these provide automatic serialization with minimal boilerplate.


Quick Start
===========

Define a serializable class::

    from jangada.serialization import Serializable, SerializableProperty

    class Experiment(Serializable):
        name = SerializableProperty(default="")
        temperature = SerializableProperty(default=293.15)
        data = SerializableProperty(default=None)

Create and serialize::

    exp = Experiment(name="Test1", temperature=373.15)
    data = Serializable.serialize(exp)

Deserialize::

    restored = Serializable.deserialize(data)
    print(restored.name)  # 'Test1'


Key Features
============

Automatic Registration
----------------------

Classes are automatically registered when defined - no manual setup required::

    class MyClass(Serializable):
        prop = SerializableProperty(default=0)

    # Automatically registered!
    assert 'mymodule.MyClass' in Serializable

Nested Objects
--------------

Serialization handles nested objects recursively::

    class Inner(Serializable):
        value = SerializableProperty(default=0)

    class Outer(Serializable):
        inner = SerializableProperty(default=None)

    obj = Outer(inner=Inner(value=42))
    data = Serializable.serialize(obj)
    restored = Serializable.deserialize(data)

    assert restored.inner.value == 42

Collections Support
-------------------

Lists, dicts, tuples, and sets work automatically::

    objects = [
        Experiment(name="Exp1"),
        Experiment(name="Exp2"),
        Experiment(name="Exp3")
    ]

    data = Serializable.serialize(objects)
    restored = Serializable.deserialize(data)

Property Features
-----------------

SerializableProperty provides rich functionality:

- **Defaults**: Static or factory-generated default values
- **Parsers**: Validate and transform values
- **Observers**: Track changes with callbacks
- **Post-initializers**: Lazy setup after first access
- **Write-once**: Immutable properties
- **Copiable flag**: Control what gets persisted

See :doc:`serializable_property` for full details.

Type System
-----------

Register primitive types (serialized as-is)::

    from decimal import Decimal
    Serializable.register_primitive_type(Decimal)

Register dataset types (converted to/from arrays)::

    def disassemble(obj):
        return np.array(obj.data), {}

    def assemble(arr, attrs):
        return CustomType(arr)

    Serializable.register_dataset_type(
        CustomType,
        disassemble=disassemble,
        assemble=assemble
    )

Built-in Support
----------------

These types work out of the box:

**Primitives**:
    - ``str``
    - ``int``, ``float``, ``complex`` (via ``numbers.Number``)
    - ``pathlib.Path``

**Dataset Types**:
    - ``numpy.ndarray``
    - ``pandas.Timestamp``
    - ``pandas.DatetimeIndex``


Architecture
============

.. graphviz::

   digraph serialization {
       rankdir=LR;
       node [shape=box];

       Property [label="SerializableProperty\n(Descriptor)"];
       Meta [label="SerializableMetatype\n(Metaclass)"];
       Base [label="Serializable\n(Base Class)"];
       User [label="User Classes"];

       Meta -> Base [label="creates"];
       Meta -> User [label="registers"];
       Base -> User [label="inherited by"];
       Property -> User [label="used in"];
   }

The metaclass automatically:

1. Registers each subclass in a global registry
2. Discovers SerializableProperty descriptors via MRO walk
3. Enables subscript access (``Serializable['module.Class']``)
4. Manages primitive and dataset type registries


API Reference
=============

.. toctree::
   :maxdepth: 2

   serializable
   serializable_property
   serializable_metatype
   type_aliases
   helper_functions


Core Classes
------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Serializable
   SerializableProperty
   SerializableMetatype


Helper Functions
----------------

.. autosummary::
   :toctree: generated/

   get_full_qualified_name
   check_types


Use Cases
=========

Scientific Data Persistence
----------------------------

Store experimental data with metadata::

    class Measurement(Serializable):
        timestamp = SerializableProperty(default=None)
        temperature = SerializableProperty(default=0.0)
        pressure = SerializableProperty(default=0.0)
        readings = SerializableProperty(default=None)

    measurement = Measurement(
        timestamp=pd.Timestamp('2024-01-15 12:30:00'),
        temperature=298.15,
        pressure=101.3,
        readings=np.array([1.2, 3.4, 5.6, 7.8])
    )

    # Save to HDF5 (simplified)
    data = Serializable.serialize(measurement)
    # ... write data to HDF5 ...

Configuration Management
------------------------

Serialize configuration objects::

    class AppConfig(Serializable):
        api_key = SerializableProperty(default="", writeonce=True)
        debug = SerializableProperty(default=False)
        timeout = SerializableProperty(default=30)

    config = AppConfig(api_key="secret123", debug=True)

    # Save config
    data = Serializable.serialize(config)
    # ... save to JSON or YAML ...

    # Load config
    loaded = Serializable.deserialize(data)

System/Subsystem Hierarchies
-----------------------------

Model complex systems with nested objects::

    class Sensor(Serializable):
        sensor_id = SerializableProperty(default="")
        calibration = SerializableProperty(default=None)

    class Subsystem(Serializable):
        name = SerializableProperty(default="")
        sensors = SerializableProperty(default=None)

    class System(Serializable):
        name = SerializableProperty(default="")
        subsystems = SerializableProperty(default=None)

    system = System(
        name="Observatory",
        subsystems=[
            Subsystem(name="Telescope", sensors=[...]),
            Subsystem(name="Spectrometer", sensors=[...])
        ]
    )

Data Pipelines
--------------

Serialize intermediate results::

    class ProcessingStep(Serializable):
        input_data = SerializableProperty(default=None, copiable=True)
        output_data = SerializableProperty(default=None, copiable=True)
        parameters = SerializableProperty(default=None, copiable=True)
        cache = SerializableProperty(default=None, copiable=False)

    step = ProcessingStep(
        input_data=raw_data,
        parameters={'threshold': 0.5}
    )

    # Process...
    step.output_data = process(step.input_data, step.parameters)
    step.cache = expensive_computation()

    # Save (cache is excluded because copiable=False)
    data = Serializable.serialize(step, is_copy=True)


Design Patterns
===============

Factory Pattern with Defaults
------------------------------

Use callable defaults as factories::

    class DataContainer(Serializable):
        data = SerializableProperty()

        @data.default
        def data(self):
            # Factory creates new instance for each object
            return []

    c1 = DataContainer()
    c2 = DataContainer()
    c1.data.append(1)
    # c2.data is still [] (separate list)

Observer Pattern
----------------

Track property changes::

    class Observable(Serializable):
        value = SerializableProperty(default=0)

        @value.add_observer
        def value(self, old, new):
            print(f"Value changed: {old} -> {new}")

    obj = Observable()
    obj.value = 42  # Prints: Value changed: 0 -> 42

Lazy Initialization
-------------------

Defer expensive setup::

    class LazyLoader(Serializable):
        data = SerializableProperty(default=None)

        @data.postinitializer
        def data(self):
            if self.data is None:
                self.data = load_expensive_data()

    loader = LazyLoader()
    # Data not loaded yet...

    _ = loader.data  # First access triggers loading

Validation Pattern
------------------

Ensure data integrity::

    class ValidatedData(Serializable):
        temperature = SerializableProperty(default=0.0)

        @temperature.parser
        def temperature(self, value):
            value = float(value)
            if value < 0:
                raise ValueError("Temperature cannot be negative")
            return value

Immutable Configuration
-----------------------

Prevent accidental changes::

    class Config(Serializable):
        api_endpoint = SerializableProperty(default="", writeonce=True)
        api_key = SerializableProperty(default="", writeonce=True)

    config = Config(
        api_endpoint="https://api.example.com",
        api_key="secret"
    )

    # Cannot change after first set
    # config.api_key = "different"  # Raises AttributeError


Best Practices
==============

Property Naming
---------------

Use descriptive names that indicate purpose::

    # Good
    measurement_timestamp = SerializableProperty()
    calibration_coefficients = SerializableProperty()

    # Avoid
    ts = SerializableProperty()
    data = SerializableProperty()

Default Values
--------------

Provide sensible defaults::

    # Good - clear default behavior
    enabled = SerializableProperty(default=False)
    retry_count = SerializableProperty(default=3)

    # Use factories for mutables
    items = SerializableProperty(default=lambda self: [])

Copiable Flag
-------------

Mark cached/derived data as non-copiable::

    # Data to persist
    input_array = SerializableProperty(default=None, copiable=True)

    # Cached computation (don't persist)
    _cached_fft = SerializableProperty(default=None, copiable=False)

Parsers
-------

Keep parsers simple and focused::

    @property.parser
    def property(self, value):
        # Single responsibility: type conversion
        return float(value)

    # Not this:
    @property.parser
    def property(self, value):
        # Too much: validation + transformation + side effects
        if not valid(value):
            raise ValueError()
        transformed = transform(value)
        self.other_property = side_effect(transformed)
        return transformed

Documentation
-------------

Document non-obvious behavior::

    class MyClass(Serializable):
        # Document units, valid ranges, special values
        temperature = SerializableProperty(default=293.15)  # Kelvin

        # Document when parsers/observers run
        data = SerializableProperty(default=None)  # None triggers lazy load


Performance Considerations
==========================

Memory Usage
------------

- Properties use mangled instance attributes (``_serializable_property__name``)
- Class-level registries are shared, not per-instance
- Large arrays should use dataset types (avoid copies)

Serialization Speed
-------------------

- Recursive serialization is depth-first
- No cycles detection (will hang on circular references)
- Primitive types are fastest (no transformation)

Deserialization Speed
---------------------

- Class lookup is O(1) via dict
- Property setting triggers parsers/observers
- Large object graphs may be slow if many observers

Optimization Tips
-----------------

1. Use copiable=False for non-essential data
2. Avoid expensive observers during deserialization
3. Register custom types as primitives if possible
4. Use dataset types for large arrays


Limitations and Gotchas
=======================

Circular References
-------------------

**Problem**: Circular references cause infinite recursion::

    node = Node()
    node.next = node
    Serializable.serialize(node)  # Hangs!

**Solution**: Restructure to avoid cycles, or track visited objects manually.

Thread Safety
-------------

**Problem**: Global registries not thread-safe.

**Solution**: Register all types at startup (single-threaded) before spawning threads.

Property Initialization Order
------------------------------

**Problem**: Properties set during ``__init__`` may fire observers before object fully initialized.

**Solution**: Use post-initializers for setup that depends on multiple properties.

Unknown Classes
---------------

**Problem**: Deserializing data for unimported classes creates generic types.

**Solution**: Import all Serializable classes before deserialization.

Type Changes
------------

**Problem**: Changing property types between serialization/deserialization may fail.

**Solution**: Use parsers to handle type evolution, or implement version migration.


Troubleshooting
===============

Import Errors
-------------

**Problem**: ``KeyError`` during deserialization - class not found.

**Solution**: Ensure the class's module is imported before deserializing::

    import mymodule  # Imports and registers MyClass
    data = Serializable.deserialize(data)  # Now works

Type Errors
-----------

**Problem**: ``TypeError: No serialisation process implemented for ...``

**Solution**: Register the type::

    Serializable.register_primitive_type(MyCustomType)

Validation Errors
-----------------

**Problem**: Parser raises exception during deserialization.

**Solution**: Update parser to handle old data formats::

    @prop.parser
    def prop(self, value):
        # Handle both old and new formats
        if isinstance(value, OldType):
            value = convert_to_new_type(value)
        return validate(value)


See Also
========

* :doc:`../hdf5/persistence` - Using Serializable with HDF5 storage
* :doc:`../examples/index` - Complete examples and tutorials
* :doc:`../api/index` - Full API reference


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`