.. _api-mixin:

##########################
Mixins (``jangada.mixin``)
##########################

.. currentmodule:: jangada.mixin

The mixin module provides orthogonal, composable capabilities for domain objects
through lightweight mixin classes. Each mixin adds exactly one conceptual feature
and can be freely combined with others.


Overview
========

Six mixin classes are provided:

.. autosummary::
   :toctree: generated/

   Identifiable
   Taggable
   Nameable
   Describable
   Colorable
   Activatable


Quick Start
===========

Basic usage::

    from jangada.mixin import Identifiable, Nameable, Colorable

    class Component(Identifiable, Nameable, Colorable):
        pass

    comp = Component()
    comp.name = "Temperature Sensor"
    comp.color = 'red'

    print(comp.id)    # Auto-generated UUID
    print(comp.name)  # 'Temperature Sensor'
    print(comp.color) # '#FF0000'


Composition Patterns
====================

Combining Mixins
----------------

Mixins are designed to be freely composed. Order doesn't matter::

    class Asset(Identifiable, Taggable, Nameable,
                Describable, Colorable, Activatable):
        pass

    asset = Asset()
    asset.tag = "asset_001"
    asset.name = "Primary Asset"
    asset.description = "Main processing unit"
    asset.color = '#00FF00'
    asset.active = True

Common Combinations
-------------------

**Component with Identity and Display**::

    class Component(Identifiable, Nameable, Colorable):
        pass

**Namespace Member**::

    class Subsystem(Taggable, Nameable, Describable):
        pass

**UI Element**::

    class Widget(Nameable, Colorable, Activatable):
        pass

**Full-Featured Object**::

    class System(Identifiable, Taggable, Nameable,
                 Describable, Colorable, Activatable):
        pass


Design Patterns
===============

Namespace Access
----------------

Use ``Taggable`` for attribute-style access::

    class System:
        def __init__(self):
            self._components = {}

        def add(self, component: Taggable):
            if component.tag:
                self._components[component.tag] = component

        def __getattr__(self, tag: str):
            return self._components.get(tag)

    system = System()

    sensor = Component()
    sensor.tag = "temp_sensor"
    system.add(sensor)

    # Access by tag
    assert system.temp_sensor is sensor

ID-Based Lookup
---------------

Use ``Identifiable`` for registry patterns::

    # Create object
    comp = Component()
    comp_id = comp.id

    # Store ID (in database, config file, etc.)
    config = {'component_id': comp_id}

    # Later, retrieve by ID
    retrieved = Identifiable.get_instance(comp_id)
    assert retrieved is comp

Display vs Programmatic
-----------------------

Combine ``Taggable`` and ``Nameable`` for dual-purpose identification::

    comp = Component()
    comp.tag = "temp_sensor_01"  # For code
    comp.name = "Temperature Sensor #1"  # For users

    # Programmatic access
    system.temp_sensor_01  # Uses tag

    # User display
    print(f"Component: {comp.name}")  # Shows name

Color-Coded Categories
----------------------

Use ``Colorable`` for visual organization::

    sensors = [Component() for _ in range(5)]
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for sensor, color in zip(sensors, colors):
        sensor.color = color

    # Plot with consistent colors
    for sensor in sensors:
        plt.plot(data, color=sensor.color, label=sensor.name)

Filtering by State
------------------

Use ``Activatable`` for conditional processing::

    components = [Component() for _ in range(10)]

    # Deactivate some
    components[3].active = False
    components[7].active = False

    # Process only active
    active = [c for c in components if c.active]
    for comp in active:
        process(comp)


Examples
========

Scientific Instrument
---------------------

Model a scientific instrument with full metadata::

    class Instrument(Identifiable, Taggable, Nameable,
                     Describable, Colorable, Activatable):
        pass

    spectrometer = Instrument()
    spectrometer.tag = "spec_main"
    spectrometer.name = "UV-Vis Spectrometer"
    spectrometer.description = "Main spectroscopy unit for absorption measurements"
    spectrometer.color = '#1F77B4'  # Blue for optical
    spectrometer.active = True

    # Retrieve by ID later
    spec_id = spectrometer.id
    # ... store spec_id in database ...

    # Retrieve
    spec = Identifiable.get_instance(spec_id)

Data Pipeline Component
-----------------------

Processing stage with enable/disable::

    class Stage(Nameable, Colorable, Activatable):
        def process(self, data):
            if not self.active:
                return data  # Skip if inactive
            return self._do_process(data)

    pipeline = [
        Stage(name="Normalize", color='green', active=True),
        Stage(name="Filter", color='blue', active=True),
        Stage(name="Smooth", color='orange', active=False),  # Disabled
    ]

    for stage in pipeline:
        if stage.active:
            data = stage.process(data)

Configuration System
--------------------

Hierarchical configuration with namespace access::

    class Config(Taggable, Nameable, Describable):
        pass

    database = Config()
    database.tag = "database"
    database.name = "Database Settings"
    database.description = "PostgreSQL connection parameters"

    api = Config()
    api.tag = "api"
    api.name = "API Settings"
    api.description = "REST API configuration"


Best Practices
==============

Identity
--------

**Use Identifiable when:**
    - Objects need global uniqueness
    - You need to reference objects by stable ID
    - Objects should be retrievable from IDs
    - Hash-based collections are needed (sets, dict keys)

**Don't use Identifiable for:**
    - Value objects (use normal equality)
    - Temporary objects
    - Objects that should be compared by content

Tags
----

**Use Taggable when:**
    - Objects will be accessed via attribute syntax
    - Programmatic identifiers are needed
    - Namespace-style organization is desired

**Tag naming conventions:**
    - Use snake_case: ``temp_sensor_01``
    - Be descriptive but concise
    - Avoid abbreviations unless standard
    - Include type/category info: ``sensor_temp``, ``ctrl_main``

**Don't use tags for:**
    - Display labels (use names)
    - Long descriptive text (use descriptions)
    - User-facing identifiers

Names
-----

**Use Nameable when:**
    - Objects need human-readable labels
    - UI display text is required
    - Log messages need object identification

**Name guidelines:**
    - Use Title Case for UI elements
    - Include units/context: "Temperature (°C)"
    - Keep under ~50 characters for display
    - Use full words, not abbreviations

Colors
------

**Use Colorable when:**
    - Visual distinction is needed
    - Plotting/visualization is involved
    - UI theming is required

**Color best practices:**
    - Use meaningful color schemes (red=error, green=ok)
    - Ensure sufficient contrast
    - Consider colorblind accessibility
    - Document color meanings

Activation
----------

**Use Activatable when:**
    - Objects can be temporarily disabled
    - Conditional processing is needed
    - Feature flags are required

**Activation patterns:**
    - Default to active unless there's a reason
    - Document what "inactive" means for your object
    - Consider whether deletion is more appropriate


Advanced Topics
===============

Equality and Hashing
---------------------

``Identifiable`` provides ID-based equality and hashing::

    comp1 = Component()
    comp2 = Component()

    # Different IDs
    assert comp1 != comp2
    assert hash(comp1) != hash(comp2)

    # Same object from registry
    retrieved = Identifiable.get_instance(comp1.id)
    assert comp1 == retrieved
    assert hash(comp1) == hash(retrieved)

Override equality for content comparison::

    class DataComponent(Identifiable):
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            if not isinstance(other, DataComponent):
                return NotImplemented
            # Compare content, not ID
            return self.value == other.value

        # Keep ID-based hash
        # __hash__ is inherited from Identifiable

Weak References
---------------

The ``Identifiable`` registry uses weak references::

    comp = Component()
    comp_id = comp.id

    # Object is in registry
    assert Identifiable.get_instance(comp_id) is comp

    # Delete all strong references
    del comp

    # Object can be garbage collected
    assert Identifiable.get_instance(comp_id) is None

To keep objects alive, maintain strong references::

    registry = {}

    comp = Component()
    registry[comp.id] = comp  # Strong reference

    # Now comp won't be garbage collected

Property Customization
----------------------

Override defaults in subclasses::

    class InactiveByDefault(Activatable):
        active = SerializableProperty(default=False)

    obj = InactiveByDefault()
    assert obj.active is False

Multiple inheritance resolution::

    class Base(Identifiable):
        def __eq__(self, other):
            # Custom equality
            return super().__eq__(other) and self.custom_check(other)

    # ID-based hash still works from Identifiable


Troubleshooting
===============

Common Errors
-------------

**ValueError: Invalid tag**::

    obj.tag = "invalid-tag"  # Hyphens not allowed
    # Fix: Use underscores
    obj.tag = "invalid_tag"

**TypeError: active must be a boolean**::

    obj.active = 1  # Not a bool
    # Fix: Use True/False
    obj.active = True

**AttributeError: write-once property**::

    comp.id = "new-id"  # Can't change ID
    # Fix: IDs are immutable

**ValueError: Invalid UUID**::

    comp = Component()
    comp.id = "not-a-uuid"  # Invalid format
    # Fix: Let ID auto-generate or provide valid UUID

Tag Not Found
-------------

If namespace access doesn't work::

    system.sensor_a  # AttributeError

    # Check tag is set
    sensor = get_sensor()
    if sensor.tag is None:
        sensor.tag = "sensor_a"
    system.add(sensor)

Garbage Collection
------------------

If ``get_instance()`` returns None unexpectedly::

    # Object was garbage collected
    comp_id = create_component().id  # Temporary object!
    comp = Identifiable.get_instance(comp_id)  # None

    # Fix: Keep strong reference
    comp = create_component()
    comp_id = comp.id
    # Now comp stays alive


See Also
========

* :doc:`../serialization/serializable_property` - Property descriptor used by mixins
* :doc:`../serialization/serializable` - Serialization support