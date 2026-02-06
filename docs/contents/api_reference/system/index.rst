###################
Hierarchical System
###################

.. currentmodule:: jangada.system

The system module provides hierarchical namespace organization for domain objects.
Systems can be nested arbitrarily deep with automatic parent-child relationship
management, tag-based lookup, and dynamic reorganization.


Overview
========

The System class is the culmination of all Jangada's mixins, providing:

- **Hierarchical organization** - Tree-structured subsystem relationships
- **Namespace access** - Dict-style and attribute-style subsystem lookup
- **Dynamic reorganization** - Automatic updates when tags change
- **Circular dependency prevention** - Validates hierarchy constraints
- **Full persistence** - Save/load entire hierarchies
- **Rich display** - Beautiful terminal visualization

.. autosummary::
   :toctree: generated/

   System


Quick Start
===========

Basic hierarchy::

    from jangada.system import System

    # Create systems
    root = System(tag='root', name='Root System')
    child = System(tag='child', name='Child System')

    # Build hierarchy
    root.add(child)

    # Access subsystems
    root['child']    # Dict-style
    root.child       # Attribute-style

Dict-style registration::

    system = System(tag='system')
    system['sensors'] = System(name='Sensors')
    system['controllers'] = System(name='Controllers')

    # Tags are automatically set from keys
    assert system['sensors'].tag == 'sensors'

Complex hierarchy::

    system = System(tag='system')

    # Build via dict-style
    system['sensors'] = System(name='Sensors')
    system['sensors']['temperature'] = System(name='Temp Sensor')
    system['sensors']['pressure'] = System(name='Pressure Sensor')

    # Navigate
    temp = system.sensors.temperature
    print(temp.name)  # 'Temp Sensor'


System Reference
================

.. autoclass:: System
    :members:
    :inherited-members:
    :show-inheritance:


Core Concepts
=============

Hierarchy Structure
-------------------

Systems form a **tree structure** (not DAG). Each system has:

- **Zero or one parent** (supersystem)
- **Zero or more children** (subsystems)
- **No circular dependencies** (automatically prevented)

Example hierarchy::

    root
    ├── sensors
    │   ├── temperature
    │   └── pressure
    └── controllers
        └── main

Tag-Based Access
----------------

Subsystems are accessed by their tags using either dict-style or attribute-style
syntax:

**Dict-Style**::

    system['sensor_name']
    system['sensor_name'] = System()

    if 'sensor_name' in system:
        sensor = system['sensor_name']

**Attribute-Style**::

    system.sensor_name
    # Note: __setattr__ not yet implemented

    if hasattr(system, 'sensor_name'):
        sensor = system.sensor_name

**Which to use?**

- Dict-style: More explicit, works with any valid identifier
- Attribute-style: Cleaner syntax for static structures

Bidirectional Relationships
----------------------------

Parent-child relationships are **bidirectional** and automatically managed:

**Setting parent automatically updates child list**::

    parent = System(tag='parent')
    child = System(tag='child')

    child.supersystem = parent
    # Automatically: parent.subsystems['child'] = child

**Setting child automatically updates parent**::

    parent = System(tag='parent')
    child = System(tag='child')

    parent.add(child)
    # Automatically: child.supersystem = parent

**Reparenting**::

    parent1 = System(tag='parent1')
    parent2 = System(tag='parent2')
    child = System(tag='child')

    parent1.add(child)
    parent2.add(child)  # Removes from parent1, adds to parent2

Dynamic Tag Reorganization
---------------------------

When a subsystem's tag changes, parent dicts **update automatically** via observers:

Example::

    root = System(tag='root')
    child = System(tag='old_tag')
    root.add(child)

    assert 'old_tag' in root

    # Change tag
    child.tag = 'new_tag'

    # Dict keys updated automatically
    assert 'new_tag' in root
    assert 'old_tag' not in root


Usage Patterns
==============

Building Hierarchies
--------------------

**Incremental construction**::

    system = System(tag='system')

    sensors = System(tag='sensors')
    system.add(sensors)

    temp = System(tag='temperature')
    sensors.add(temp)

**Nested construction**::

    system = System(tag='system')
    system['sensors'] = System()
    system['sensors']['temperature'] = System()
    system['sensors']['pressure'] = System()

**Bulk addition**::

    system = System(tag='system')

    components = [
        System(tag='comp1'),
        System(tag='comp2'),
        System(tag='comp3'),
    ]

    system.add(*components)

Navigation
----------

**Accessing subsystems**::

    # Direct access
    sensor = system.sensors.temperature

    # Via parent reference
    parent = sensor.supersystem
    sibling = parent.pressure

**Walking hierarchy**::

    # Iterate direct children
    for subsystem in system:
        print(subsystem.tag)

    # Get ancestor chain
    for ancestor in system.supersystem_chain:
        print(ancestor.tag)

**Checking membership**::

    if sensor in system:
        print("Direct child")

    if 'temperature' in system.sensors:
        print("Subsystem exists")

Persistence
-----------

**Save hierarchy**::

    root = System(tag='root')
    root['child'] = System(tag='child', name='Child')

    root.save('hierarchy.sys')

**Load hierarchy**::

    loaded = System.load('hierarchy.sys')

    # Hierarchy reconstructed
    assert loaded.child.name == 'Child'
    assert loaded.child.supersystem is loaded

**Content comparison**::

    sys1 = System(tag='test', name='Name')
    sys2 = System(tag='test', name='Name')

    # Different IDs
    assert sys1 != sys2  # ID-based equality

    # Same content
    assert sys1.equal(sys2)  # Content-based equality

Display
-------

**Terminal output**::

    system = System(tag='system', name='My System')
    system['sensor'] = System(tag='sensor', name='Sensor')

    print(system)  # Rich formatted panel

**Customize display**::

    system.display_settings.panel_border_style = 'green'
    system.display_settings.console_width = 120
    print(system)

**Export to HTML/SVG**::

    html = system.to_html()
    svg = system.to_svg()


Advanced Topics
===============

Validation and Constraints
---------------------------

Systems automatically validate:

**Tag requirements**::

    child = System(tag=None)
    parent.add(child)  # ValueError: must have tag

**Circular dependencies**::

    sys_a = System(tag='a')
    sys_b = System(tag='b')

    sys_a.add(sys_b)
    sys_b.add(sys_a)  # ValueError: circular dependency

**Tag conflicts**::

    parent = System(tag='parent')
    parent['sensor'] = System()
    parent['sensor'] = System()  # OK: replaces

    child1 = System(tag='sensor')
    child2 = System(tag='sensor')
    parent.add(child1)
    parent.add(child2)  # ValueError: tag conflict

Observer Pattern
----------------

System uses observers to maintain consistency when tags change:

**How it works**::

    1. Subsystem added → Observer attached to tag property
    2. Tag changes → Observer notified with old and new values
    3. Observer updates parent's subsystems dict
    4. Validates no conflicts, updates keys

**Observer lifecycle**::

    parent = System(tag='parent')
    child = System(tag='child')

    parent.add(child)
    # → Observer created and attached to child.tag

    child.tag = 'renamed'
    # → Observer fires, updates parent.subsystems

    parent.remove(child)
    # → Observer detached and cleaned up

**Manual observer access** (advanced)::

    # Check observers
    observers = parent._tag_observers

    # observers[child] contains the observer function

Identity vs Content Equality
-----------------------------

System supports two types of equality:

**ID-Based (``==`` operator)**::

    Uses Identifiable.__eq__

    sys1 = System(tag='test')
    sys2 = System(tag='test')

    sys1 == sys2  # False (different IDs)

**Content-Based (``equal()`` method)**::

    Uses Persistable.__eq__

    sys1 = System(tag='test', name='Name')
    sys2 = System(tag='test', name='Name')

    sys1.equal(sys2)  # True (same content)

**Use cases**:

- ``==``: Check if same object (by ID)
- ``equal()``: Check if same structure/data (for copies)

Inheritance Hierarchy
---------------------

System inherits from five base classes:

.. code-block:: text

    System
    ├── Persistable      → Save/load to HDF5
    ├── Displayable      → Rich terminal display
    ├── Identifiable     → Unique ID, hash, equality
    ├── Taggable         → Symbolic tag for namespace
    ├── Nameable         → Human-readable name
    └── Describable      → Free-form description

Each mixin provides orthogonal capabilities that System combines.


API Design Decisions
====================

Why Store Parent as ID?
------------------------

The ``_supersystem_id`` property stores the parent's **ID** rather than a
direct reference. This design:

**Advantages**:

- Avoids circular references in serialization
- Allows garbage collection (weak reference behavior)
- Enables proper save/load of hierarchies
- Works with Identifiable's global registry

**Trade-offs**:

- Requires ID lookup on every ``supersystem`` access
- Parent must exist in Identifiable registry
- Slightly more complex implementation

Why Mutable Tags?
-----------------

Tags are **mutable** (can be changed after creation). This design:

**Advantages**:

- Dynamic reorganization without re-creating objects
- Flexible namespace evolution
- Observer pattern enables automatic updates

**Trade-offs**:

- More complex implementation (observers required)
- Potential for tag conflicts (validated automatically)
- Users must be careful when changing tags

Tree vs DAG Structure
---------------------

System enforces a **tree structure** (single parent per system) rather than
a DAG (multiple parents).

**Why trees?**:

- Simpler mental model
- Easier serialization
- Clear ownership semantics
- Prevents complex circular dependency scenarios

**If you need DAG**:

- Use references instead of hierarchy
- Store IDs in custom properties
- Implement custom relationship management


Examples Gallery
================

Scientific Instrument System
----------------------------

Model a scientific instrument with subsystems::

    instrument = System(tag='instrument', name='UV-Vis Spectrometer')

    # Optical components
    instrument['optics'] = System(name='Optical System')
    instrument['optics']['source'] = System(name='Light Source')
    instrument['optics']['monochromator'] = System(name='Monochromator')
    instrument['optics']['detector'] = System(name='Detector')

    # Control systems
    instrument['control'] = System(name='Control System')
    instrument['control']['temperature'] = System(name='Temp Controller')
    instrument['control']['shutter'] = System(name='Shutter Controller')

    # Access
    detector = instrument.optics.detector
    print(f"Path: {detector.supersystem.supersystem.tag}")

Data Processing Pipeline
------------------------

Build a data processing pipeline::

    pipeline = System(tag='pipeline', name='Data Pipeline')

    # Stages
    pipeline['ingest'] = System(name='Data Ingest')
    pipeline['clean'] = System(name='Data Cleaning')
    pipeline['transform'] = System(name='Transformation')
    pipeline['export'] = System(name='Export')

    # Configure each stage
    for stage in pipeline:
        stage.display_settings.panel_border_style = 'green'
        print(f"Stage: {stage.name}")

Organization Hierarchy
----------------------

Model an organization structure::

    company = System(tag='company', name='Acme Corp')

    # Departments
    company['engineering'] = System(name='Engineering')
    company['engineering']['backend'] = System(name='Backend Team')
    company['engineering']['frontend'] = System(name='Frontend Team')

    company['sales'] = System(name='Sales')
    company['sales']['enterprise'] = System(name='Enterprise Sales')

    # Reorganize
    company['engineering']['devops'] = System(name='DevOps Team')

    # Move team
    company['operations'] = System(name='Operations')
    company['operations']['devops'] = company['engineering']['devops']
    # DevOps moved from Engineering to Operations

Configuration Management
------------------------

Hierarchical configuration::

    config = System(tag='config', name='Application Config')

    # Database settings
    config['database'] = System(name='Database')
    config['database'].description = 'PostgreSQL connection params'
    config['database']['host'] = System(name='localhost')
    config['database']['port'] = System(name='5432')

    # API settings
    config['api'] = System(name='API')
    config['api']['base_url'] = System(name='https://api.example.com')
    config['api']['timeout'] = System(name='30')

    # Access configuration
    db_host = config.database.host.name


Best Practices
==============

Naming Conventions
------------------

**Tags** (machine identifiers):

- Use lowercase with underscores: ``sensor_temp``
- Be descriptive but concise
- Include type/category: ``ctrl_main``, ``sensor_pressure``
- Avoid abbreviations unless standard

**Names** (human labels):

- Use Title Case: ``"Temperature Sensor"``
- Include units where relevant: ``"Temp (°C)"``
- Be descriptive: ``"Primary Temperature Sensor"``

Organization
------------

**Logical grouping**::

    system = System(tag='system')
    system['sensors'] = System()       # Group by type
    system['controllers'] = System()
    system['processors'] = System()

**Hierarchical decomposition**::

    root = System(tag='root')
    root['subsystem_a'] = System()
    root['subsystem_a']['component_1'] = System()
    root['subsystem_a']['component_2'] = System()

**Keep hierarchies shallow**: 3-5 levels is typically sufficient.

Error Handling
--------------

**Check before accessing**::

    if 'sensor' in system:
        sensor = system['sensor']
    else:
        print("Sensor not found")

**Handle KeyError/AttributeError**::

    try:
        sensor = system['nonexistent']
    except KeyError:
        print("Subsystem not found")

    try:
        sensor = system.nonexistent
    except AttributeError:
        print("Subsystem not found")

**Validate before adding**::

    if child.tag is None:
        child.tag = 'default_tag'

    if child.tag not in parent:
        parent.add(child)

Memory Management
-----------------

**Weak references**: Identifiable uses weak references, so systems can be
garbage collected when no longer referenced.

**Break cycles**: When deleting systems, explicitly remove from parent::

    parent.remove(child)
    del child

**Large hierarchies**: Be mindful of memory usage with deep/wide trees.


Troubleshooting
===============

Common Errors
-------------

**ValueError: Subsystems must have a tag**::

    # Problem
    child = System(tag=None)
    parent.add(child)

    # Solution
    child.tag = 'child_tag'
    parent.add(child)

**ValueError: circular dependency**::

    # Problem
    sys_a.add(sys_b)
    sys_b.add(sys_a)

    # Solution: Don't create cycles
    # Use references instead if bidirectional link needed

**ValueError: tag already used**::

    # Problem
    parent['sensor'] = System()
    child = System(tag='sensor')
    parent.add(child)

    # Solution: Use different tag or replace
    parent['sensor'] = child  # Replaces existing

**KeyError: No subsystem registered**::

    # Problem
    system['nonexistent']

    # Solution: Check first
    if 'nonexistent' in system:
        subsys = system['nonexistent']

Tag Observer Issues
-------------------

**Tag conflicts during reorganization**::

    # If tag observer raises error
    try:
        child.tag = 'conflicting_tag'
    except ValueError as e:
        print(f"Tag change failed: {e}")
        # Tag remains unchanged

**Observer not firing**::

    # Verify subsystem is registered
    assert child in parent

    # Change tag
    child.tag = 'new_tag'

    # Verify update
    assert 'new_tag' in parent

Persistence Issues
------------------

**Hierarchy not reconstructed**::

    # Ensure entire tree is saved
    root.save('hierarchy.sys')  # Saves root and all children

    # Not this:
    child.save('child.sys')  # Only saves child, loses parent

**IDs change after load**::

    # IDs are non-copiable, so each load gets new IDs
    original_id = system.id
    system.save('system.sys')
    loaded = System.load('system.sys')

    assert loaded.id != original_id  # Expected behavior


Performance Considerations
==========================

Hierarchy Depth
---------------

- **O(1)**: Dict-style access (``system['tag']``)
- **O(n)**: Supersystem chain (``system.supersystem_chain``)
- **O(d)**: Navigation by path (d = depth)

For deep hierarchies, cache supersystem chain if accessed frequently.

Tag Changes
-----------

Tag changes trigger observer which:

1. Validates new tag (O(1) dict lookup)
2. Updates parent dict (O(1) operations)
3. No recursive updates

Impact is minimal for most use cases.

Serialization
-------------

Saving a system serializes:

- The system itself
- All subsystems recursively
- All SerializableProperty data

For large hierarchies:

- Save can be slow (all subsystems written)
- Load reconstructs entire tree
- Consider splitting very large hierarchies


See Also
========

Related Modules
---------------

* :doc:`../serialization/persistable` - Persistence base class
* :doc:`../display/displayable` - Display base class
* :doc:`../mixin/identifiable` - Unique identification
* :doc:`../mixin/taggable` - Tag-based identification
* :doc:`../mixin/nameable` - Human-readable names
* :doc:`../mixin/describable` - Free-form descriptions

External Resources
------------------

* `Composite Pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_ - Design pattern for tree structures
* `Observer Pattern <https://en.wikipedia.org/wiki/Observer_pattern>`_ - Design pattern for event notification
