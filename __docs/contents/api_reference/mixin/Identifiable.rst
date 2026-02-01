.. _api-identifiable:

==============
Identifiable
==============

.. currentmodule:: jangada.mixin

.. autoclass:: Identifiable
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __hash__

Overview
========

The :class:`Identifiable` mixin provides automatic unique identifier management
for objects that need to be tracked and retrieved by ID. It uses UUID v4 for
globally unique identifiers and maintains a weak reference registry for instance
lookup.

Key Features
============

Automatic UUID Generation
-------------------------

IDs are generated automatically on first access using UUID v4, ensuring global
uniqueness without coordination:

.. code-block:: python

   obj = Identifiable()
   print(obj.id)  # Automatically generated
   # '02248d1fd3c14f3aa16cb1eb61d0d68e'

Write-Once Immutability
-----------------------

Once an ID is assigned (either automatically or manually), it cannot be changed:

.. code-block:: python

   obj = Identifiable()
   obj.id = 'custom-uuid-here'  # Set before first access: OK
   obj.id = 'different-uuid'    # Try to change: ERROR

Copy Protection
---------------

When copying objects, new IDs are generated rather than copying the original:

.. code-block:: python

   original = Identifiable()
   copy = Identifiable(original)

   assert original.id != copy.id  # Different IDs

Instance Registry
-----------------

All instances are tracked in a global registry for lookup by ID:

.. code-block:: python

   obj = Identifiable()
   obj_id = obj.id

   retrieved = Identifiable.get_instance(obj_id)
   assert retrieved is obj

Usage Examples
==============

Basic Usage
-----------

Simple mixin usage for objects that need unique IDs:

.. code-block:: python

   from jangada.atomic import Identifiable

   class Document(Identifiable):
       def __init__(self, title):
           self.title = title

   doc = Document("My Document")
   print(doc.id)  # Auto-generated UUID
   print(doc)     # Document(id='...')

Custom ID Assignment
--------------------

Set a specific ID before first access:

.. code-block:: python

   doc = Document("Important Doc")
   doc.id = '550e8400e29b41d4a716446655440000'
   # ID is now set and immutable

Multiple Mixins
---------------

Combine with other mixins for rich object metadata:

.. code-block:: python

   from jangada.mixin import Identifiable, Nameable, Describable

   class Asset(Identifiable, Nameable, Describable):
       pass

   asset = Asset()
   asset.name = "Server-01"
   asset.description = "Production web server"
   print(asset.id)    # UUID
   print(asset.name)  # "Server-01"

Using in Collections
--------------------

Identifiable objects are hashable and can be used in sets and dicts:

.. code-block:: python

   obj1 = Identifiable()
   obj2 = Identifiable()

   # Use in sets
   unique_objects = {obj1, obj2}

   # Use as dict keys
   metadata = {
       obj1: {"status": "active"},
       obj2: {"status": "archived"},
   }

Instance Lookup
---------------

Retrieve objects by their ID:

.. code-block:: python

   # Create and store ID
   obj = Identifiable()
   obj_id = obj.id

   # Later, retrieve by ID
   retrieved = Identifiable.get_instance(obj_id)
   assert retrieved is obj

With Serialization
------------------

The ID property integrates seamlessly with serialization:

.. code-block:: python

   from jangada.serialization import Persistable, SerializableProperty

   class SaveableDocument(Persistable, Identifiable):

       extension = '.mydoc'

       title = SerializableProperty()
       content = SerializableProperty()

   doc = SaveableDocument()
   doc.title = "My Doc"
   doc.save('document.mydoc')

   # Later...
   loaded = SaveableDocument.load('document.mydoc')
   assert loaded.id == doc.id  # Same ID preserved

Implementation Details
======================

ID Format
---------

IDs are 32-character hexadecimal strings (UUID v4 without hyphens):

.. code-block:: python

   obj = Identifiable()
   print(obj.id)
   # '550e8400e29b41d4a716446655440000'
   print(len(obj.id))  # 32

The format is the hexadecimal representation of a UUID v4, which provides:

- 122 bits of randomness
- Vanishingly small collision probability (< 1 in 10^36)
- No coordination required between systems

Lazy Generation
---------------

IDs are not generated at object creation, but only when first accessed:

.. code-block:: python

   obj = Identifiable()  # No ID yet

   # ID generated here on first access
   my_id = obj.id

This allows setting custom IDs before the automatic generation occurs.

Weak References
---------------

The instance registry uses :py:class:`weakref.WeakValueDictionary`, which means:

- Objects can be garbage collected normally
- The registry doesn't prevent cleanup
- Lookups may return None if object was collected

.. code-block:: python

   obj = Identifiable()
   obj_id = obj.id

   del obj  # Object can be garbage collected

   # Later lookup returns None
   retrieved = Identifiable.get_instance(obj_id)
   assert retrieved is None

Thread Safety
-------------

.. warning::
   The instance registry is **not** thread-safe. If you're using Identifiable
   objects across multiple threads, you should implement your own synchronization.

Performance Considerations
==========================

Memory Usage
------------

- Each Identifiable object stores a 32-character string (~37 bytes for the string object)
- The instance registry adds one weak reference per object (~80 bytes)
- Total overhead: ~120 bytes per instance

Lookup Performance
------------------

- ID generation: O(1) - simple UUID v4 generation
- Instance lookup: O(1) - dictionary lookup
- Hash computation: O(1) - hash of 32-character string

Best Practices
==============

Do's
----

✓ Use as a mixin with other classes
✓ Set custom IDs before first access if needed
✓ Use :meth:`get_instance` for lookups by ID
✓ Rely on automatic generation for most use cases
✓ Include ID in serialization for persistence

Don'ts
------

✗ Don't try to modify IDs after they're set
✗ Don't assume IDs can be looked up after garbage collection
✗ Don't use across threads without synchronization
✗ Don't use the same ID for multiple objects
✗ Don't store IDs separately - use :meth:`get_instance`

Troubleshooting
===============

ID Not Generated
----------------

**Problem**: Accessing ``obj.id`` returns ``None``

**Solution**: This shouldn't happen with the default implementation. If it does,
check that the default generator is set up correctly.

Lookup Returns None
-------------------

**Problem**: :meth:`get_instance` returns ``None`` for a valid ID

**Causes**:

1. Object was garbage collected (most common)
2. ID doesn't exist in the system
3. Registry was cleared with :meth:`reset_instance_registry`

**Solution**: Keep a strong reference to objects you need to retrieve later.

Cannot Change ID
----------------

**Problem**: Getting ``AttributeError`` when setting ID

**Cause**: ID was already set (either automatically or manually)

**Solution**: Set the ID before first access:

.. code-block:: python

   obj = Identifiable()
   obj.id = 'custom-id'  # Set BEFORE first access
   # Now obj.id is set and immutable

Invalid UUID Error
------------------

**Problem**: ``ValueError`` when setting custom ID

**Cause**: The provided value is not a valid UUID v4 format

**Solution**: Use proper UUID v4 format (32 hex characters):

.. code-block:: python

   import uuid

   obj = Identifiable()
   obj.id = str(uuid.uuid4().hex)  # Valid UUID v4

See Also
========

* :class:`Nameable` - Mixin for human-readable names
* :class:`Taggable` - Mixin for mnemonic tags
* :class:`Describable` - Mixin for descriptions
* :class:`SerializableProperty` - Property descriptor system
* :class:`Serializable` - Serialization framework

API Reference
=============

.. autosummary::
   :toctree: generated/

   Identifiable.id
   Identifiable.get_instance
   Identifiable.reset_instance_registry