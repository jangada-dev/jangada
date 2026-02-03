******************
Helper Functions
******************

.. currentmodule:: jangada.serialization

Utility functions used throughout the serialization system.


Function Reference
==================

.. autosummary::
   :toctree: generated/

   get_full_qualified_name
   check_types


Detailed Documentation
======================

``get_full_qualified_name``
---------------------------

.. autofunction:: get_full_qualified_name

Returns the fully qualified name of a class in the format ``module.qualname``.
This is used to create unique identifiers for classes in the serialization registry.

**Examples**::

    >>> from mypackage import MyClass
    >>> get_full_qualified_name(MyClass)
    'mypackage.MyClass'

    >>> get_full_qualified_name(int)
    'int'

    >>> get_full_qualified_name(list)
    'list'

**Usage in Serialization**::

    # The __class__ key in serialized data uses qualified names
    class MyClass(Serializable):
        value = SerializableProperty(default=0)

    obj = MyClass(value=42)
    data = Serializable.serialize(obj)

    print(data['__class__'])
    # Output: 'mymodule.MyClass'

**Notes**:

- Builtin types (like ``int``, ``str``, ``list``) return just the ``qualname``
- User-defined classes return ``module.qualname``
- Nested classes include their parent: ``'module.OuterClass.InnerClass'``


``check_types``
---------------

.. autofunction:: check_types

Type checking utility with optional None handling and flexible error behavior.

**Basic Usage**::

    >>> check_types(42, int)
    True

    >>> check_types("hello", str)
    True

    >>> check_types([1, 2, 3], list)
    True

**Multiple Types**::

    >>> check_types(42, (int, float))
    True

    >>> check_types(3.14, (int, float))
    True

    >>> check_types("text", (int, float), raise_error=False)
    False

**Allowing None**::

    >>> check_types(None, int, can_be_none=True)
    True

    >>> check_types(42, int, can_be_none=True)
    True

    >>> check_types(None, int, can_be_none=False)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    TypeError: Expected instance of one of the following classes: int...

**Error Behavior**::

    # With raise_error=True (default)
    >>> check_types(42, str)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    TypeError: Expected instance of one of the following classes: str. Given int instead

    # With raise_error=False
    >>> check_types(42, str, raise_error=False)
    False
    >>> check_types("hello", str, raise_error=False)
    True

**Usage in Validation**::

    def process_data(data):
        # Validate input type
        check_types(data, dict)

        # Process data...
        pass

    # Or with optional None
    def process_optional_data(data):
        check_types(data, dict, can_be_none=True)

        if data is not None:
            # Process data...
            pass

**Error Messages**:

The error message includes the fully qualified names of expected and actual types::

    >>> check_types(MyCustomClass(), ExpectedClass)  # doctest: +SKIP
    TypeError: Expected instance of one of the following classes:
    mypackage.ExpectedClass. Given mypackage.MyCustomClass instead


Use Cases
=========

Class Name Resolution
---------------------

``get_full_qualified_name`` is essential for the serialization registry::

    # When a class is defined
    class MyClass(Serializable):
        pass

    # The metaclass registers it:
    qualname = get_full_qualified_name(MyClass)
    Serializable._subclasses[qualname] = MyClass

    # Later, during deserialization:
    data = {'__class__': 'mymodule.MyClass', ...}
    cls = Serializable[data['__class__']]  # Retrieves MyClass


Type Validation
---------------

``check_types`` is used throughout the codebase for validation::

    # In Serializable._initialize_from_data
    def _initialize_from_data(self, data):
        check_types(data, dict)  # Ensure data is a dict
        # ... rest of initialization

    # In property parsers
    @property.parser
    def property(self, value):
        check_types(value, (int, float))
        return float(value)

    # In custom validation
    def validate_input(obj):
        check_types(obj, Serializable, can_be_none=False)
        # ... further validation


API Consistency
---------------

Using these helpers ensures consistent:

- Error messages across the codebase
- Type checking behavior
- Class name formatting


Implementation Notes
====================

``get_full_qualified_name``
---------------------------

**Edge Cases**:

- Anonymous classes (``lambda``, ``type()``) may have unusual qualified names
- Classes defined in ``__main__`` will have ``'__main__'`` as their module
- Nested classes include parent: ``'Outer.Inner'``

**Performance**:

This function is called frequently during (de)serialization. It's kept simple
for performance - just attribute access and string formatting.


``check_types``
---------------

**Implementation Details**:

When ``can_be_none=True``, the function adds ``type(None)`` to the types tuple::

    if can_be_none:
        if isinstance(types, tuple):
            types = (*types, type(None))
        else:
            types = (types, type(None))

This allows ``isinstance(None, types)`` to succeed.

**Why Not Use typing.isinstance()?**:

- ``check_types`` is simpler and more explicit
- Provides custom error messages with qualified names
- Handles the ``can_be_none`` flag cleanly
- No dependency on ``typing`` module


**Performance**:

Type checking is fast (uses built-in ``isinstance``). The error message
construction only happens on failure, so the happy path is efficient.


See Also
========

* :doc:`serializable` - Main class using these helpers
* :doc:`serializable_metatype` - Metaclass that uses get_full_qualified_name
* :doc:`serializable_property` - Property descriptor that may use check_types