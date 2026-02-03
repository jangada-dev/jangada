#  -*- coding: utf-8 -*-
"""
A small, explicit serialization and persistence framework.

This module provides a descriptor-driven schema (``SerializableProperty``),
a registry-backed in-memory serialization protocol (``Serializable``), and an
HDF5 persistence layer (``Persistable``).

Design goals
------------
- Explicit schemas: Classes declare which attributes are serialized via
  ``SerializableProperty`` descriptors (including inheritance across the MRO).
- Portable serialized form: The in-memory serialized representation uses only
  Python-native containers (dict/list) plus registered primitive values, and a
  ``"__class__"`` envelope for reconstructing ``Serializable`` objects.
- Extensibility: New primitives and dataset-backed types can be registered
  globally through the metaclass API.
- Numerical efficiency: Large arrays and similar objects can be stored as true
  HDF5 datasets (not blobs), optionally accessed lazily through proxies.

Serialized representation
-------------------------
The serialized form returned by ``Serializable.serialize`` is a tree composed of:
- ``None``
- lists (always serialized as ``list``)
- dictionaries
- registered primitive values
- ``Serializable`` instances serialized as dictionaries that include a
  ``"__class__"`` key containing a fully qualified name.

The general object envelope is:

    {
        "__class__": "module.QualifiedName",
        "<property_name_1>": <serialized_value>,
        ...
    }

Primitive types
---------------
A "primitive type" is any type that is allowed to pass through serialization
unchanged (stored as-is in the in-memory representation). Examples in this
module include strings, numbers, and ``pathlib.Path`` (with persistence support
provided by ``Persistable``).

Dataset-backed types
--------------------
A "dataset type" is a type that:
- is treated as a primitive during in-memory serialization, but
- is persisted as an HDF5 dataset when using ``Persistable``.

Dataset types are registered with a pair of functions:

- ``disassemble(obj) -> (ndarray, attrs_dict)``
- ``assemble(ndarray, attrs_dict) -> obj``

During persistence, the ndarray becomes a dataset and attrs become dataset
attributes.

Forward-compatibility behavior
------------------------------
If ``Serializable.deserialize`` encounters a ``"__class__"`` name that is not
registered, it creates a synthetic class named ``_Generic<ClassName>`` that
accepts the serialized keys as properties. This allows loading data produced by
newer code with older code at the cost of type-specific behavior.

Notes
-----
This module is intentionally conservative: it does not attempt to serialize
arbitrary Python objects (unlike pickle). Anything not explicitly supported
must be registered as a primitive or dataset type, or be a ``Serializable``.
"""

from __future__ import annotations

import numpy
import pandas
import h5py

from abc import ABCMeta
from numbers import Number
from pathlib import Path

# ---------- ---------- ---------- ---------- ---------- ---------- typing
from typing import TypeVar, Callable, Any, TypeAlias, Self, Type

from numpy.typing import NDArray


T = TypeVar('T')
"""Type variable representing the type of the property value."""

DefaultCallable: TypeAlias = Callable[[object], T]
"""
Callable that produces a default value for a property.

Parameters
----------
instance : object
    The instance for which to generate the default value.

Returns
-------
T
    The default value.
"""

Observer: TypeAlias = Callable[[object, T, T], None]
"""
Callable that observes property value changes.

Parameters
----------
instance : object
    The instance whose property changed.
old_value : T
    The previous value of the property.
new_value : T
    The new value of the property.

Returns
-------
None
"""

Parser: TypeAlias = Callable[[object, Any], T]
"""
Callable that parses and validates property values.

Parameters
----------
instance : object
    The instance for which to parse the value.
value : Any
    The raw value to parse.

Returns
-------
T
    The parsed and validated value.
"""

Postinitializer: TypeAlias = Callable[[object], T]
"""
Callable that performs post-initialization setup.

Parameters
----------
instance : object
    The instance to initialize.

Returns
-------
None
"""


class SerializableProperty:
    """
     A descriptor for properties that support defaults, parsing, observation,
     and post-initialization hooks.

     SerializableProperty provides a rich descriptor implementation designed for
     scientific computing applications where properties need validation, change
     tracking, lazy initialization, and eventual serialization to disk (e.g., HDF5).

     Features
     --------
     - **Default values**: Static or factory-generated defaults
     - **Parsing/validation**: Transform and validate values before storage
     - **Change observation**: Track value changes with callback functions
     - **Post-initialization**: Run setup code after first property access
     - **Write-once semantics**: Optionally prevent reassignment after first set
     - **Serialization support**: Mark properties as copiable for persistence

     Parameters
     ----------
     postinitializer : Postinitializer | None, optional
         Function called once after the property is first set or accessed.
         Receives the instance as its only argument. Useful for lazy setup
         of related state or expensive initialization. Default is None.
     default : T | DefaultCallable | None, optional
         Default value for the property. Can be:
         - A static value (used directly)
         - A callable receiving the instance and returning the default
         - None (property defaults to None)
         Setting the property to None resets it to this default. Default is None.
     parser : Parser | None, optional
         Function to parse/validate values before storage. Receives the instance
         and raw value, returns the parsed value. Applied to both explicitly set
         values and defaults. Exceptions raised by the parser propagate to the
         caller. Default is None.
     observers : set[Observer] | None, optional
         Set of functions called after the property value changes. Each observer
         receives (instance, old_value, new_value). Observers are called after
         parsing and storage, but before post-initialization. Default is None.
     writeonce : bool, optional
         If True, the property can only be set once (including initialization).
         Subsequent sets raise AttributeError. Useful for immutable configuration.
         Default is False.
     copiable : bool, optional
         Flag indicating whether this property should be included when copying
         or serializing instances. Used by serialization systems to determine
         which properties to persist. Default is True.
     doc : str | None, optional
         Documentation string for the property. Default is None.

     Attributes
     ----------
     name : str
         Name of the property (set by __set_name__).
     owner : type
         Class that owns this descriptor (set by __set_name__).
     private_name : str
         Internal attribute name used to store the value on instances.
     writeonce : bool
         Whether the property is write-once (read-only after first set).
     copiable : bool
         Whether the property should be included in serialization.

     Examples
     --------
     Basic usage with static default:

     >>> class Experiment:
     ...     temperature = SerializableProperty(default=293.15)
     ...
     >>> exp = Experiment()
     >>> exp.temperature
     293.15
     >>> exp.temperature = 373.15
     >>> exp.temperature
     373.15

     Using a callable default for mutable objects:

     >>> class DataContainer:
     ...     data = SerializableProperty(default=lambda self: [])
     ...
     >>> c1 = DataContainer()
     >>> c2 = DataContainer()
     >>> c1.data.append(1)
     >>> c2.data.append(2)
     >>> c1.data
     [1]
     >>> c2.data
     [2]

     Using a parser for validation:

     >>> class PositiveValue:
     ...     value = SerializableProperty(
     ...         default=1.0,
     ...         parser=lambda self, v: max(0.0, float(v))
     ...     )
     ...
     >>> obj = PositiveValue()
     >>> obj.value = -5
     >>> obj.value
     0.0

     Using decorators for cleaner syntax:

     >>> class System:
     ...     data = SerializableProperty()
     ...
     ...     @data.default
     ...     def data(self):
     ...         return {"initialized": True}
     ...
     ...     @data.parser
     ...     def data(self, value):
     ...         if not isinstance(value, dict):
     ...             raise TypeError("data must be a dict")
     ...         return value
     ...
     ...     @data.add_observer
     ...     def data(self, old, new):
     ...         print(f"Data changed from {old} to {new}")

     Using post-initializer for lazy setup:

     >>> class LazyLoader:
     ...     data = SerializableProperty(default=None)
     ...
     ...     @data.postinitializer
     ...     def data(self):
     ...         print("Loading expensive data...")
     ...         if self.data is None:
     ...             self.data = list(range(1000))
     ...
     >>> loader = LazyLoader()
     >>> # No output yet - not accessed
     >>> _ = loader.data
     Loading expensive data...
     >>> len(loader.data)
     1000

     Write-once property for configuration:

     >>> class Config:
     ...     api_key = SerializableProperty(writeonce=True)
     ...
     >>> cfg = Config()
     >>> cfg.api_key = "secret123"
     >>> cfg.api_key = "different"  # Raises AttributeError
     Traceback (most recent call last):
         ...
     AttributeError: api_key is a write-once property and has already been set

     Notes
     -----
     - Properties are stored in instance.__dict__ with mangled names to avoid
       conflicts with user attributes.
     - Setting a property to None resets it to its default value.
     - Observers are called on first access (with old_value=None) when the
       property is initialized with its default.
     - Post-initializers run after the first set completes, allowing them to
       safely access the property value.
     - Parsers are applied to default values as well as explicitly set values.
     - All decorator methods (.default(), .parser(), etc.) return new descriptor
       instances rather than mutating the existing one.
     - Properties cannot be deleted (raises AttributeError).

     See Also
     --------
     property : Python's built-in property descriptor
     dataclasses.field : Similar concept for dataclasses
     """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __init__(self,
                 postinitializer: Postinitializer | None = None,
                 default: T | DefaultCallable | None = None,
                 parser: Parser | None = None,
                 observers: set[Observer] | None = None,
                 writeonce: bool = False,
                 copiable: bool = True,
                 doc: str | None = None) -> None:
        """
        Initialize a SerializableProperty descriptor.

        Parameters
        ----------
        postinitializer : Postinitializer | None, optional
            Post-initialization function.
        default : T | DefaultCallable | None, optional
            Default value or factory function.
        parser : Parser | None, optional
            Value parsing/validation function.
        observers : set[Observer] | None, optional
            Set of observer functions.
        writeonce : bool, optional
            Whether property can only be set once.
        copiable : bool, optional
            Whether property should be serialized.
        doc : str | None, optional
            Documentation string.
        """
        self._postinitializer: Postinitializer | None = postinitializer
        self._default: T | DefaultCallable | None = default
        self._parser: Parser | None = parser
        self._observers: set[Observer] = set() if observers is None else observers
        self._writeonce: bool = writeonce
        self._copiable: bool = copiable

        self.__doc__: str | None = doc

    # ---------- ---------- descriptor protocol
    def __set_name__(self, owner: type, name: str) -> None:
        """
        Called when the descriptor is assigned to a class attribute.

        This method is part of the descriptor protocol and is automatically
        called by Python when the class is created. It stores metadata about
        the property's name and owner class.

        Parameters
        ----------
        owner : type
            The class that contains this descriptor.
        name : str
            The name of the attribute this descriptor is assigned to.

        Notes
        -----
        This method sets three attributes:
        - name: The attribute name
        - owner: The owning class
        - private_name: Mangled name for storing the actual value

        The private_name uses a prefix to avoid namespace collisions with
        user-defined attributes.
        """
        self.name: str = name
        self.owner: type = owner
        self.private_name: str = f"_serializable_property__{name}"

    def __get__(self, instance: object|None, owner: type) -> T|Self:
        """
        Get the property value from an instance.

        This method implements the descriptor get protocol. When accessed from
        the class, it returns the descriptor itself (for introspection). When
        accessed from an instance, it returns the stored value, initializing
        it with the default if this is the first access.

        Parameters
        ----------
        instance : object | None
            The instance from which the property is accessed, or None if
            accessed from the class.
        owner : type
            The class that owns this descriptor.

        Returns
        -------
        T | Self
            If accessed from an instance, returns the property value (type T).
            If accessed from the class, returns the descriptor itself (Self).

        Notes
        -----
        On first access from an instance, this method:
        1. Calls __set__(instance, None) to initialize with default
        2. Returns the initialized value
        3. This triggers observers and post-initializer on first access

        Examples
        --------
        >>> class MyClass:
        ...     prop = SerializableProperty(default=42)
        ...
        >>> MyClass.prop  # Access from class
        <SerializableProperty object at 0x...>
        >>> obj = MyClass()
        >>> obj.prop  # Access from instance
        42
        """
        if instance is None:
            # Accessing from class, return descriptor for introspection
            return self

        try:
            return instance.__getattribute__(self.private_name)
        except AttributeError:
            # if it reaches here, then property is being accessed for the first
            # time and has never been set before
            self.__set__(instance, None)

            return instance.__getattribute__(self.private_name)

    def __set__(self, instance: object, value: Any) -> None:
        """
        Set the property value on an instance.

        This method implements the descriptor set protocol. It handles default
        value resolution, parsing, write-once enforcement, observer notification,
        and post-initialization.

        Parameters
        ----------
        instance : object
            The instance on which to set the property.
        value : Any
            The value to set. If None, the property is reset to its default.

        Raises
        ------
        AttributeError
            If the property is write-once and has already been set.
        Any exception raised by parser
            Parser exceptions propagate to the caller.

        Notes
        -----
        The set operation follows this sequence:
        1. Check if this is the first time being set
        2. Enforce write-once constraint if applicable
        3. If value is None, resolve to default (static or from callable)
        4. Apply parser if present (to both defaults and explicit values)
        5. Store the parsed value in instance.__dict__
        6. Call all observers with (instance, old_value, new_value)
        7. If first set, call post-initializer if present

        Setting to None always resets to the default value:

        >>> class Example:
        ...     value = SerializableProperty(default=10)
        ...
        >>> obj = Example()
        >>> obj.value = 50
        >>> obj.value
        50
        >>> obj.value = None  # Reset to default
        >>> obj.value
        10

        Write-once enforcement:

        >>> class Config:
        ...     setting = SerializableProperty(writeonce=True)
        ...
        >>> cfg = Config()
        >>> cfg.setting = "value1"
        >>> cfg.setting = "value2"  # Raises AttributeError
        Traceback (most recent call last):
            ...
        AttributeError: setting is a write-once property and has already been set
        """
        # find the current value (if any)
        # the property is considered unset if it is None or have never been set before

        try:
            current_value = instance.__getattribute__(self.private_name)
        except AttributeError:
            # if it reaches here, then property is being set for the first time
            current_value = None
            first_time = True
        else:
            first_time = False

        # ---------- ---------- ---------- ---------- handle write-once behavior
        if self.writeonce and current_value is not None:
            raise AttributeError(f"{self.name} is a write-once property and has already been set")

        # ---------- ---------- ---------- ---------- set value
        if value is None:
            # then the user is trying to reset the property to its default value
            if callable(self._default):
                value = self._default(instance)
            else:
                value = self._default

        if self._parser is not None:
            # the parser is applied to the default value as well
            value = self._parser(instance, value)

        setattr(instance, self.private_name, value)

        # ---------- ---------- ---------- ---------- call observers
        old_value = current_value
        new_value = self.__get__(instance, self.owner)

        for observer in self._observers:
            observer(instance, old_value, new_value)

        # ---------- ---------- ---------- ---------- pos-initialization
        if first_time and self._postinitializer is not None:
            self._postinitializer(instance)

    def __delete__(self, instance: object) -> None:
        """
        Prevent deletion of the property.

        SerializableProperty instances cannot be deleted from instances.
        This method always raises AttributeError.

        Parameters
        ----------
        instance : object
            The instance from which deletion is attempted.

        Raises
        ------
        AttributeError
            Always raised to prevent deletion.

        Examples
        --------
        >>> class MyClass:
        ...     prop = SerializableProperty()
        ...
        >>> obj = MyClass()
        >>> del obj.prop
        Traceback (most recent call last):
            ...
        AttributeError: can't delete attribute 'prop'
        """
        raise AttributeError(f"can't delete attribute '{self.name}'")

    # ========== ========== decorators
    def postinitializer(self, func: Postinitializer) -> Self:
        """
        Set a post-initialization function using decorator syntax.

        The post-initializer is called once after the property is first set
        or accessed. It receives the instance as its only argument and runs
        after the value has been stored and observers have been called.

        Parameters
        ----------
        func : Postinitializer
            Function to call after first initialization. Should accept one
            argument (the instance) and return None.

        Returns
        -------
        Self
            A new SerializableProperty instance with the initializer set.

        Notes
        -----
        This method creates a new descriptor instance rather than modifying
        the existing one, following the immutable descriptor pattern.

        The post-initializer can safely:
        - Access the property value (it's already set)
        - Modify the property value (won't trigger another initialization)
        - Set up related properties or state
        - Perform expensive setup operations

        Examples
        --------
        >>> class DataLoader:
        ...     data = SerializableProperty(default=None)
        ...
        ...     @data.postinitializer
        ...     def data(self):
        ...         print("Initializing data...")
        ...         if self.data is None:
        ...             self.data = load_expensive_data()
        ...
        >>> loader = DataLoader()
        >>> # Post-initializer not yet called
        >>> _ = loader.data  # First access triggers initialization
        Initializing data...
        """
        return type(self)(
            postinitializer=func,
            default=self._default,
            parser=self._parser,
            observers=self._observers,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def default(self, func: DefaultCallable) -> Self:
        """
        Set a default value factory using decorator syntax.

        The default factory is called when the property needs a default value
        (on first access or when explicitly set to None). It receives the
        instance as its argument and should return the default value.

        Parameters
        ----------
        func : DefaultCallable
            Function that generates default values. Should accept one argument
            (the instance) and return the default value.

        Returns
        -------
        Self
            A new SerializableProperty instance with the default factory set.

        Notes
        -----
        This method creates a new descriptor instance rather than modifying
        the existing one.

        Using a callable default is essential for mutable default values to
        avoid sharing state between instances (similar to avoiding mutable
        default arguments in function definitions).

        Examples
        --------
        >>> class Container:
        ...     items = SerializableProperty()
        ...
        ...     @items.default
        ...     def items(self):
        ...         return []  # New list for each instance
        ...
        >>> c1 = Container()
        >>> c2 = Container()
        >>> c1.items.append(1)
        >>> c2.items.append(2)
        >>> c1.items
        [1]
        >>> c2.items
        [2]

        The default factory can access instance attributes:

        >>> class Experiment:
        ...     trial_number = SerializableProperty(default=1)
        ...     data_file = SerializableProperty()
        ...
        ...     @data_file.default
        ...     def data_file(self):
        ...         return f"trial_{self.trial_number}.dat"
        ...
        >>> exp = Experiment()
        >>> exp.data_file
        'trial_1.dat'
        """
        return type(self)(
            postinitializer=self._postinitializer,
            default=func,
            parser=self._parser,
            observers=self._observers,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def parser(self, func: Parser) -> Self:
        """
        Set a parser/validator function using decorator syntax.

        The parser is called on every value set operation (including defaults)
        to validate, transform, or normalize the value before storage.

        Parameters
        ----------
        func : Parser
            Function to parse values. Should accept two arguments (instance
            and raw value) and return the parsed value.

        Returns
        -------
        Self
            A new SerializableProperty instance with the parser set.

        Notes
        -----
        This method creates a new descriptor instance rather than modifying
        the existing one.

        Parsers are applied to:
        - Explicitly set values
        - Default values (both static and from factories)
        - Values set to None (after resolving to default)

        If a parser raises an exception, it propagates to the caller and the
        property value is not changed.

        Examples
        --------
        Type conversion:

        >>> class TypedProperty:
        ...     value = SerializableProperty()
        ...
        ...     @value.parser
        ...     def value(self, val):
        ...         return int(val)
        ...
        >>> obj = TypedProperty()
        >>> obj.value = "123"
        >>> obj.value
        123
        >>> type(obj.value)
        <class 'int'>

        Validation:

        >>> class PositiveNumber:
        ...     number = SerializableProperty(default=1)
        ...
        ...     @number.parser
        ...     def number(self, val):
        ...         val = float(val)
        ...         if val <= 0:
        ...             raise ValueError("Number must be positive")
        ...         return val
        ...
        >>> obj = PositiveNumber()
        >>> obj.number = 5
        >>> obj.number = -1
        Traceback (most recent call last):
            ...
        ValueError: Number must be positive

        Normalization:

        >>> class Path:
        ...     directory = SerializableProperty()
        ...
        ...     @directory.parser
        ...     def directory(self, val):
        ...         import os
        ...         return os.path.abspath(os.path.expanduser(val))
        """
        return type(self)(
            postinitializer=self._postinitializer,
            default=self._default,
            parser=func,
            observers=self._observers,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def add_observer(self, func: Observer) -> Self:
        """
        Add an observer function using decorator syntax.

        Observers are called after the property value changes. They receive
        the instance, old value, and new value as arguments.

        Parameters
        ----------
        func : Observer
            Function to observe changes. Should accept three arguments
            (instance, old_value, new_value) and return None.

        Returns
        -------
        Self
            A new SerializableProperty instance with the observer added.

        Notes
        -----
        This method creates a new descriptor instance with the observer added
        to the set of observers.

        Observers are called:
        - After the value has been parsed and stored
        - Before the post-initializer runs (if applicable)
        - In arbitrary order if multiple observers exist

        On first access, observers receive old_value=None and new_value=default.

        If an observer raises an exception, it propagates to the caller and
        subsequent observers are not called.

        Examples
        --------
        Track changes:

        >>> class Tracked:
        ...     value = SerializableProperty(default=0)
        ...
        ...     @value.add_observer
        ...     def value(self, old, new):
        ...         print(f"Value changed: {old} -> {new}")
        ...
        >>> obj = Tracked()
        >>> obj.value = 10
        Value changed: 0 -> 10
        >>> obj.value = 20
        Value changed: 10 -> 20

        Update dependent properties:

        >>> class Rectangle:
        ...     width = SerializableProperty(default=1.0)
        ...     height = SerializableProperty(default=1.0)
        ...     area = SerializableProperty(default=1.0)
        ...
        ...     @width.add_observer
        ...     def width(self, old, new):
        ...         self.area = self.width * self.height
        ...
        ...     @height.add_observer
        ...     def height(self, old, new):
        ...         self.area = self.width * self.height
        ...
        >>> rect = Rectangle()
        >>> rect.width = 5
        >>> rect.height = 3
        >>> rect.area
        15.0

        Multiple observers:

        >>> class MultiObserved:
        ...     value = SerializableProperty()
        ...
        ...     @value.add_observer
        ...     def value(self, old, new):
        ...         print(f"Observer 1: {old} -> {new}")
        ...
        >>> # Can chain add_observer calls
        >>> MultiObserved.value = MultiObserved.value.add_observer(
        ...     lambda self, old, new: print(f"Observer 2: {old} -> {new}")
        ... )
        """
        return type(self)(
            postinitializer=self._postinitializer,
            default=self._default,
            parser=self._parser,
            observers=self._observers | {func},
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def remove_observer(self, func: Observer) -> Self:
        """
        Remove an observer function.

        Creates a new descriptor instance without the specified observer.

        Parameters
        ----------
        func : Observer
            The observer function to remove.

        Returns
        -------
        Self
            A new SerializableProperty instance with the observer removed.

        Raises
        ------
        KeyError
            If the observer is not in the set of observers.

        Notes
        -----
        This method creates a new descriptor instance rather than modifying
        the existing one.

        Examples
        --------
        >>> def my_observer(instance, old, new):
        ...     print(f"Changed: {old} -> {new}")
        ...
        >>> class MyClass:
        ...     value = SerializableProperty(observers={my_observer})
        ...
        >>> # Remove the observer
        >>> MyClass.value = MyClass.value.remove_observer(my_observer)
        """
        return type(self)(
            postinitializer=self._postinitializer,
            default=self._default,
            parser=self._parser,
            observers=self._observers - {func},
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    # ========== ========== ========== ========== ========== properties
    @property
    def writeonce(self) -> bool:
        """
        Whether this property can only be set once.

        Returns
        -------
        bool
            True if the property is write-once, False otherwise.

        Notes
        -----
        This is a read-only property of the descriptor. The write-once flag
        is set during descriptor creation and cannot be changed afterward.

        Examples
        --------
        >>> class MyClass:
        ...     immutable = SerializableProperty(writeonce=True)
        ...     mutable = SerializableProperty(writeonce=False)
        ...
        >>> MyClass.immutable.writeonce
        True
        >>> MyClass.mutable.writeonce
        False
        """
        return self._writeonce

    @property
    def copiable(self) -> bool:
        """
        Whether this property should be included in serialization.

        Returns
        -------
        bool
            True if the property should be serialized, False otherwise.

        Notes
        -----
        This flag is used by serialization systems to determine which properties
        to include when persisting instances to disk (e.g., HDF5 files).

        Properties marked as non-copiable (copiable=False) are typically:
        - Cached/derived values that can be recomputed
        - Temporary state not needed after deserialization
        - References to external resources (file handles, network connections)

        This is a read-only property of the descriptor.

        Examples
        --------
        >>> class System:
        ...     data = SerializableProperty(copiable=True)
        ...     cache = SerializableProperty(copiable=False)
        ...
        >>> System.data.copiable
        True
        >>> System.cache.copiable
        False
        """
        return self._copiable


# ========== ========== ========== ========== ========== ==========
def get_full_qualified_name(cls: type) -> str:
    """
    Get the fully qualified name of a class.

    Returns the module path and class name (e.g., 'mypackage.module.ClassName').
    For builtin types without a module, returns just the qualified name.

    Parameters
    ----------
    cls : type
        The class for which to get the qualified name.

    Returns
    -------
    str
        The fully qualified name in the format 'module.qualname'.

    Examples
    --------
    >>> class MyClass:
    ...     pass
    >>> get_full_qualified_name(MyClass)
    '__main__.MyClass'

    >>> get_full_qualified_name(int)
    'int'

    Notes
    -----
    This function is used internally to create unique identifiers for classes
    in the serialization registry.
    """

    module = cls.__module__

    if module is None or module == 'builtins':
        return cls.__qualname__

    return f"{module}.{cls.__qualname__}"


def check_types(obj: Any,
                types: type | tuple[type],
                can_be_none: bool = False,
                raise_error: bool = True) -> bool:
    """
    Check if an object is an instance of the specified type(s).

    Parameters
    ----------
    obj : Any
        The object to check.
    types : type | tuple[type]
        A single type or tuple of types to check against.
    can_be_none : bool, optional
        If True, None is considered a valid value. Default is False.
    raise_error : bool, optional
        If True, raise TypeError on type mismatch. If False, return False instead.
        Default is True.

    Returns
    -------
    bool
        True if the object matches the expected type(s), False otherwise
        (only when raise_error=False).

    Raises
    ------
    TypeError
        If the object does not match the expected type(s) and raise_error=True.

    Examples
    --------
    >>> check_types(5, int)
    True

    >>> check_types("hello", (int, str))
    True

    >>> check_types(None, int, can_be_none=True)
    True

    >>> check_types(5, str, raise_error=False)
    False

    >>> check_types(5, str)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    TypeError: Expected instance of one of the following classes: str. Given int instead
    """
    if can_be_none:
        if isinstance(types, tuple):
            types = (*types, None.__class__)
        else:
            types = (types, None.__class__)

    result = isinstance(obj, types)

    if not result and raise_error:

        if isinstance(types, tuple):
            cls_names = ', '.join(get_full_qualified_name(cls) for cls in types)
        else:
            cls_names = get_full_qualified_name(types)

        error_msg = f"Expected instance of one of the following classes: {cls_names}. " \
                    f"Given {get_full_qualified_name(type(obj))} instead"
        raise TypeError(error_msg)

    return result


class SerializableMetatype(ABCMeta):
    """
    Metaclass for automatic registration and introspection of Serializable classes.

    This metaclass provides automatic registration of all Serializable subclasses,
    discovery of SerializableProperty descriptors, and management of type registries
    for serialization support.

    Features
    --------
    - **Automatic registration**: All subclasses are registered by qualified name
    - **Property discovery**: Automatically collects SerializableProperty descriptors
    - **Type registries**: Manages primitive types and dataset types
    - **Subscript access**: Access registered classes via Serializable[name]
    - **Contains protocol**: Check class registration via 'name in Serializable'

    Class Attributes (on Serializable base)
    ----------------------------------------
    _subclasses : dict[str, Type[Serializable]]
        Registry of all Serializable subclasses by qualified name.
    _primitive_types : set[type]
        Registry of types that can be serialized as-is.
    _dataset_types : dict[type | str, dict[str, Any]]
        Registry of types requiring special disassemble/assemble handling.

    Instance Attributes (on subclasses)
    -----------------------------------
    _serializable_properties : dict[str, SerializableProperty]
        Collected SerializableProperty descriptors from the class and its bases.

    Examples
    --------
    Creating a Serializable class automatically registers it:

    >>> class MyClass(Serializable):
    ...     value = SerializableProperty(default=0)
    ...
    >>> get_full_qualified_name(MyClass) in Serializable
    True

    Access classes by qualified name:

    >>> qualname = get_full_qualified_name(MyClass)
    >>> retrieved = Serializable[qualname]
    >>> retrieved is MyClass
    True

    Property discovery happens automatically:

    >>> 'value' in MyClass._serializable_properties
    True

    Register custom primitive types:

    >>> class CustomType:
    ...     pass
    >>> Serializable.register_primitive_type(CustomType)
    >>> Serializable.is_primitive_type(CustomType)
    True

    Notes
    -----
    This metaclass extends ABCMeta to allow Serializable classes to also be
    abstract base classes if needed.

    The metaclass automatically:
    1. Registers each subclass in the global registry
    2. Walks the MRO to collect all SerializableProperty descriptors
    3. Creates the _serializable_properties dict on each subclass

    See Also
    --------
    Serializable : The base class using this metaclass
    SerializableProperty : Property descriptor for serializable attributes
    """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __new__(mcs,
                name: str,
                bases: tuple[type, ...],
                namespace: dict[str, Any],
                **kwargs: Any) -> Type[Serializable]:
        """
        Create a new Serializable class with automatic registration.

        This method is called when a new class is created. It handles:
        - Creating class-level registries on the Serializable base class
        - Registering subclasses in the global registry
        - Collecting SerializableProperty descriptors from the MRO

        Parameters
        ----------
        name : str
            The name of the class being created.
        bases : tuple[type, ...]
            The base classes.
        namespace : dict[str, Any]
            The class namespace (attributes and methods).
        **kwargs : Any
            Additional keyword arguments passed to the metaclass.

        Returns
        -------
        Type[Serializable]
            The newly created class.

        Notes
        -----
        For the Serializable base class itself, this creates the registries.
        For subclasses, this registers them and collects their properties.
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # ---------- ---------- ---------- ---------- ---------- ----------
        if name == 'Serializable':
            cls._subclasses: dict[str, Type[Serializable]] = {}
            cls._primitive_types: set[type] = set()
            cls._dataset_types: dict[type|str, dict[str, Any]] = {}

        else:
            qualname: str = get_full_qualified_name(cls)
            Serializable._subclasses[qualname] = cls

            cls._serializable_properties = {}

            for base in cls.__mro__:

                if base is object:
                    continue  # just for not wasting time

                for attr_name, attr_value in base.__dict__.items():

                    if isinstance(attr_value, SerializableProperty):
                        cls._serializable_properties[attr_name] = attr_value

        # ---------- ---------- ---------- ---------- ---------- ----------
        return cls

    def __getitem__(cls, qualname: str) -> Type[Serializable]:
        """
        Get a registered Serializable subclass by qualified name.

        Parameters
        ----------
        qualname : str
            The fully qualified name of the class (e.g., 'module.ClassName').

        Returns
        -------
        Type[Serializable]
            The registered class.

        Raises
        ------
        KeyError
            If the qualified name is not registered or if called on a subclass
            instead of the Serializable base.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     pass
        >>> qualname = get_full_qualified_name(MyClass)
        >>> retrieved = Serializable[qualname]
        >>> retrieved is MyClass
        True

        Notes
        -----
        This method only works on the Serializable base class itself, not on
        subclasses. Attempting to use subscript notation on a subclass will
        raise a KeyError.
        """
        if cls is Serializable:
            return Serializable._subclasses[qualname]

        raise KeyError(f'Class {cls.__name__} is not subscriptable')

    def __contains__(cls, subclass: str | type) -> bool:
        """
        Check if a class or qualified name is registered.

        Parameters
        ----------
        subclass : str | type
            Either a qualified name string or a class type to check.

        Returns
        -------
        bool
            True if the class/name is registered, False otherwise.

        Raises
        ------
        TypeError
            If subclass is neither a string nor a type.
        NotImplementedError
            If called on a subclass instead of Serializable base.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     pass
        >>> get_full_qualified_name(MyClass) in Serializable
        True
        >>> MyClass in Serializable
        True
        >>> 'nonexistent.Class' in Serializable
        False
        """
        if cls is Serializable:
            if isinstance(subclass, str):
                return subclass in cls._subclasses

            if isinstance(subclass, type):
                return subclass in cls._subclasses.values()

            raise TypeError(f'Expected the class full qualified name or the class itself')

        raise NotImplementedError()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    ...

    # ========== ========== ========== ========== ========== public methods
    def register_primitive_type(cls, primitive_type: type) -> None:
        """
        Register a type as a primitive that can be serialized as-is.

        Primitive types are serialized without any transformation - they pass
        through the serialization process unchanged. Examples include str,
        int, float, Path, etc.

        Parameters
        ----------
        primitive_type : type
            The type to register as primitive.

        Raises
        ------
        TypeError
            If attempting to register list, dict, tuple, or Serializable as
            primitive types (these are handled specially).

        Examples
        --------
        >>> class CustomPrimitive:
        ...     pass
        >>> Serializable.register_primitive_type(CustomPrimitive)
        >>> Serializable.is_primitive_type(CustomPrimitive)
        True

        Notes
        -----
        By default, the following types are registered as primitives:
        - str
        - numbers.Number (int, float, complex, etc.)
        - pathlib.Path

        Collections (list, dict, tuple, set) are handled recursively and
        cannot be registered as primitives.
        """
        if issubclass(primitive_type, (list, dict, tuple, Serializable)):
            raise TypeError(f'Cannot register {primitive_type} as primitive type')

        Serializable._primitive_types.add(primitive_type)

    def remove_primitive_type(cls, primitive_type: type) -> None:
        """
        Remove a type from the primitive types registry.

        Parameters
        ----------
        primitive_type : type
            The type to remove from the registry.

        Notes
        -----
        If the type is not in the registry, this method does nothing (no error).

        Examples
        --------
        >>> class CustomType:
        ...     pass
        >>> Serializable.register_primitive_type(CustomType)
        >>> Serializable.remove_primitive_type(CustomType)
        >>> Serializable.is_primitive_type(CustomType)
        False
        """
        Serializable._primitive_types.discard(primitive_type)

    def is_primitive_type(cls, type_: type) -> bool:
        """
        Check if a type is registered as a primitive.

        Parameters
        ----------
        type_ : type
            The type to check.

        Returns
        -------
        bool
            True if the type is registered as primitive, False otherwise.

        Examples
        --------
        >>> Serializable.is_primitive_type(str)
        True
        >>> Serializable.is_primitive_type(list)
        False
        """
        return type_ in Serializable._primitive_types

    def register_dataset_type(cls,
                              dataset_type: type,
                              disassemble: Callable[[Any], tuple[NDArray, dict]],
                              assemble: Callable[[NDArray, dict], Any]) -> None:
        """
        Register a type that requires special handling for serialization.

        Dataset types are types that need to be converted to/from numpy arrays
        for storage (e.g., in HDF5 datasets). The disassemble function converts
        the object to an array and metadata dict, while assemble reconstructs
        the object.

        Parameters
        ----------
        dataset_type : type
            The type to register.
        disassemble : Callable[[Any], tuple[NDArray, dict]]
            Function that converts an object to (array, attributes_dict).
            The array will be stored as an HDF5 dataset, and attributes as
            HDF5 attributes or group metadata.
        assemble : Callable[[NDArray, dict], Any]
            Function that reconstructs the object from array and attributes.

        Examples
        --------
        Register a custom array-like type:

        >>> class CustomArray:
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        >>> def disassemble(obj):
        ...     return np.array(obj.data), {'shape': len(obj.data)}
        ...
        >>> def assemble(arr, attrs):
        ...     return CustomArray(arr.tolist())
        ...
        >>> Serializable.register_dataset_type(CustomArray, disassemble, assemble)

        Notes
        -----
        Registering a dataset type also automatically registers it as a
        primitive type. The type is registered both by type object and by
        qualified name string for deserialization.

        Built-in registered dataset types:
        - numpy.ndarray
        - pandas.Timestamp
        - pandas.DatetimeIndex

        See Also
        --------
        remove_dataset_type : Remove a dataset type registration
        is_dataset_type : Check if a type is registered as a dataset type
        """
        dataset_type_name = get_full_qualified_name(dataset_type)

        process = {
            'disassemble': disassemble,
            'assemble': assemble,
        }

        Serializable._dataset_types[dataset_type] = process
        Serializable._dataset_types[dataset_type_name] = process
        Serializable.register_primitive_type(dataset_type)

    def remove_dataset_type(cls, dataset_type: type) -> None:
        """
        Remove a dataset type registration.

        Parameters
        ----------
        dataset_type : type
            The type to remove from the registry.

        Notes
        -----
        This also removes the type from the primitive types registry.
        If the type is not registered, this method does nothing.

        Examples
        --------
        >>> class CustomDataset:
        ...     pass
        >>> Serializable.register_dataset_type(
        ...     CustomDataset,
        ...     lambda obj: (np.array([]), {}),
        ...     lambda arr, attrs: CustomDataset()
        ... )
        >>> Serializable.remove_dataset_type(CustomDataset)
        >>> Serializable.is_dataset_type(CustomDataset)
        False
        """
        Serializable.remove_primitive_type(dataset_type)

        if dataset_type in Serializable._dataset_types:

            dataset_type_name = get_full_qualified_name(dataset_type)

            Serializable._dataset_types.pop(dataset_type)
            Serializable._dataset_types.pop(dataset_type_name)

    def is_dataset_type(cls, type_: type) -> bool:
        """
        Check if a type is registered as a dataset type.

        Parameters
        ----------
        type_ : type
            The type to check.

        Returns
        -------
        bool
            True if the type is registered as a dataset type, False otherwise.

        Examples
        --------
        >>> Serializable.is_dataset_type(np.ndarray)
        True
        >>> Serializable.is_dataset_type(str)
        False
        """
        return type_ in Serializable._dataset_types

    # ---------- ---------- ---------- ---------- ---------- properties
    @property
    def serializable_types(cls) -> list[Type[Serializable]]:
        """
        Get a list of all registered Serializable subclasses.

        Returns
        -------
        list[Type[Serializable]]
            List of all registered Serializable classes.

        Examples
        --------
        >>> class MyClass1(Serializable):
        ...     pass
        >>> class MyClass2(Serializable):
        ...     pass
        >>> types = Serializable.serializable_types
        >>> MyClass1 in types
        True
        >>> MyClass2 in types
        True
        """
        return list(Serializable._subclasses.values())

    @property
    def primitive_types(cls) -> list[type]:
        """
        Get a list of all registered primitive types.

        Returns
        -------
        list[type]
            List of all registered primitive types.

        Examples
        --------
        >>> types = Serializable.primitive_types
        >>> str in types
        True
        >>> Number in types
        True
        """
        return list(Serializable._primitive_types)

    @property
    def dataset_types(cls) -> list[type]:
        """
        Get a list of all registered dataset types.

        Returns
        -------
        list[type]
            List of all registered dataset types (type objects only, not strings).

        Examples
        --------
        >>> types = Serializable.dataset_types
        >>> np.ndarray in types
        True
        >>> all(isinstance(t, type) for t in types)
        True

        Notes
        -----
        This property filters out string keys (qualified names) and returns
        only the actual type objects.
        """
        return [_type for _type in Serializable._dataset_types.keys() if not isinstance(_type, str)]

    @property
    def serializable_properties(cls) -> dict[str, SerializableProperty]:
        """
        Get all SerializableProperty descriptors on this class.

        Returns
        -------
        dict[str, SerializableProperty]
            Dictionary mapping property names to descriptors.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     prop1 = SerializableProperty(default=0)
        ...     prop2 = SerializableProperty(default="")
        ...
        >>> props = MyClass.serializable_properties
        >>> 'prop1' in props
        True
        >>> 'prop2' in props
        True

        Notes
        -----
        This includes properties inherited from base classes.
        Returns a copy of the internal dictionary to prevent modification.
        """
        return {**cls._serializable_properties}

    @property
    def copiable_properties(cls) -> dict[str, SerializableProperty]:
        """
        Get SerializableProperty descriptors marked as copiable.

        Returns
        -------
        dict[str, SerializableProperty]
            Dictionary mapping property names to copiable descriptors.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     data = SerializableProperty(default=0, copiable=True)
        ...     cache = SerializableProperty(default=0, copiable=False)
        ...
        >>> props = MyClass.copiable_properties
        >>> 'data' in props
        True
        >>> 'cache' in props
        False

        Notes
        -----
        Copiable properties are those that should be persisted to disk.
        Non-copiable properties (like cached values or temporary state) are
        excluded from serialization when is_copy=True.

        See Also
        --------
        Serializable.serialize : Uses copiable flag when is_copy=True
        """
        return {k: v for k, v in cls.serializable_properties.items() if v.copiable}


class Serializable(metaclass=SerializableMetatype):
    """
    Base class for objects that can be serialized to/from dictionaries.

    Serializable provides a framework for converting Python objects to dictionary
    structures suitable for persistence in HDF5 files or other storage backends.
    It handles nested objects, collections, primitive types, and special dataset
    types like NumPy arrays and Pandas timestamps.

    Features
    --------
    - **Recursive serialization**: Handles nested Serializable objects and collections
    - **Type preservation**: Stores class information for accurate deserialization
    - **Flexible construction**: Initialize from kwargs or copy from another instance
    - **Copy functionality**: Create independent copies with selective property copying
    - **Equality comparison**: Compare objects based on their serializable state
    - **Extensible type system**: Register custom primitive and dataset types

    Parameters
    ----------
    *args : tuple
        Variable positional arguments. Supports:
        - No args: Initialize from kwargs
        - One Serializable instance: Copy constructor
    **kwargs : dict
        Keyword arguments mapping property names to values.

    Raises
    ------
    ValueError
        If args don't match any supported initialization signature, or if
        kwargs contain unknown property names.

    Examples
    --------
    Define a Serializable class:

    >>> class Experiment(Serializable):
    ...     name = SerializableProperty(default="")
    ...     temperature = SerializableProperty(default=293.15)
    ...     data = SerializableProperty(default=None)
    ...
    >>> exp = Experiment(name="Test", temperature=373.15)
    >>> exp.name
    'Test'

    Serialize to a dictionary:

    >>> data = Serializable.serialize(exp)
    >>> data['__class__']  # doctest: +SKIP
    '__main__.Experiment'
    >>> data['name']
    'Test'

    Deserialize back to an object:

    >>> restored = Serializable.deserialize(data)
    >>> restored.name
    'Test'
    >>> restored.temperature
    373.15

    Copy an object:

    >>> copy = exp.copy()
    >>> copy is exp
    False
    >>> copy == exp
    True

    Nested objects work automatically:

    >>> class Trial(Serializable):
    ...     trial_num = SerializableProperty(default=0)
    ...     experiment = SerializableProperty(default=None)
    ...
    >>> trial = Trial(trial_num=1, experiment=exp)
    >>> data = Serializable.serialize(trial)
    >>> restored = Serializable.deserialize(data)
    >>> restored.experiment.name
    'Test'

    Collections are handled recursively:

    >>> exp_list = [Experiment(name=f"Exp{i}") for i in range(3)]
    >>> serialized = Serializable.serialize(exp_list)
    >>> restored = Serializable.deserialize(serialized)
    >>> len(restored)
    3

    Notes
    -----
    All Serializable properties must be SerializableProperty descriptors.
    Regular attributes are not included in serialization.

    The serialization format uses dictionaries with a special '__class__' key
    to store the fully qualified class name. This allows accurate reconstruction
    of the original object type.

    When using is_copy=True in serialize(), only properties with copiable=True
    are included. This is useful for persistence - you can mark cached or
    temporary properties as non-copiable.

    See Also
    --------
    SerializableProperty : Descriptor for serializable properties
    SerializableMetatype : Metaclass that enables serialization
    """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Serializable object.

        Supports three initialization modes:
        1. From keyword arguments: `obj = MyClass(prop1=val1, prop2=val2)`
        2. Copy constructor: `obj2 = MyClass(obj1)`
        3. Internal use during deserialization

        Parameters
        ----------
        *args : tuple
            If empty, initialize from kwargs.
            If one Serializable instance, perform copy construction.
        **kwargs : dict
            Property names mapped to values.

        Raises
        ------
        ValueError
            If args don't match a supported signature, or if kwargs contain
            properties not defined on the class.

        Examples
        --------
        Initialize from kwargs:

        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> obj = MyClass(value=42)
        >>> obj.value
        42

        Copy constructor:

        >>> obj2 = MyClass(obj)
        >>> obj2.value
        42
        >>> obj2 is obj
        False
        """
        if not args:
            # then it is trying to initialize by setting the serializable properties
            self._initialize_from_data(kwargs)

        elif len(args) == 1:

            if isinstance(args[0], type(self)):
                # then it is performing a copy
                data = Serializable.serialize(args[0], is_copy=True)
                self._initialize_from_data(data)

            else:
                error = "Given args do not match any available signature for initializing the Serializable interface"
                raise ValueError(error)
        else:
            error = "Given args do not match any available signature for initializing the Serializable interface"
            raise ValueError(error)

    def __eq__(self, other):
        """
        Compare equality based on copiable properties.

        Two Serializable objects are equal if they are the same type and all
        their copiable properties have equal values.

        Parameters
        ----------
        other : Any
            The object to compare with.

        Returns
        -------
        bool
            True if objects are equal, False otherwise.

        Notes
        -----
        Only copiable properties are compared. Non-copiable properties (like
        caches or temporary state) are ignored.

        For nested Serializable objects, comparison is recursive.
        For NumPy arrays, uses numpy.all() for element-wise comparison.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> obj1 = MyClass(value=42)
        >>> obj2 = MyClass(value=42)
        >>> obj3 = MyClass(value=99)
        >>> obj1 == obj2
        True
        >>> obj1 == obj3
        False

        Different types are never equal:

        >>> class OtherClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> obj4 = OtherClass(value=42)
        >>> obj1 == obj4
        False
        """
        if type(other) is not type(self):
            return False

            # I no longer think this behavior should raise an error
            error = f"Comparison must be taken between the same type: " \
                    f"{type(other).__name__} is not {type(self).__name__}"

            raise TypeError(error)

        for key in type(self).copiable_properties:

            other_value = getattr(other, key)
            self_value = getattr(self, key)

            if isinstance(other_value, Serializable) and isinstance(self_value, Serializable):
                if not Serializable.__eq__(other_value, self_value):
                    return False
            else:
                if not numpy.all(other_value == self_value):
                    return False

        return True

    def __copy__(self):
        """
        Support for the copy module.

        Returns
        -------
        Serializable
            A copy of this object.

        Examples
        --------
        >>> import copy
        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> obj = MyClass(value=42)
        >>> copied = copy.copy(obj)
        >>> copied.value
        42
        >>> copied is obj
        False
        """
        return self.copy()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _initialize_from_data(self, data: dict[str, Any]) -> None:
        """
        Initialize object properties from a dictionary.

        This is an internal method called during construction and deserialization.
        It validates the data, removes the __class__ key if present, sets all
        properties, and ensures no unknown keys remain.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary mapping property names to values.

        Raises
        ------
        TypeError
            If data is not a dictionary.
        AssertionError
            If __class__ key is present but doesn't match this class.
        ValueError
            If data contains keys that are not defined properties.

        Notes
        -----
        This method:
        1. Validates data is a dict
        2. Checks __class__ if present
        3. Deserializes each property value
        4. Sets each property via setattr (triggers parsers, observers, etc.)
        5. Ensures all keys were consumed (no unknowns)
        """
        # data must be a dictionary
        check_types(data, dict)

        if '__class__' in data:
            assert Serializable[data.pop('__class__')] is type(self)

        # set the serialisable properties
        for key in type(self).serializable_properties:
            value = Serializable.deserialize(data.pop(key, None))
            setattr(self, key, value)

        # at this point, the dictionary should be empty
        if data:
            error = f"There is no signature with the keys {data.keys()}"
            raise ValueError(error)

    # ========== ========== ========== ========== ========== public methods
    @staticmethod
    def serialize(obj: Any, is_copy: bool = False) -> Any:
        """
        Recursively serialize an object to a dictionary structure.

        Converts Python objects to a nested dictionary structure suitable for
        JSON, HDF5 attributes, or other storage formats. Handles Serializable
        objects, collections, primitives, and dataset types.

        Parameters
        ----------
        obj : Any
            The object to serialize.
        is_copy : bool, optional
            If True, only serialize copiable properties. If False, serialize
            all properties. Default is False.

        Returns
        -------
        Any
            The serialized representation:
            - Serializable → dict with '__class__' key
            - list/tuple/set → list (recursively serialized)
            - dict → dict (recursively serialized values)
            - Primitives → unchanged
            - None → None

        Raises
        ------
        TypeError
            If the object's type is not registered as primitive, dataset, or
            Serializable.

        Examples
        --------
        Serialize a simple object:

        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> obj = MyClass(value=42)
        >>> data = Serializable.serialize(obj)
        >>> '__class__' in data
        True
        >>> data['value']
        42

        Serialize with copy mode (only copiable properties):

        >>> class MyClass2(Serializable):
        ...     important = SerializableProperty(default=0, copiable=True)
        ...     cache = SerializableProperty(default=0, copiable=False)
        ...
        >>> obj = MyClass2(important=10, cache=20)
        >>> data = Serializable.serialize(obj, is_copy=True)
        >>> 'important' in data
        True
        >>> 'cache' in data
        False

        Collections are handled recursively:

        >>> data = Serializable.serialize([1, "two", 3.0])
        >>> data
        [1, 'two', 3.0]

        Notes
        -----
        The serialization format for Serializable objects is:
        ```python
        {
            '__class__': 'module.ClassName',
            'property1': value1,
            'property2': value2,
            ...
        }
        ```

        Tuples and sets are converted to lists (Python's JSON doesn't
        distinguish these).

        See Also
        --------
        deserialize : Reconstruct objects from serialized data
        copy : Create a copy using serialize with is_copy=True
        """
        if obj is None:
            return None

        if isinstance(obj, (tuple, list, set)):
            return [Serializable.serialize(o, is_copy=is_copy) for o in obj]

        if isinstance(obj, dict):
            return {k: Serializable.serialize(v, is_copy=is_copy) for k, v in obj.items()}

        if isinstance(obj, Serializable):

            cls = type(obj)

            data = {'__class__': get_full_qualified_name(cls)}

            keys = (cls.copiable_properties if is_copy else cls.serializable_properties).keys()

            for key in keys:
                data[key] = Serializable.serialize(getattr(obj, key), is_copy=is_copy)

            return data

        if not isinstance(obj, tuple(Serializable.primitive_types)):
            error = f"No serialisation process is implemented for object of " \
                    f"type {type(obj).__name__}."
            raise TypeError(error)

        return obj

    @staticmethod
    def deserialize(data: Any) -> Any:
        """
        Recursively deserialize data to reconstruct objects.

        Converts serialized dictionary structures back into Python objects.
        Handles nested Serializable objects, collections, and primitives.

        Parameters
        ----------
        data : Any
            The serialized data to deserialize.

        Returns
        -------
        Any
            The reconstructed object:
            - dict with '__class__' → Serializable instance
            - list → list (recursively deserialized)
            - dict without '__class__' → dict (recursively deserialized)
            - Primitives → unchanged
            - None → None

        Raises
        ------
        TypeError
            If the data type is not supported for deserialization.

        Examples
        --------
        Deserialize a simple object:

        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0)
        ...
        >>> qualname = get_full_qualified_name(MyClass)
        >>> data = {'__class__': qualname, 'value': 42}
        >>> obj = Serializable.deserialize(data)
        >>> obj.value
        42

        Roundtrip serialization:

        >>> original = MyClass(value=99)
        >>> data = Serializable.serialize(original)
        >>> restored = Serializable.deserialize(data)
        >>> restored.value
        99
        >>> original == restored
        True

        Collections are handled recursively:

        >>> data = [1, 2, {'a': 3}]
        >>> result = Serializable.deserialize(data)
        >>> result
        [1, 2, {'a': 3}]

        Unknown classes create generic Serializable objects:

        >>> data = {
        ...     '__class__': 'unknown.Module.UnknownClass',
        ...     'prop1': 42,
        ...     'prop2': "test"
        ... }
        >>> obj = Serializable.deserialize(data)
        >>> obj.prop1
        42

        Notes
        -----
        If a class referenced in '__class__' is not registered (not imported),
        deserialize creates a generic Serializable subclass on-the-fly with
        the necessary properties. This allows reading data even when the
        original class definition is unavailable.

        Deserialization triggers all property parsers, observers, and
        post-initializers as if the object were being constructed normally.

        See Also
        --------
        serialize : Convert objects to serializable data
        """
        if data is None:
            return None

        if isinstance(data, (tuple, list, set)):
            return [Serializable.deserialize(d) for d in data]

        if isinstance(data, dict) and '__class__' not in data:
            return {k: Serializable.deserialize(v) for k, v in data.items()}

        if isinstance(data, dict) and '__class__' in data:
            cls_name = data.pop('__class__')

            try:
                cls = Serializable[cls_name]

            except KeyError:

                name = cls_name.split('.')[-1]

                cls = SerializableMetatype(f'_Generic{name}', (Serializable,), {})

                prop = {}

                for key, value in data.items():
                    prop[key] = SerializableProperty()
                    prop[key].__set_name__(cls, key)
                    setattr(cls, key, value)
                    cls._serializable_properties[key] = prop[key]

            return cls(**data)

        if not isinstance(data, (*tuple(Serializable.primitive_types), Serializable)):
            error = f"No serialisation method is implemented for object of " \
                    f"type {type(data)}."
            raise TypeError(error)

        return data

    def copy(self) -> Serializable:
        """
        Create an independent copy of this object.

        Uses serialization with is_copy=True, so only copiable properties
        are included in the copy.

        Returns
        -------
        Serializable
            A new instance with the same copiable property values.

        Examples
        --------
        >>> class MyClass(Serializable):
        ...     value = SerializableProperty(default=0, copiable=True)
        ...     cache = SerializableProperty(default=0, copiable=False)
        ...
        >>> obj = MyClass(value=42, cache=100)
        >>> copied = obj.copy()
        >>> copied.value
        42
        >>> copied.cache
        0
        >>> copied is obj
        False

        Notes
        -----
        The copy is created via serialization, so:
        - Only copiable properties are copied
        - Non-copiable properties use their defaults
        - Nested objects are also copied (deep copy)
        - All property initialization (parsers, observers, initializers) runs

        See Also
        --------
        __copy__ : Implements copy.copy() support
        """
        return type(self)(self)

    # ---------- ---------- ---------- ---------- ---------- properties
    ...


Serializable.register_primitive_type(str)
Serializable.register_primitive_type(Number)
Serializable.register_primitive_type(Path)

# ========== ========== ========== Register ndarray as dataset_type
def disassemble_ndarray(arr: NDArray) -> tuple[NDArray, dict[str]]:
    """
    Disassemble a NumPy array for serialization.

    Parameters
    ----------
    arr : NDArray
        The array to disassemble.

    Returns
    -------
    tuple[NDArray, dict]
        The array itself and an empty attributes dictionary.

    Notes
    -----
    NumPy arrays don't need special processing, but are registered as
    dataset types to indicate they should be stored as HDF5 datasets
    rather than attributes.
    """
    return arr, {}


def assemble_ndarray(arr: NDArray, arr_attrs: dict[str]) -> NDArray:
    """
    Assemble a NumPy array from serialized data.

    Parameters
    ----------
    arr : NDArray
        The array data.
    arr_attrs : dict
        Attributes dictionary (unused for plain arrays).

    Returns
    -------
    NDArray
        The array itself.
    """
    return arr


Serializable.register_dataset_type(numpy.ndarray,
                                   disassemble=disassemble_ndarray,
                                   assemble=assemble_ndarray)

# ========== ========== ========== Register pandas DatetimeIndex and Timestamp as dataset_type
def disassemble_pandas_time(time: pandas.DatetimeIndex | pandas.Timestamp) -> tuple[NDArray, dict[str]]:
    """
    Disassemble pandas datetime objects for serialization.

    Converts timezone-aware datetimes to UTC then removes timezone for storage
    as int64 nanoseconds. Timezone information is stored in attributes.

    Parameters
    ----------
    time : pd.DatetimeIndex | pd.Timestamp
        The datetime object to disassemble.

    Returns
    -------
    tuple[NDArray, dict]
        Array of int64 nanoseconds and dict with 'timezone' key.

    Notes
    -----
    Timezone-naive datetimes are stored directly as int64.
    Timezone-aware datetimes are converted to UTC, stored as int64,
    with the timezone string saved in attributes for reconstruction.
    """
    if time.tz is None:
        time_array = time.to_numpy().astype('datetime64[ns]').astype(numpy.int64)
        timezone = None
    else:
        time_array = time.tz_localize(None).to_numpy().astype('datetime64[ns]').astype(numpy.int64)
        timezone = str(time.tz)

    return time_array, {'timezone': timezone}


def assemble_pandas_time(time_array: NDArray, time_attrs: dict[str]) -> pandas.DatetimeIndex | pandas.Timestamp:
    """
    Assemble pandas datetime objects from serialized data.

    Reconstructs timezone-aware datetimes by first converting from int64,
    then localizing to the stored timezone.

    Parameters
    ----------
    time_array : NDArray
        Array of int64 nanosecond values.
    time_attrs : dict
        Dictionary with 'timezone' key (None or timezone string).

    Returns
    -------
    pd.DatetimeIndex | pd.Timestamp
        Reconstructed datetime object with timezone if applicable.
    """
    time = pandas.to_datetime(time_array.astype('datetime64[ns]'))
    tz = time_attrs.pop('timezone')

    if tz is not None:
        time = time.tz_localize(tz)

    return time


Serializable.register_dataset_type(pandas.Timestamp,
                                   disassemble=disassemble_pandas_time,
                                   assemble=assemble_pandas_time)

Serializable.register_dataset_type(pandas.DatetimeIndex,
                                   disassemble=disassemble_pandas_time,
                                   assemble=assemble_pandas_time)


# ========== ========== ========== ========== ========== ==========
class Persistable(Serializable):

    # ========== ========== ========== ========== ========== class attributes
    extension: str = '.hdf5'

    class ProxyDataset:

        def __init__(self, dataset: h5py.Dataset) -> None:
            self._dataset = dataset
            self._attrs = {k: Persistable._load_data_from_h5py_tree(v) for k, v in self._dataset.attrs.items()}
            self._dataset_type_name = self._attrs.pop('__dataset_type__')

        def __getitem__(self, item) -> Any:
            assemble = Serializable._dataset_types[self._dataset_type_name]['assemble']
            array = self._dataset[item]

            return assemble(array, self.attrs)

        def __setitem__(self, key, value) -> None:
            # this code block is meant to test if key is inside the dataset limits,
            # otherwise resizing it. This is actually pretty experimental, but passed
            # in the proposed tests
            if isinstance(key, slice) and key.stop >= self.shape[0]:
                self._dataset.resize(key.stop + 1, axis=0)

            elif isinstance(key, int) and key >= self.shape[0]:
                self._dataset.resize(key + 1, axis=0)

            # this code block is reliable and tested
            disassemble = Serializable._dataset_types[self._dataset_type_name]['disassemble']
            arr, attrs = disassemble(value)

            assert attrs == self.attrs

            self._dataset[key] = arr

        def append(self, value: Any) -> None:
            disassemble = Serializable._dataset_types[self._dataset_type_name]['disassemble']
            arr, attrs = disassemble(value)
            assert attrs == self.attrs

            old_size = self.shape[0]
            new_size = old_size + arr.shape[0]

            self._dataset.resize(new_size, axis=0)

            self._dataset[old_size:new_size] = arr

        @property
        def attrs(self) -> dict:
            return self._attrs.copy()

        # ---------- ---------- ---------- ---------- ---------- ----------
        @property
        def shape(self) -> tuple[int]:
            return self._dataset.shape

        @property
        def size(self) -> int:
            return self._dataset.size

        @property
        def dtype(self) -> numpy.dtype:
            return self._dataset.dtype

        @property
        def ndim(self) -> int:
            return self._dataset.ndim

        @property
        def nbytes(self) -> int:
            return self._dataset.nbytes

    # ========== ========== ========== ========== ========== special methods
    def __init__(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], (str, Path)):

            filepath = Path(args[0])
            if not kwargs:
                # then it's trying to load a file
                data = self.load_serialized_data(filepath)
                self._initialize_from_data(data)

            else:
                # then it must be trying to open the file
                self._path = filepath
                self._mode = kwargs.pop('mode')
                # refer to https://docs.h5py.org/en/stable/high/file.html#opening-creating-files
                # when writing docs

                if kwargs:
                    raise ValueError(f'Unexpected keyword arguments: {kwargs.keys()}')
        else:
            super().__init__(*args, **kwargs)

    def __enter__(self) -> Persistable:
        self.__file = h5py.File(self._path, mode=self._mode)

        data = type(self)._load_data_from_h5py_tree(self.__file['root'], use_proxy_dataset=True)

        self._initialize_from_data(data)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__file.close()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    @staticmethod
    def _save_data_in_group(key: str, value: Any, group: h5py.Group) -> None:
        if value is None:
            group.attrs[key] = 'NoneType:None'

        elif isinstance(value, Path):
            group.attrs[key] = f'Path:{str(value.absolute())}'

        elif isinstance(value, (str, Number)):
            group.attrs[key] = value

        elif isinstance(value, list):
            subgroup = group.create_group(key, track_order=True)
            subgroup.attrs['__container_type__'] = 'list'

            for idx, obj in enumerate(value):
                Persistable._save_data_in_group(str(idx), obj, subgroup)

        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            subgroup.attrs['__container_type__'] = 'dict'

            for _key, obj in value.items():
                Persistable._save_data_in_group(_key, obj, subgroup)

        elif type(value) in Serializable.dataset_types:

            disassemble = Serializable._dataset_types[type(value)]['disassemble']

            value_array, value_attrs = disassemble(value)

            if value_array.ndim > 0:
                shape = value_array.shape
                maxshape = (None, *shape[1:])
                dataset = group.create_dataset(key, data=value_array, maxshape=maxshape)
            else:  # scalar dataset cannot be extended
                dataset = group.create_dataset(key, data=value_array)

            dataset.attrs['__dataset_type__'] = get_full_qualified_name(type(value))

            for _key, obj in value_attrs.items():
                dataset.attrs[_key] = obj if obj is not None else 'NoneType:None'

        else:
            raise TypeError(f"instances of {type(value).__name__} cannot be saved in h5py.Groups")

    @staticmethod
    def _load_data_from_h5py_tree(value: Any, use_proxy_dataset: bool = False) -> Any:
        if isinstance(value, h5py.Group):

            data = {k: Persistable._load_data_from_h5py_tree(v, use_proxy_dataset=use_proxy_dataset) for k, v in value.items()}
            data.update({k: Persistable._load_data_from_h5py_tree(v, use_proxy_dataset=use_proxy_dataset) for k, v in value.attrs.items()})

            # it must have a __container_type__ attr
            container_type = data.pop('__container_type__')

            if container_type == 'list':
                return [data[key] for key in sorted(data.keys())]

            if container_type == 'dict':
                return data

            raise ValueError(f"Could not resolve __container_type__={container_type}")

        if isinstance(value, h5py.Dataset):

            if use_proxy_dataset:
                return Persistable.ProxyDataset(value)

            array = value[...]
            attrs = {k: Persistable._load_data_from_h5py_tree(v, use_proxy_dataset=True) for k, v in value.attrs.items()}

            # it must have a __dataset_type__ attr
            dataset_type_name = attrs.pop('__dataset_type__')

            assemble = Serializable._dataset_types[dataset_type_name]['assemble']

            return assemble(array, attrs)

        if isinstance(value, str):

            if value == 'NoneType:None':
                return None

            if value.startswith('Path:'):
                return Path(value.removeprefix('Path:'))

            return value  # just regular strings

        if isinstance(value, numpy.bool_):
            return bool(value)

        return value  # basically numbers

    # ========== ========== ========== ========== ========== public methods
    def save_serialized_data(self, path: Path|str, data: Any) -> None:
        with h5py.File(Path(path), 'w') as file:
            self._save_data_in_group('root', data, file)

    @classmethod
    def load_serialized_data(cls, path: Path|str) -> Any:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Path {path} does not exist")

        with h5py.File(path, 'r') as file:
            data = cls._load_data_from_h5py_tree(file['root'])

        return data

    def save(self,
             path: Path | str,
             overwrite: bool = True,
             use_default_extension: bool = True) -> None:
        # ---------- ---------- resolve path
        path = Path(path)

        if use_default_extension:
            path = path.with_suffix(type(self).extension)

        if path.is_file() and not overwrite:
            raise FileExistsError(f"Path {path} already exists")

        # ---------- ---------- save
        data = Serializable.serialize(self)
        self.save_serialized_data(path, data)

    @classmethod
    def load(cls, path: Path|str) -> Persistable:
        data = cls.load_serialized_data(path)
        return Serializable.deserialize(data)

    # ---------- ---------- ---------- ---------- ---------- properties
    ...


Serializable.register_primitive_type(Persistable.ProxyDataset)


load = Persistable.load


__all__ = [
    'SerializableProperty',
    'Serializable',
    'Persistable',
    'load',
]