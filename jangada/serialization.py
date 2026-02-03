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
    Return the fully qualified class name used by the serialization registry.

    For built-in types (module is ``builtins``), returns ``cls.__qualname__``.
    For user-defined types, returns ``"<module>.<qualname>"``.

    Parameters
    ----------
    cls : type
        The class to identify.

    Returns
    -------
    str
        Fully qualified name suitable for registry keys.

    Notes
    -----
    This function uses ``__qualname__`` (not ``__name__``) so that nested class
    names are preserved, e.g. ``Outer.Inner``. This helps prevent collisions and
    supports reconstructing nested classes when importable.
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
    Check whether an object is an instance of expected types.

    This helper standardizes runtime type validation and error messages.

    Parameters
    ----------
    obj : object
        Value to test.
    types : type or tuple of type
        Expected type(s).
    can_be_none : bool, default False
        If True, ``None`` is accepted as valid and treated as matching.
    raise_error : bool, default True
        If True, raises TypeError when the check fails. If False, returns False
        on mismatch.

    Returns
    -------
    bool
        True if obj is an instance of one of the expected types (or None, when
        permitted), False otherwise.

    Raises
    ------
    TypeError
        If ``raise_error`` is True and the check fails.

    Examples
    --------
    >>> check_types(1, int)
    True
    >>> check_types(None, int, can_be_none=True)
    True
    >>> check_types("x", int, raise_error=False)
    False
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
    Metaclass implementing the global serialization registry.

    Any subclass of ``Serializable`` is automatically registered by fully
    qualified name. Additionally, this metaclass:

    - Collects ``SerializableProperty`` descriptors from the full MRO into a
      per-class mapping (``_serializable_properties``).
    - Maintains global registries on the ``Serializable`` root class:

      * ``_subclasses``: mapping from fully qualified name to class
      * ``_primitive_types``: set of types allowed as raw serialized values
      * ``_dataset_types``: mapping from dataset type and dataset type name
        to the disassemble/assemble process

    Notes
    -----
    Registry operations are global, not per-subclass: registering a primitive
    or dataset type affects all serialization across the process.

    The metaclass supports:
    - lookup: ``Serializable[qualname]``
    - membership: ``qualname in Serializable`` or ``cls in Serializable``

    See Also
    --------
    Serializable
    Persistable
    """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __new__(mcs,
                name: str,
                bases: tuple[type, ...],
                namespace: dict[str, Any],
                **kwargs: Any) -> Type[Serializable]:

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
        Resolve a registered Serializable subclass by fully qualified name.

        Parameters
        ----------
        qualname : str
            Fully qualified class name as returned by ``get_full_qualified_name``.

        Returns
        -------
        type
            Registered subclass.

        Raises
        ------
        KeyError
            If the class is not registered.
        """
        if cls is Serializable:
            return Serializable._subclasses[qualname]

        raise KeyError(f'Class {cls.__name__} is not subscriptable')

    def __contains__(cls, subclass: str | type) -> bool:
        """
        Membership test for the Serializable registry.

        Parameters
        ----------
        subclass : str or type
            Either a fully qualified class name or a class object.

        Returns
        -------
        bool
            True if the class is registered.

        Raises
        ------
        TypeError
            If input is not a str or type.

        Notes
        -----
        This operation is only defined on the ``Serializable`` base class.
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
        Register a primitive type.

        Primitive types are allowed to pass through in-memory serialization
        unchanged. They must not be containers (list/dict/tuple) or Serializable
        subclasses (which already have explicit serialization semantics).

        Parameters
        ----------
        primitive_type : type
            Type to register.

        Raises
        ------
        TypeError
            If the type is not eligible to be registered as primitive.

        Notes
        -----
        Persistence semantics for a primitive type depend on ``Persistable``.
        For example, ``Path`` is registered as a primitive here but is persisted
        via a dedicated string encoding in HDF5 attributes.
        """
        if issubclass(primitive_type, (list, dict, tuple, Serializable)):
            raise TypeError(f'Cannot register {primitive_type} as primitive type')

        Serializable._primitive_types.add(primitive_type)

    def remove_primitive_type(cls, primitive_type: type) -> None:
        """
        Remove a primitive type from the registry.

        Parameters
        ----------
        primitive_type : type
            Type to remove.

        Notes
        -----
        This operation is idempotent: removing a non-registered type does not
        raise.
        """
        Serializable._primitive_types.discard(primitive_type)

    def is_primitive_type(cls, type_: type) -> bool:
        """
        Return True if a type is registered as primitive.

        Parameters
        ----------
        type_ : type

        Returns
        -------
        bool
        """
        return type_ in Serializable._primitive_types

    def register_dataset_type(cls,
                              dataset_type: type,
                              disassemble: Callable[[Any], tuple[NDArray, dict]],
                              assemble: Callable[[NDArray, dict], Any]) -> None:
        """
        Register a dataset-backed type.

        Dataset types are treated as primitives during in-memory serialization,
        but when persisted via ``Persistable``, they are stored as HDF5 datasets.

        Parameters
        ----------
        dataset_type : type
            Type to register.
        disassemble : callable
            Function with signature ``disassemble(obj) -> (array, attrs)`` where
            ``array`` is a NumPy ndarray and ``attrs`` is a dict of metadata to
            store in dataset attributes.
        assemble : callable
            Function with signature ``assemble(array, attrs) -> obj`` used to
            reconstruct objects from dataset storage.

        Notes
        -----
        Registration stores the process under both:
        - the type object (dataset_type), and
        - its fully qualified name (string)

        The dataset type is also registered as a primitive type to allow it to
        pass through in-memory serialization.
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
        Remove a dataset-backed type from the registry.

        Parameters
        ----------
        dataset_type : type

        Notes
        -----
        This also removes the corresponding primitive type registration.
        """
        Serializable.remove_primitive_type(dataset_type)

        if dataset_type in Serializable._dataset_types:

            dataset_type_name = get_full_qualified_name(dataset_type)

            Serializable._dataset_types.pop(dataset_type)
            Serializable._dataset_types.pop(dataset_type_name)

    def is_dataset_type(cls, type_: type) -> bool:
        """
        Return True if a type is registered as dataset-backed.

        Parameters
        ----------
        type_ : type

        Returns
        -------
        bool
        """
        return type_ in Serializable._dataset_types

    # ---------- ---------- ---------- ---------- ---------- properties
    @property
    def serializable_types(cls) -> list[Type[Serializable]]:
        """
        list of Serializable subclasses currently registered.
        """
        return list(Serializable._subclasses.values())

    @property
    def primitive_types(cls) -> list[type]:
        """
        list of primitive types currently registered.
        """
        return list(Serializable._primitive_types)

    @property
    def dataset_types(cls) -> list[type]:
        """
        list of dataset-backed types currently registered.
        """
        return [_type for _type in Serializable._dataset_types.keys() if not isinstance(_type, str)]

    @property
    def serializable_properties(cls) -> dict[str, SerializableProperty]:
        """
        dict[str, SerializableProperty]
            Copy of the serialization schema mapping for this class, including
            inherited properties discovered across the full MRO.
        """
        return {**cls._serializable_properties}

    @property
    def copiable_properties(cls) -> dict[str, SerializableProperty]:
        """
        dict[str, SerializableProperty]
            Subset of ``serializable_properties`` containing only properties
            with ``copiable=True``.
        """
        return {k: v for k, v in cls.serializable_properties.items() if v.copiable}


class Serializable(metaclass=SerializableMetatype):
    """
    Base class for registry-backed structured serialization.

    Serializable defines:
    - A declarative schema via ``SerializableProperty`` descriptors.
    - A global registry that maps fully qualified names to classes.
    - An in-memory serialization format (dict/list/scalars) with a class
      envelope key ``"__class__"``.
    - Copy and equality semantics based on ``copiable`` properties.

    Construction
    ------------
    Serializable(**kwargs)
        Initialize an instance by setting serializable properties. Keys must
        match the class schema (including inherited properties). Extra keys
        raise ValueError.

    Serializable(other)
        Copy construction: if a single positional argument is provided and it
        is an instance of the same type, initializes from the serialized copy
        representation (copiable properties only).

    Equality
    --------
    ``a == b`` is defined only for objects of the same type and compares all
    copiable properties. For values that support NumPy broadcasting, equality
    uses ``numpy.all(other_value == self_value)``.

    Notes
    -----
    In-memory serialization is strict: if an object is not:
    - None,
    - a list/tuple,
    - a dict,
    - a Serializable, or
    - an instance of a registered primitive type,
    then serialization raises TypeError.

    See Also
    --------
    Persistable
    SerializableProperty
    """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __init__(self, *args, **kwargs) -> None:

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
        Compare two Serializable objects for equality.

        Parameters
        ----------
        other : object
            Object to compare.

        Returns
        -------
        bool
            True if all copiable properties are equal.

        Raises
        ------
        TypeError
            If ``other`` is not the same concrete type as ``self``.
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
        return self.copy()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _initialize_from_data(self, data: dict[str, Any]) -> None:
        """
        Initialize an instance from a serialized mapping.

        Parameters
        ----------
        data : dict
            Mapping containing serialized values. If the mapping contains a
            ``"__class__"`` key, it must resolve to the current type.

        Raises
        ------
        TypeError
            If data is not a dict.
        ValueError
            If extra keys remain after consuming the class schema.
        AssertionError
            If ``"__class__"`` exists and does not match this type.
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
        Serialize an object into the module's Python-native representation.

        Parameters
        ----------
        obj : object
            Object to serialize.
        is_copy : bool, default False
            If True, uses the object's copy schema: only copiable properties of
            Serializable instances are serialized. This is used internally by
            copy construction and ``copy()``.

        Returns
        -------
        object
            Serialized representation. For Serializable instances, returns a
            dict containing ``"__class__"`` and serialized properties.

        Raises
        ------
        TypeError
            If no serialization process exists for the object type.

        Examples
        --------
        >>> class A(Serializable):
        ...     @serializable_property(default=1)
        ...     def x(self): ...
        ...
        >>> a = A(x=2)
        >>> Serializable.serialize(a)["__class__"]  # doctest: +SKIP
        '...A'
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
        Deserialize an object from the module's serialized representation.

        Parameters
        ----------
        data : object
            Serialized data produced by ``serialize`` or loaded via
            ``Persistable``.

        Returns
        -------
        object
            Reconstructed object graph.

        Raises
        ------
        TypeError
            If no deserialization process exists for the data.

        Notes
        -----
        If ``data`` is a dict containing a ``"__class__"`` key, this function
        attempts to resolve the class name using the registry. If not found,
        it constructs a synthetic class ``_Generic<ClassName>`` and installs
        placeholder SerializableProperties for each serialized key.

        This behavior is intended for forward compatibility. Code consuming
        the object may inspect ``type(obj).__name__.startswith("_Generic")``
        if it needs to detect unknown types.
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
        Create a logical copy of this object.

        Returns
        -------
        Serializable
            New instance of the same type initialized using copy semantics
            (copiable properties only).
        """
        return type(self)(self)

    # ---------- ---------- ---------- ---------- ---------- properties
    ...


Serializable.register_primitive_type(str)
Serializable.register_primitive_type(Number)
Serializable.register_primitive_type(Path)

# ========== ========== ========== Register ndarray as dataset_type
def disassemble_ndarray(arr: NDArray) -> tuple[NDArray, dict[str]]:
    return arr, {}


def assemble_ndarray(arr: NDArray, arr_attrs: dict[str]) -> NDArray:
    return arr


Serializable.register_dataset_type(numpy.ndarray,
                                   disassemble=disassemble_ndarray,
                                   assemble=assemble_ndarray)

# ========== ========== ========== Register pandas DatetimeIndex and Timestamp as dataset_type
def disassemble_pandas_time(time: pandas.DatetimeIndex | pandas.Timestamp) -> tuple[NDArray, dict[str]]:

    if time.tz is None:
        time_array = time.to_numpy().astype(numpy.int64)
        timezone = None

    else:
        time_array = time.tz_localize(None).to_numpy().astype(numpy.int64)
        timezone = str(time.tz)

    return time_array, {'timezone': timezone}


def assemble_pandas_time(time_array: NDArray, time_attrs: dict[str]) -> pandas.DatetimeIndex | pandas.Timestamp:
    time = pandas.to_datetime(time_array)
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
    """
    Serializable that can be persisted to disk using HDF5.

    Persistable extends Serializable with:
    - HDF5 save/load methods,
    - a tree encoding for lists/dicts/scalars/paths/None,
    - dataset storage for registered dataset-backed types, and
    - lazy dataset access through ProxyDataset when used as a context manager.

    File extension
    --------------
    The default file extension is stored in the class attribute ``extension``.
    When ``save(..., use_default_extension=True)`` is used, the given path is
    rewritten with this suffix.

    HDF5 encoding rules
    -------------------
    Values are encoded into groups as follows:

    - None:
      Stored as attribute value "NoneType:None".
    - Path:
      Stored as attribute "Path:<absolute path>".
    - str, Number:
      Stored directly as an attribute.
    - list:
      Stored as a subgroup with ``__container_type__ = "list"`` and items
      stored under numeric string keys ("0", "1", ...).
    - dict:
      Stored as a subgroup with ``__container_type__ = "dict"`` and items stored
      under their keys.
    - dataset-backed types:
      Stored as HDF5 datasets with metadata attributes including:
      ``__dataset_type__ = "<qualified type name>"``.

    Context manager behavior
    ------------------------
    If constructed as ``Persistable(path, mode="r")`` (or similar) and used as
    ``with ... as obj:``, the instance loads its metadata immediately, but
    dataset values are replaced with ``ProxyDataset`` objects that fetch and
    assemble on access.

    See Also
    --------
    Serializable.register_dataset_type
    Persistable.ProxyDataset
    """

    # ========== ========== ========== ========== ========== class attributes
    extension: str = '.hdf5'

    class ProxyDataset:
        """
        Lazy wrapper for a persisted dataset.

        ProxyDataset provides indexed access to an underlying ``h5py.Dataset``.
        When an element or slice is read, the dataset's registered ``assemble``
        function is applied to return a high-level object. When writing, the
        registered ``disassemble`` function is applied.

        This is primarily intended to avoid loading large arrays eagerly.

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset handle.

        Notes
        -----
        The dataset must include a ``__dataset_type__`` attribute identifying
        the dataset-backed type. The remaining dataset attributes are treated
        as assembly metadata.

        ProxyDataset enforces that writes are compatible with existing metadata
        by asserting that the produced attrs dict matches the dataset's attrs.
        """

        def __init__(self, dataset: h5py.Dataset) -> None:
            self._dataset = dataset
            self._attrs = {k: Persistable._load_data_from_h5py_tree(v) for k, v in self._dataset.attrs.items()}
            self._dataset_type_name = self._attrs.pop('__dataset_type__')

        def __getitem__(self, item) -> Any:
            """
            Retrieve item(s) from the dataset and assemble them.

            Parameters
            ----------
            item : int or slice or tuple
                Index/slice passed directly to the underlying dataset.

            Returns
            -------
            object
                Result of applying the registered ``assemble`` function to the
                selected ndarray and the dataset metadata.

            Notes
            -----
            This method reads only the portion of the dataset addressed by
            ``item`` and assembles the corresponding object.
            """
            assemble = Serializable._dataset_types[self._dataset_type_name]['assemble']
            array = self._dataset[item]

            return assemble(array, self.attrs)

        def __setitem__(self, key, value) -> None:
            """
            Assign item(s) in the dataset from a high-level object.

            Parameters
            ----------
            key : int or slice or tuple
                Index/slice passed directly to the underlying dataset.
            value : object
                High-level object compatible with the dataset type.

            Notes
            -----
            Resizing behavior:
            If key addresses positions beyond the current size on axis 0, this
            method attempts to resize the dataset along axis 0 before writing.
            This behavior is marked experimental in the implementation, but is
            supported by tests.

            Metadata behavior:
            The disassembled attrs must match the dataset attrs (excluding the
            internal dataset type marker).
            """
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
            """
            Append value to the dataset along axis 0.

            Parameters
            ----------
            value : object
                Value to append. Must disassemble into an ndarray whose first
                dimension corresponds to append length (typically 1 for scalar-
                like items or N for batch append).

            Notes
            -----
            The underlying dataset is resized on axis 0 and the new block is
            assigned in one operation.
            """
            disassemble = Serializable._dataset_types[self._dataset_type_name]['disassemble']
            arr, attrs = disassemble(value)
            assert attrs == self.attrs

            old_size = self.shape[0]
            new_size = old_size + arr.shape[0]

            self._dataset.resize(new_size, axis=0)

            self._dataset[old_size:new_size] = arr

        @property
        def attrs(self) -> dict:
            """
            dict
                A copy of the dataset metadata attributes used for assembly.
            """
            return self._attrs.copy()

        # ---------- ---------- ---------- ---------- ---------- ----------
        @property
        def shape(self) -> tuple[int]:
            """
            tuple of int
                Dataset shape.
            """
            return self._dataset.shape

        @property
        def size(self) -> int:
            """
            int
                Total number of elements.
            """
            return self._dataset.size

        @property
        def dtype(self) -> numpy.dtype:
            """
            numpy.dtype
                Dataset dtype.
            """
            return self._dataset.dtype

        @property
        def ndim(self) -> int:
            """
            int
                Number of dataset dimensions.
            """
            return self._dataset.ndim

        @property
        def nbytes(self) -> int:
            """
            int
                Total bytes consumed by the dataset payload (not counting
                metadata overhead).
            """
            return self._dataset.nbytes

    # ========== ========== ========== ========== ========== special methods
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Persistable instance.

        Construction patterns
        ---------------------
        Persistable(**kwargs)
            Regular Serializable initialization.

        Persistable(path)
            If a path is provided as the first positional argument and no
            keyword arguments are supplied, the instance loads data from disk
            and initializes itself immediately.

        Persistable(path, mode=...)
            If a path is provided and keyword arguments are present, the
            instance is prepared for context manager usage. The file is opened
            in ``__enter__`` and closed in ``__exit__``.

        Parameters
        ----------
        *args
            Either standard Serializable args or a single path-like string/Path.
        **kwargs
            If using context manager mode, must include ``mode`` (as accepted by
            ``h5py.File``).

        Raises
        ------
        ValueError
            If unexpected keyword arguments remain after consuming ``mode``.
        """
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
        """
        Enter context manager: open the HDF5 file and load metadata.

        Returns
        -------
        Persistable
            The initialized object.

        Notes
        -----
        When entering context manager mode, datasets are loaded as ProxyDataset
        objects (lazy). Non-dataset data is loaded eagerly as normal Python
        objects.
        """
        self.__file = h5py.File(self._path, mode=self._mode)

        data = type(self)._load_data_from_h5py_tree(self.__file['root'], use_proxy_dataset=True)

        self._initialize_from_data(data)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager: close the HDF5 file.

        Parameters
        ----------
        exc_type, exc_val, exc_tb
            Exception information as provided by the context manager protocol.

        Notes
        -----
        This method always closes the file handle. Exceptions are not suppressed.
        """
        self.__file.close()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    @staticmethod
    def _save_data_in_group(key: str, value: Any, group: h5py.Group) -> None:
        """
        Save a serialized value under a given key inside an HDF5 group.

        Parameters
        ----------
        key : str
            Name to store the value under.
        value : object
            Value to store. Must be one of the supported types described in the
            Persistable class docstring.
        group : h5py.Group
            Group where the value will be stored.

        Raises
        ------
        TypeError
            If the value type cannot be persisted.

        Notes
        -----
        This function implements the recursive encoding for:
        - None and Path markers stored as strings in attributes,
        - scalars stored directly as attributes,
        - lists/dicts stored as subgroups,
        - dataset-backed types stored as datasets.
        """
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
        """
        Load a value from an HDF5 group/dataset/attribute recursively.

        Parameters
        ----------
        value : object
            An h5py.Group, h5py.Dataset, or attribute value.
        use_proxy_dataset : bool, default False
            If True, datasets are wrapped in ProxyDataset objects instead of
            being assembled eagerly.

        Returns
        -------
        object
            The reconstructed Python object.

        Raises
        ------
        ValueError
            If a container group has an unknown ``__container_type__``.
        KeyError
            If a dataset references an unknown dataset type in the registry.

        Notes
        -----
        Special markers:
        - "NoneType:None" maps to None.
        - "Path:<...>" maps to pathlib.Path.
        """
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
        """
        Persist a serialized representation to an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        data : object
            Serialized data tree (as produced by ``Serializable.serialize``).

        Notes
        -----
        This method does not enforce a file extension. Extension handling is
        performed by ``save``.
        """
        with h5py.File(Path(path), 'w') as file:
            self._save_data_in_group('root', data, file)

    @classmethod
    def load_serialized_data(cls, path: Path|str) -> Any:
        """
        Load serialized data from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        object
            Serialized data tree.

        Raises
        ------
        FileNotFoundError
            If the path does not exist or is not a file.
        """
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
        """
        Serialize and save this instance to an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Output path.
        overwrite : bool, default True
            If False and the file exists, raises FileExistsError.
        use_default_extension : bool, default True
            If True, rewrites the suffix of ``path`` to ``type(self).extension``.

        Raises
        ------
        FileExistsError
            If the file exists and overwrite is False.
        """
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
        """
        Load and deserialize a Persistable object from disk.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        Persistable
            The reconstructed object. The returned object is the concrete class
            encoded in the serialized representation (not necessarily ``cls``).

        Notes
        -----
        This method reads the serialized tree and then calls
        ``Serializable.deserialize`` to reconstruct the object graph.
        """
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