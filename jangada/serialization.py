#  -*- coding: utf-8 -*-
"""
Serialization and persistence framework for Jangada.

This module provides a declarative and extensible system for serializing
Python object graphs and persisting them to disk using HDF5.

The framework is composed of three layers:

- SerializableProperty: descriptor-based property schema definition.
- Serializable: in-memory serialization to Python-native data structures.
- Persistable: disk persistence with native HDF5 dataset support.

The design emphasizes explicit schemas, numerical efficiency, forward
compatibility, and minimal magic.

Author: Rafael R. L. Benevides
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
"""Represent the type of the property"""

Getter: TypeAlias = Callable[[object], T]
Setter: TypeAlias = Callable[[object, Any], None]
Deleter: TypeAlias = Callable[[object], None]
Observer: TypeAlias = Callable[[object, T, T], None]
Parser: TypeAlias = Callable[[object, Any], T]


class SerializableProperty:
    """
    Descriptor representing a serializable attribute.

    SerializableProperty extends the built-in ``property`` concept with
    features required for persistent object models, including default
    values, validation, observation hooks, immutability constraints, and
    copy semantics.

    SerializableProperty instances are automatically discovered by
    ``SerializableMetatype`` and define the serialization schema of a class.

    Parameters
    ----------
    fget : callable, optional
        Getter function with signature ``fget(instance) -> value``.
    fset : callable, optional
        Setter function with signature ``fset(instance, value)``.
    fdel : callable, optional
        Deleter function with signature ``fdel(instance)``.
    default : object or callable, optional
        Default value or default factory. If callable, must have signature
        ``default(instance) -> value``.
    parser : callable, optional
        Value parser applied before assignment. Signature:
        ``parser(instance, raw_value) -> parsed_value``.
    observer : callable, optional
        Observer callback executed after assignment. Signature:
        ``observer(instance, old_value, new_value)``.
    readonly : bool, default False
        If True, the property cannot be assigned.
    writeonce : bool, default False
        If True, the property may only be assigned once to a non-None value.
    copiable : bool, default True
        If True, the property participates in copy and equality operations.
    doc : str, optional
        Docstring for the property. If omitted, the getter docstring is used.

    Notes
    -----
    If the stored value is missing or None, the default is returned.
    Explicit assignment of None is treated as "unset".

    Setting both ``readonly`` and ``writeonce`` is not allowed.
    """

    # ========== ========== ========== ========== ========== class attributes
    __slots__ = ('fget', 'fset', 'fdel',
                 '_default', '_parser', '_observer',
                 '_writeonce', '_copiable', '_readonly',
                 'name', 'private_name', 'owner', '__doc__')

    # ========== ========== ========== ========== ========== special methods
    def __init__(self,
                 fget: Getter | None = None,
                 fset: Setter | None = None,
                 fdel: Deleter | None = None,
                 *,
                 default: T | Getter | None = None,
                 parser: Parser | None = None,
                 observer: Observer | None = None,
                 readonly: bool = False,
                 writeonce: bool = False,
                 copiable: bool = True,
                 doc: str | None = None) -> None:

        self.fget: Getter | None = fget
        self.fset: Setter | None = fset
        self.fdel: Deleter | None = fdel

        self._default: T | Getter | None = default
        self._parser: Parser | None = parser
        self._observer: Observer | None = observer

        self._readonly: bool = readonly
        self._writeonce: bool = writeonce
        self._copiable: bool = copiable

        if readonly and writeonce:
            raise ValueError("Cannot be both readonly and writeonce")

        if self._readonly:
            self.fset = None

        # Use getter docstring if not provided (which can also be None)
        self.__doc__: str | None = fget.__doc__ if doc is None and fget is not None else doc

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name: str = name
        self.owner: type = owner
        self.private_name: str = f"_serializable_property__{name}"

        if self.fget is None:
            self.fget = lambda obj: getattr(obj, self.private_name)

        if self.fset is None and not self._readonly:
            self.fset = lambda obj, value: setattr(obj, self.private_name, value)

    def __get__(self, instance: object|None, owner: type) -> T|Self:
        """
        Return the property value.

        If accessed from the class, returns the descriptor itself.
        If the stored value is None or missing, the default is returned.
        """
        if instance is None:
            # Accessing from class, return descriptor for introspection
            return self

        if self.fget is None:
            raise AttributeError(f"unreadable attribute '{self.name}'")

        try:
            value = self.fget(instance)
        except AttributeError:
            value = None

        if value is None:

            if callable(self._default):
                value = self._default(instance)

            else:
                value = self._default

        return value

    def __set__(self, instance: object, value: Any) -> None:
        """
        Assign a value to the property.

        The assignment process follows this order:
        1. Enforce read-only or write-once constraints.
        2. Apply default if value is None.
        3. Apply parser if defined.
        4. Assign the value.
        5. Notify observer if defined.
        """
        if self.fset is None:
            # No setter provided - property is read-only
            raise AttributeError(
                f"can't set attribute '{self.name}' (read-only property)"
            )

        if self._writeonce:
            # Check if already set (write-once behavior)

            try:
                current_value = self.fget(instance)
            except AttributeError:
                current_value = None

            if current_value is not None:
                raise AttributeError(
                    f"{self.name} is a write-once property and has already been set"
                )

        if value is None:

            if callable(self._default):
                value = self._default(instance)

            else:
                value = self._default

        if self._parser is not None:
            value = self._parser(instance, value)

        old_value = self.__get__(instance, self.owner)

        self.fset(instance, value)

        if self._observer is not None:
            self._observer(instance, old_value, value)

    def __delete__(self, instance: object) -> None:
        """
        Delete the property value.

        Raises
        ------
        AttributeError
            If no deleter is defined.
        """
        if self.fdel is None:
            raise AttributeError(f"can't delete attribute '{self.name}'")

        self.fdel(instance)

    # ========== ========== Descriptor protocol methods to work like @property
    def getter(self, fget: Getter) -> Self:
        """
        Return a new SerializableProperty with a replaced getter.

        Parameters
        ----------
        fget : callable
            New getter function.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            fget, self.fset, self.fdel,
            default=self._default,
            parser=self._parser,
            observer=self._observer,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def setter(self, fset: Setter) -> Self:
        """
        Return a new SerializableProperty with a replaced setter.

        Parameters
        ----------
        fset : callable
            New setter function.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            self.fget, fset, self.fdel,
            default=self._default,
            parser=self._parser,
            observer=self._observer,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def deleter(self, fdel: Deleter) -> Self:
        """
        Return a new SerializableProperty with a replaced deleter.

        Parameters
        ----------
        fdel : callable
            New deleter function.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            self.fget, self.fset, fdel,
            default=self._default,
            parser=self._parser,
            observer=self._observer,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    # ---------- ---------- and more!!
    def default(self, func: Getter) -> Self:
        """
        Return a new SerializableProperty with a replaced default factory.

        Parameters
        ----------
        func : callable
            Default factory with signature ``func(instance) -> value``.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            self.fget, self.fset, self.fdel,
            default=func,
            parser=self._parser,
            observer=self._observer,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def parser(self, func: Parser) -> Self:
        """
        Return a new SerializableProperty with a replaced parser.

        Parameters
        ----------
        func : callable
            Parser function.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            self.fget, self.fset, self.fdel,
            default=self._default,
            parser=func,
            observer=self._observer,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    def observer(self, func: Observer) -> Self:
        """
        Return a new SerializableProperty with a replaced observer.

        Parameters
        ----------
        func : callable
            Observer callback.

        Returns
        -------
        SerializableProperty
        """
        return type(self)(
            self.fget, self.fset, self.fdel,
            default=self._default,
            parser=self._parser,
            observer=func,
            readonly=self._readonly,
            writeonce=self._writeonce,
            copiable=self._copiable,
            doc=self.__doc__
        )

    @property
    def readonly(self) -> bool:
        """
        bool : Whether the property is read-only.
        """
        return self._readonly

    @property
    def writeonce(self) -> bool:
        """
        bool : Whether the property is write-once.
        """
        return self._writeonce

    @property
    def copiable(self) -> bool:
        """
        bool : Whether the property participates in copy operations.
        """
        return self._copiable


def serializable_property(
        default: T | Getter | None = None,
        readonly: bool = False,
        writeonce: bool = False,
        copiable: bool = True) -> Callable[[Getter], SerializableProperty]:
    """
    Decorator that creates a SerializableProperty from a getter function.

    This is a convenience wrapper for defining serializable attributes with
    implicit storage and default behavior.

    Parameters
    ----------
    default : object or callable, optional
        Default value or default factory.
    readonly : bool, default False
        Whether the property is read-only.
    writeonce : bool, default False
        Whether the property is write-once.
    copiable : bool, default True
        Whether the property participates in copy operations.

    Returns
    -------
    callable
        Decorator returning a SerializableProperty.
    """
    def decorator(getter: Getter) -> SerializableProperty:
        return SerializableProperty(
            fget=getter,
            default=default,
            readonly=readonly,
            writeonce=writeonce,
            copiable=copiable,
        )

    return decorator


# ========== ========== ========== ========== ========== ==========
def get_full_qualified_name(cls: type) -> str:
    """
    Get the fully qualified name of a class.

    Returns the full module path and class name, except for built-in types
    which return only the class name.

    Parameters
    ----------
    cls : type
        The class to get the qualified name for.

    Returns
    -------
    str
        Fully qualified name in the format 'module.ClassName' for user-defined
        classes, or just 'ClassName' for built-in types.

    Examples
    --------
    >>> get_full_qualified_name(int)
    'int'

    >>> get_full_qualified_name(list)
    'list'

    >>> from pathlib import Path
    >>> get_full_qualified_name(Path)
    'pathlib.Path'

    >>> class MyClass:
    ...     pass
    >>> get_full_qualified_name(MyClass)
    '__main__.MyClass'

    >>> class Outer:
    ...     class Inner:
    ...         pass
    >>> get_full_qualified_name(Outer.Inner)
    '__main__.Outer.Inner'

    Notes
    -----
    This function uses `__qualname__` which includes nested class names,
    making it suitable for uniquely identifying classes in a registry.
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
    Validate that an object is an instance of one or more expected types.

    Parameters
    ----------
    obj : object
        Object to validate.
    types : type or tuple of type
        Expected type or types.
    can_be_none : bool, default False
        Whether None is considered a valid value.
    raise_error : bool, default True
        Whether to raise TypeError on mismatch.

    Returns
    -------
    bool
        True if the object matches the expected types, False otherwise.

    Raises
    ------
    TypeError
        If the object does not match the expected types and raise_error is True.
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
    Metaclass responsible for building the serialization registry.

    This metaclass automatically registers Serializable subclasses,
    collects SerializableProperty descriptors across the full MRO,
    and maintains global registries for serializable classes,
    primitive types, and dataset-backed types.
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

        if cls is Serializable:
            return Serializable._subclasses[qualname]

        raise KeyError(f'Class {cls.__name__} is not subscriptable')

    def __contains__(cls, subclass: str | type) -> bool:
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

        Primitive types are serialized verbatim without additional metadata.

        Parameters
        ----------
        primitive_type : type
            Type to register.

        Raises
        ------
        TypeError
            If the type is not eligible to be primitive.
        """
        if issubclass(primitive_type, (list, dict, tuple, Serializable)):
            raise TypeError(f'Cannot register {primitive_type} as primitive type')

        Serializable._primitive_types.add(primitive_type)

    def remove_primitive_type(cls, primitive_type: type) -> None:
        """
        Remove a previously registered primitive type.

        Parameters
        ----------
        primitive_type : type
            Type to remove.
        """
        Serializable._primitive_types.discard(primitive_type)

    def is_primitive_type(cls, type_: type) -> bool:
        """
        Check whether a type is registered as primitive.

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

        Dataset types behave as primitives during in-memory serialization,
        but are stored as HDF5 datasets when persisted.

        Parameters
        ----------
        dataset_type : type
            Type to register.
        disassemble : callable
            Function with signature ``disassemble(obj) -> (ndarray, attrs)``.
        assemble : callable
            Function with signature ``assemble(ndarray, attrs) -> obj``.
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
        Remove a registered dataset-backed type.

        Parameters
        ----------
        dataset_type : type
        """
        Serializable.remove_primitive_type(dataset_type)

        if dataset_type in Serializable._dataset_types:

            dataset_type_name = get_full_qualified_name(dataset_type)

            Serializable._dataset_types.pop(dataset_type)
            Serializable._dataset_types.pop(dataset_type_name)

    def is_dataset_type(cls, type_: type) -> bool:
        """
        Check whether a type is registered as dataset-backed.

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
        return list(Serializable._subclasses.values())

    @property
    def primitive_types(cls) -> list[type]:
        return list(Serializable._primitive_types)

    @property
    def dataset_types(cls) -> list[type]:
        return [_type for _type in Serializable._dataset_types.keys() if not isinstance(_type, str)]

    @property
    def serializable_properties(cls) -> dict[str, SerializableProperty]:
        """
        dict[str, SerializableProperty]
        """
        return {**cls._serializable_properties}

    @property
    def copiable_properties(cls) -> dict[str, SerializableProperty]:
        """
        dict[str, SerializableProperty]
        """
        return {k: v for k, v in cls.serializable_properties.items() if v.copiable}


class Serializable(metaclass=SerializableMetatype):
    """
    Base class for structured serialization.

    Serializable objects can be converted to and from Python-native
    data structures composed of dictionaries, lists, scalars, and
    nested Serializable instances.

    Construction
    ------------
    Serializable(**kwargs)
        Initialize by assigning serializable properties.

    Serializable(other)
        Create a logical copy of another instance of the same class.
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

        if type(other) is not type(self):
            error = f"Comparison must be taken between the same type: " \
                    f"{type(other).__name__} is not {type(self).__name__}"

            raise TypeError(error)

        for key in type(self).copiable_properties:

            other_value = getattr(other, key)
            self_value = getattr(self, key)

            if not numpy.all(other_value == self_value):
                return False

        return True

    def __copy__(self):
        return self.copy()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _initialize_from_data(self, data: dict[str, Any]) -> None:
        """Initialize the object from the given data."""
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
        Serialize an object into Python-native data structures.

        Parameters
        ----------
        obj : object
            Object to serialize.
        is_copy : bool, default False
            Whether copy semantics should be applied.

        Returns
        -------
        object
            Serialized representation.

        Raises
        ------
        TypeError
            If no serialization method exists for the object type.
        """
        if obj is None:
            return None

        if isinstance(obj, (tuple, list)):
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
        Deserialize data into a Python object.

        Parameters
        ----------
        data : object
            Serialized data.

        Returns
        -------
        object
            Reconstructed object.

        Raises
        ------
        TypeError
            If deserialization is not possible.
        """
        if data is None:
            return None

        if isinstance(data, (tuple, list)):
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
        Create a logical copy of the object.

        Returns
        -------
        Serializable
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
    Serializable object that can be persisted to disk using HDF5.

    Persistable extends Serializable with structured file I/O,
    dataset-backed storage, and lazy dataset access.
    """

    # ========== ========== ========== ========== ========== class attributes
    extension: str = '.hdf5'

    class ProxyDataset:
        """
        Lazy wrapper around an HDF5 dataset.

        ProxyDataset defers reconstruction until data is accessed and
        exposes a NumPy-like interface for reading and writing.
        """

        def __init__(self, dataset: h5py.Dataset) -> None:
            self._dataset = dataset
            self._attrs = {k: Persistable._load_data_from_h5py_tree(v) for k, v in self._dataset.attrs.items()}
            self._dataset_type_name = self._attrs.pop('__dataset_type__')

        def __getitem__(self, item) -> Any:
            """
            Retrieve one or more elements from the dataset.

            Parameters
            ----------
            item : slice or int

            Returns
            -------
            object
            """
            assemble = Serializable._dataset_types[self._dataset_type_name]['assemble']
            array = self._dataset[item]

            return assemble(array, self.attrs)

        def __setitem__(self, key, value) -> None:
            """
            Assign value(s) to the dataset, resizing if necessary.

            Parameters
            ----------
            key : slice or int
            value : object
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
            Append data to the dataset along axis 0.

            Parameters
            ----------
            value : object
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
        """
        Serialize and save the object to disk.

        Parameters
        ----------
        path : str or Path
            Output path.
        overwrite : bool, default True
            Whether to overwrite an existing file.
        use_default_extension : bool, default True
            Whether to enforce the default file extension.
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
        Load and deserialize an object from disk.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Persistable
        """
        data = cls.load_serialized_data(path)
        return Serializable.deserialize(data)

    # ---------- ---------- ---------- ---------- ---------- properties
    ...


Serializable.register_primitive_type(Persistable.ProxyDataset)


__all__ = [
    'SerializableProperty',
    'serializable_property',
    'Serializable',
    'Persistable'
]