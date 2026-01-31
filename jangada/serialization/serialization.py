#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import weakref

from numbers import Number
from pathlib import Path

import numpy
import pandas

from abc import ABCMeta, abstractmethod

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

    # ========== ========== ========== ========== ========== class attributes
    __slots__ = ('fget', 'fset', 'fdel',
                 '_default', '_parser', '_observer',
                 '_writeonce', '_copiable', '_readonly',
                 'name', 'private_name', 'owner', '__doc__', '__weakref__')

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
        """Get the property value."""
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
        """Set the property value."""
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
        """Delete the property value."""
        if self.fdel is None:
            raise AttributeError(f"can't delete attribute '{self.name}'")

        self.fdel(instance)

    # ========== ========== Descriptor protocol methods to work like @property
    def getter(self, fget: Getter) -> Self:
        """Set the getter function."""
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
        """Set the setter function."""
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
        """Set the deleter function."""
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
        """Set the default value."""
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
        """Set the parser function."""
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
        """Set the observer function."""
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
        """Check if property is read-only (no setter)."""
        return self._readonly

    @property
    def writeonce(self) -> bool:
        """Check if property is write-once property."""
        return self._writeonce

    @property
    def copiable(self) -> bool:
        """Check if property is copiable property."""
        return self._copiable


def serializable_property(
        default: T | Getter | None = None,
        readonly: bool = False,
        writeonce: bool = False,
        copiable: bool = True) -> Callable[[Getter], SerializableProperty]:

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
    """Checks if ``obj`` is instance of ``types``.

    This is convenience function designed to facilitate/standardise the raising
    of an error when the type of ``obj`` is unexpected.

    Parameters
    ----------
    obj : :py:class:`object`
        Object whose type must be checked.

    types :  :py:class:`type` | :py:class:`tuple` of :py:class:`types <type>`
        Expected types of ``obj``.

    can_be_none : :py:class:`bool`
        Set this if :py:data:`None` is an acceptable value for ``obj``. If set
        and ``obj is None``, :py:data:`True` is returned.

    raise_error : :py:class:`bool`
        Controls how the function behaves when ``obj`` is not an instance of
        the given types. If :py:data:`True`, then :class:`TypeError` is raised.
        Else, no error is raised.

    Returns
    -------
    out : :py:class:`bool`
        :py:data:`True` if ``obj`` is an instance of one of the given
        ``types``. :py:data:`False` otherwise.

    Raises
    ------
    :class:`DTypeError`
        if ``obj`` is not an instance of one of the given ``types`` and
        ``raise_error = True``

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
        if name is 'Serializable':
            cls._subclasses: dict[str, Type[Serializable]] = {}
            cls._primitive_types: set[type] = set()
            cls._dataset_types: dict[type|str, dict[str, Any]] = {}

        else:
            qualname: str = get_full_qualified_name(cls)
            Serializable._subclasses[qualname] = cls

            cls._serializable_properties = weakref.WeakValueDictionary()

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
        """Register primitive type"""
        if issubclass(primitive_type, (list, dict, tuple, Serializable)):
            raise TypeError(f'Cannot register {primitive_type} as primitive type')

        Serializable._primitive_types.add(primitive_type)

    def remove_primitive_type(cls, primitive_type: type) -> None:
        """Remove primitive type"""
        Serializable._primitive_types.discard(primitive_type)

    def is_primitive_type(cls, type_: type) -> bool:
        """Test if type is primitive"""
        return type_ in Serializable._primitive_types

    def register_dataset_type(cls,
                              dataset_type: type,
                              disassemble: Callable[[Any], tuple[NDArray, dict]],
                              assemble: Callable[[NDArray, dict], Any]) -> None:
        """Register dataset type"""
        dataset_type_name = get_full_qualified_name(dataset_type)

        process = {
            'disassemble': disassemble,
            'assemble': assemble,
        }

        Serializable._dataset_types[dataset_type] = process
        Serializable._dataset_types[dataset_type_name] = process
        Serializable.register_primitive_type(dataset_type)

    def remove_dataset_type(cls, dataset_type: type) -> None:
        """Remove dataset type"""
        Serializable.remove_primitive_type(dataset_type)

        if dataset_type in Serializable._dataset_types:

            dataset_type_name = get_full_qualified_name(dataset_type)

            Serializable._dataset_types.pop(dataset_type)
            Serializable._dataset_types.pop(dataset_type_name)

    def is_dataset_type(cls, type_: type) -> bool:
        """Test if type is dataset"""
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
        return {**cls._serializable_properties}

    @property
    def copiable_properties(cls) -> dict[str, SerializableProperty]:
        return {k: v for k, v in cls.serializable_properties.items() if v.copiable}


class Serializable(metaclass=SerializableMetatype):

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

        if data is None:
            return

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

            return cls(**data)

        if not isinstance(data, (*tuple(Serializable.primitive_types), Serializable)):
            error = f"No serialisation method is implemented for object of " \
                    f"type {type(data)}."
            raise TypeError(error)

        return data

    def copy(self) -> Serializable:
        """TODO"""
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
    """Base class for serializable objects that can be persisted to disk."""

    # ========== ========== ========== ========== ========== class attributes
    extension: str = '.hdf5'

    # ========== ========== ========== ========== ========== special methods
    ...

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    ...

    # ========== ========== ========== ========== ========== public methods
    ...

    # ---------- ---------- ---------- ---------- ---------- properties
    ...









__all__ = [
    'SerializableProperty',
    'serializable_property',
    'Serializable'
]