#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

# ---------- ---------- ---------- ---------- ---------- ---------- typing
from typing import TypeVar, Callable, Any, TypeAlias, Self


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
                    f"{self.name} is a write-once property and has already been set with value {current_value}"
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