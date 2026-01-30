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

Parser: TypeAlias = Callable[[object, Any], T]


class SerializableProperty:

    # ========== ========== ========== ========== ========== class attributes
    __slots__ = ('fget', 'fset', 'fdel',
                 '_default', '_parser', 'writeonce', 'copiable',
                 'name', 'owner', '__doc__', '__weakref__')

    # ========== ========== ========== ========== ========== special methods
    def __init__(self,
                 fget: Getter | None = None,
                 fset: Setter | None = None,
                 fdel: Deleter | None = None,
                 *,
                 default: T | Getter | None = None,
                 parser: Parser | None = None,
                 writeonce: bool = False,
                 copiable: bool = True,
                 doc: str | None = None) -> None:

        self.fget: Getter | None = fget
        self.fset: Setter | None = fset
        self.fdel: Deleter | None = fdel

        self._default: T | Getter | None = default
        self._parser: Parser | None = parser
        self.writeonce: bool = writeonce
        self.copiable: bool = copiable

        # Use getter docstring if not provided (which can also be None)
        self.__doc__: str | None = fget.__doc__ if doc is None and fget is not None else doc

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name: str = name
        self.owner: type = owner

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

        if self.writeonce:
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

        self.fset(instance, value)

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
            writeonce=self.writeonce,
            copiable=self.copiable,
            doc=self.__doc__
        )

    def setter(self, fset: Setter) -> Self:
        """Set the setter function."""
        return type(self)(
            self.fget, fset, self.fdel,
            default=self._default,
            parser=self._parser,
            writeonce=self.writeonce,
            copiable=self.copiable,
            doc=self.__doc__
        )

    def deleter(self, fdel: Deleter) -> Self:
        """Set the deleter function."""
        return type(self)(
            self.fget, self.fset, fdel,
            default=self._default,
            parser=self._parser,
            writeonce=self.writeonce,
            copiable=self.copiable,
            doc=self.__doc__
        )

    # ---------- ---------- and more!!
    def default(self, func: Getter) -> Self:
        """Set the default value."""
        return type(self)(
            self.fget, self.fset, self.fdel,
            default=func,
            parser=self._parser,
            writeonce=self.writeonce,
            copiable=self.copiable,
        )

    def parser(self, func: Parser) -> Self:
        """Set the parser function."""
        return type(self)(
            self.fget, self.fset, self.fdel,
            default=self._default,
            parser=func,
            writeonce=self.writeonce,
            copiable=self.copiable,
        )

    @property
    def is_readonly(self) -> bool:
        """Check if property is read-only (no setter)."""
        return self.fset is None

    @property
    def is_writeonce(self) -> bool:
        """Check if property is write-once property."""
        return self.writeonce


def serializable_property(
        writeonce: bool = False,
        copiable: bool = True) -> Callable[[Getter], SerializableProperty]:

    def decorator(getter: Getter) -> SerializableProperty:
        return SerializableProperty(
            fget=getter,
            writeonce=writeonce,
            copiable=copiable,
        )

    return decorator