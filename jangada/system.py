#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

from jangada.serialization import Persistable, SerializableProperty
from jangada.mixin import Identifiable, Taggable, Nameable, Describable, Activatable

from typing import Any, Iterator


class System(Persistable, Identifiable, Taggable, Nameable, Describable, Activatable):

    # ========== ========== ========== ========== ========== class attributes
    __hash__ = Identifiable.__hash__

    extension = '.sys'

    subsystems = SerializableProperty()

    @subsystems.default
    def subsystems(self) -> set[System]:
        return set()

    @subsystems.parser
    def subsystems(self, value: Any) -> set[System]:
        if value is None:
            return set()

        _subsystems = set(value)
        for subsystem in _subsystems:
            if not isinstance(subsystem, System):
                raise TypeError(f'Expected System, got {type(subsystem)}')

        return _subsystems

    # ========== ========== ========== ========== ========== special methods
    def __contains__(self, obj: System|str) -> bool:
        if isinstance(obj, System):
            return obj in self.subsystems

        if not isinstance(obj, str):  # this includes obj == None
            return False

        return obj in (subsystem.tag for subsystem in self.subsystems)

    def __getitem__(self, tag: str) -> System:
        if tag is None:
            raise KeyError('Cannot retrieve a subsystem with a None tag.')

        for subsystem in self.subsystems:
            if subsystem.tag == tag:
                return subsystem

        raise KeyError(f"No subsystem registered with tag '{tag}'.")

    def __getattr__(self, tag: str) -> System:
        try:
            return self[tag]
        except KeyError as error:
            raise AttributeError(f"No subsystem registered with tag '{tag}'.") from error

    def __iter__(self) -> Iterator[System]:
        return iter(self.subsystems)

    def __len__(self) -> int:
        return len(self.subsystems)
    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    ...

    # ========== ========== ========== ========== ========== public methods
    def add(self, *subsystems: System) -> None:

        for subsystem in subsystems:

            if not isinstance(subsystem, System):
                raise TypeError(f'Expected System, got {type(subsystem)}')

            if subsystem in self:
                raise ValueError(f'Subsystem {subsystem} already registered in {self}.')

            if subsystem is self:
                raise ValueError(f'System cannot register itself as subsystem.')

            if subsystem.tag is None:
                raise ValueError(f'Systems must have a tag set before being registered as subsystems.')

            if subsystem.tag in self:
                raise ValueError(f'Subsystem with tag {subsystem.tag} already registered in {self}.')
                # FIXME: it does not solve the problem entirely. After registered, a tag might be set
                #  to None or other already registered tag. Not sure how to address this problem right
                #  now... should I maybe rewrite tag observers to the new subsystem during registration?
                #  It seems complicated, but the right direction... maybe I should rewrite the logic of
                #  observers to include "slots"?

            self.subsystems.add(subsystem)

    def remove(self, subsystem: System|str) -> None:

        try:

            if isinstance(subsystem, str):
                _subsystem = self[subsystem]
            else:
                _subsystem = subsystem

            self.subsystems.remove(_subsystem)
        except KeyError as error:
            raise ValueError(f'Subsystem {subsystem} not registered in {self}.') from error

    # ---------- ---------- ---------- ---------- ---------- properties
    ...