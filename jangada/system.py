#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

from rich.console import RenderableType, Group
from rich.text import Text
from rich.panel import Panel


from jangada.serialization import Persistable, SerializableProperty
from jangada.display import Displayable
from jangada.mixin import Identifiable, Taggable, Nameable, Describable

from typing import Any, Iterator, Callable


class System(Persistable, Displayable, Identifiable, Taggable, Nameable, Describable):

    # ========== ========== ========== ========== ========== class attributes
    __hash__ = Identifiable.__hash__
    __eq__ = Identifiable.__eq__

    extension = '.sys'

    # ========== ========== ========== ========== serializable properties
    ...

    # ---------- ---------- ---------- ---------- subsystems
    subsystems: dict[str, System] = SerializableProperty(default={})

    @subsystems.parser
    def subsystems(self, value: dict[str, System]) -> dict[str, System]:

        if not isinstance(value, dict):
            raise TypeError(f'Expected dict, got {type(value)}')

        _subsystems = {}

        self._tag_observers = {}

        for tag, subsystem in value.items():

            if not isinstance(subsystem, System):
                raise TypeError(f'Expected System, got {type(subsystem)}')

            if subsystem.tag != tag:
                raise ValueError(f'Subsystem {subsystem} has tag {subsystem.tag} different from {tag}.')

            if subsystem.tag is None:
                raise ValueError(f'Cannot register subsystem with tag None: {subsystem}.')

            if subsystem.tag in _subsystems:
                raise ValueError(f'Cannot register two subsystems with the same tag: {subsystem} and {_subsystems[subsystem.tag]}.')

            if subsystem in self:
                raise ValueError(f'Cannot register subsystem {subsystem} already registered in {self}.')

            if subsystem is self:
                raise ValueError(f'System cannot register itself as subsystem.')

            _subsystems[subsystem.tag] = subsystem

            tag_observer = self._get_tag_observer(subsystem)

            type(subsystem).tag.add_observer(tag_observer)
            self._tag_observers[subsystem] = tag_observer

        return _subsystems

    # ---------- ---------- ---------- ---------- _supersystem_id
    _supersystem_id = SerializableProperty()

    @_supersystem_id.parser
    def supersystem_id(self, value: str|None) -> str|None:

        if value is None:
            return None

        value = str(value).strip()

        if value not in Identifiable._instances:
            raise ValueError(f'Invalid supersystem ID: {value!r}. Must be a valid ID of an existing System.')

        return value

    # ========== ========== ========== ========== ========== special methods
    def __contains__(self, obj: System|str) -> bool:

        if isinstance(obj, System):
            for registered_obj in self.subsystems.values():
                if obj is registered_obj:
                    return True

            return False

            # the following comparison looks simpler, but it's not what I need
            # since it compares with __eq__ instead of "is"
            # return obj in self.subsystems.values()

        if not isinstance(obj, str):  # this includes obj == None
            return False

        return obj in self.subsystems

    def __getitem__(self, tag: str) -> System:
        if tag is None:
            raise KeyError('Cannot retrieve a subsystem with a None tag.')

        if tag in self.subsystems:
            return self.subsystems[tag]

        raise KeyError(f"No subsystem registered with tag '{tag}'.")

    def __getattr__(self, tag: str) -> System:
        try:
            return self[tag]
        except KeyError as error:
            raise AttributeError(f"No subsystem registered with tag '{tag}'.") from error

    def __iter__(self) -> Iterator[System]:
        return iter(self.subsystems.values())

    def __len__(self) -> int:
        return len(self.subsystems)

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _get_tag_observer(self, subsystem: System) -> Callable[[System, str|None, str|None], None]:

        def tag_observer(_subsystem: System, old_value: str | None, new_value: str | None) -> None:

            if _subsystem is not subsystem:
                return

            if _subsystem not in self:
                return

            if new_value is None:
                self.remove(_subsystem)
                raise ValueError(f'Cannot keep subsystem with tag None')

            if new_value in self:
                if self[new_value] is _subsystem:
                    return

                raise ValueError(f'Cannot keep subsystem with tag {new_value}: already registered in {self}.')

            self.subsystems[new_value] = _subsystem

            if old_value is not None:
                del self.subsystems[old_value]

        return tag_observer

    def _title(self) -> Text:
        return Text(f'{type(self).__name__}', style='italic bold bright_yellow')

    def _content(self) -> RenderableType:

        info_data = {
            'name': self.name,
            'description': self.description,
            'tag': self.tag,
        }

        info_panel = Panel(self.format_as_form(info_data),
                           title='Info',
                           title_align='right')

        # ---------- ---------- ---------- ---------- ---------- ----------
        subsystem_data = {tag: sys.name for tag, sys in self.subsystems.items()}

        subsystem_panel = Panel(self.format_as_form(subsystem_data),
                                title=Text('Subsystems', style='bold bright_yellow'),
                                title_align='right',
                                expand=True)


        # ---------- ---------- ---------- ---------- ---------- ----------
        id_data = {
            'id': self.id,
        }

        id_panel = Panel(self.format_as_form(id_data))



        content = Group(
            info_panel,
            subsystem_panel,
            # Rule(style='bright_cyan'),
            id_panel
        )

        return content


    # ========== ========== ========== ========== ========== public methods
    def add(self, *subsystems: System) -> None:

        for subsystem in subsystems:

            if not isinstance(subsystem, System):
                raise TypeError(f'Expected System, got {type(subsystem)}')

            if subsystem in self:
                raise ValueError(f'Subsystem {subsystem} with tag={subsystem.tag} already registered in {self}.')

            if subsystem is self:
                raise ValueError(f'System cannot register itself as subsystem.')

            if subsystem.tag is None:
                raise ValueError(f'Systems must have a tag set before being registered as subsystems.')

            if subsystem.tag in self:
                raise ValueError(f'Subsystem with tag {subsystem.tag} already registered in {self}.')

            tag_observer = self._get_tag_observer(subsystem)

            if not hasattr(self, '_tag_observers'):
                self._tag_observers = {}

            type(subsystem).tag.add_observer(tag_observer)
            self._tag_observers[subsystem] = tag_observer

            self.subsystems[subsystem.tag] = subsystem

            if subsystem.supersystem is not self:
                subsystem.supersystem = self

    def remove(self, subsystem: System|str) -> None:

        try:

            subsystem = subsystem if isinstance(subsystem, System) else self[subsystem]
            del self.subsystems[subsystem.tag]

            tag_observer = self._tag_observers.pop(subsystem)
            type(subsystem).tag.remove_observer(tag_observer)

        except KeyError as error:
            raise ValueError(f'Subsystem {subsystem} not registered in {self}.') from error

    def equal(self, system: System) -> bool:
        return Persistable.__eq__(self, system)
    # ---------- ---------- ---------- ---------- ---------- properties
    @property
    def supersystem(self) -> System|None:

        if self.supersystem_id is None:
            return None

        return Identifiable.get_instance(self.supersystem_id)

    @supersystem.setter
    def supersystem(self, value: System|None) -> None:

        previous_supersystem = self.supersystem

        # no need to do anything if the new value is the same as the previous one
        if previous_supersystem is value:
            return

        if value is not None and self in value:
            return

        # remove the subsystem from the previous supersystem
        if previous_supersystem is not None:
            if self in previous_supersystem:
                previous_supersystem.remove(self)

        # add the subsystem to the new supersystem
        if value is None:
            self.supersystem_id = None

        else:
            try:
                self.supersystem_id = value.id
                value.add(self)

            except Exception as error:
                self.supersystem_id = None
                raise error from error
