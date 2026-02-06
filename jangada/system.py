#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import uuid

from rich.console import RenderableType, Group
from rich.text import Text
from rich.panel import Panel


from jangada.serialization import Persistable, SerializableProperty, Observer
from jangada.display import Displayable
from jangada.mixin import Identifiable, Taggable, Nameable, Describable

from typing import Any, Iterator, Callable


class System(Persistable, Displayable, Identifiable, Taggable, Nameable, Describable):

    # ========== ========== ========== ========== ========== class attributes
    __hash__ = Identifiable.__hash__
    __eq__ = Identifiable.__eq__

    extension = '.sys'

    # ---------- ---------- ---------- subsystems
    subsystems: dict[str, System] = SerializableProperty(default=lambda self: {})

    @subsystems.parser
    def subsystems(self, value: dict[str, System]) -> dict[str, System]:

        if not isinstance(value, dict):
            raise TypeError(f'Expected dict, got {type(value)}')

        _subsystems = {}
        # ok, recreating the dict looks stupid, I know. But I do this to make
        # sure the tags are correct

        for subsystem in value.values():

            if not isinstance(subsystem, System):
                raise TypeError(f'Expected System, got {type(subsystem)}')

            subsystem.supersystem = self

            _subsystems[subsystem.tag] = subsystem


        return _subsystems

    # ---------- ---------- ---------- _supersystem_id
    _supersystem_id = SerializableProperty(copiable=False)

    @_supersystem_id.parser
    def _supersystem_id(self, value: str|None) -> str|None:

        if value is None:

            if hasattr(self, type(self)._supersystem_id.private_name):
                # it means that's not the first time this property is being set
                # otherwise the first time can lead to infinite recursion

                current_supersystem = self.supersystem

                if current_supersystem is not None:

                    # remove self from supersystem's subsystems
                    if self.tag is not None:
                        current_supersystem.subsystems.pop(self.tag)
                    else:
                        keys_to_delete = [key for key, subsys in current_supersystem.subsystems.items() if
                                          subsys is self]

                        for key in keys_to_delete:
                            del current_supersystem.subsystems[key]

                    # remove tag observer for self from supersystem registry
                    tag_observer = current_supersystem._tag_observers.pop(self)

                    # remove tag observer from self
                    type(self).tag = type(self).tag.remove_observer(tag_observer)
                    type(self).tag.__set_name__(type(self), 'tag')

            return None

        # ---------- ---------- ---------- ---------- validate supersystem
        value = str(value).strip()

        if value not in Identifiable._instances:
            raise ValueError(f'Invalid supersystem ID: {value!r}. Must be a valid ID of an existing System.')

        supersystem: System = Identifiable._instances[value]

        if not isinstance(supersystem, System):
            raise TypeError(f'Expected System, got {type(supersystem)}')

        if supersystem is self:
            raise ValueError(f'System cannot register itself as supersystem.')

        # ---------- ---------- ---------- ---------- check if self is prepared for supersystem
        if self.tag is None:
            raise ValueError(f'Subsystems must have a tag set before being registered as subsystems.')

        if self in supersystem.supersystem_chain:
            raise ValueError(f'System {supersystem} cannot be registered as supersystem of {self}: circular dependency.')

        if self in supersystem:
            return value

        if self.tag in supersystem:
            raise ValueError(f'System {self} cannot be registered as subsystem of {supersystem}: tag already used by another subsystem.')

        # ---------- ---------- ---------- ---------- unregister from the previous supersystem (if any)
        if hasattr(self, type(self)._supersystem_id.private_name):

            current_supersystem = self.supersystem

            if current_supersystem is not None:
                # remove self from supersystem's subsystems
                if self.tag is not None:
                    current_supersystem.subsystems.pop(self.tag)
                else:
                    keys_to_delete = [key for key, subsys in current_supersystem.subsystems.items() if subsys is self]

                    for key in keys_to_delete:
                        del current_supersystem.subsystems[key]

                # remove tag observer for self from supersystem registry
                tag_observer = current_supersystem._tag_observers.pop(self)

                # remove tag observer from self
                type(self).tag = type(self).tag = type(self).tag.remove_observer(tag_observer)
                type(self).tag.__set_name__(type(self), 'tag')

        # ---------- ---------- ---------- ---------- I think that's it... let's set it

        # add self to supersystem's subsystems'
        supersystem.subsystems[self.tag] = self

        # create tag observer for self
        tag_observer = supersystem._get_tag_observer(self)

        # add tag observer to self
        type(self).tag = type(self).tag.add_observer(tag_observer)
        type(self).tag.__set_name__(type(self), 'tag')

        # keep track of tag observer for self in supersystem's registry
        supersystem._tag_observers[self] = tag_observer

        # ---------- ---------- ---------- ----------
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

    def __setitem__(self, tag: str, value: System) -> None:

        if tag in self:
            self.remove(tag)

        value.tag = tag

        self.add(value)

    def __getattr__(self, tag: str) -> System:
        try:
            return self[tag]
        except KeyError as error:
            raise AttributeError(f"No subsystem registered with tag '{tag}'.") from error

    # TODO: attack this later when I grow up
    # def __setattr__(self, name: str, value: Any) -> None:
    #
    #     # check if this is a SerializableProperty on the class
    #     class_attr = getattr(type(self), name, None)
    #
    #     if isinstance(class_attr, SerializableProperty):
    #         super().__setattr__(name, value)  # just let nature does its thing
    #
    #     elif isinstance(value, System):
    #         self[name] = value
    #
    #     else:
    #         super().__setattr__(name, value)

    def __iter__(self) -> Iterator[System]:
        return iter(self.subsystems.values())

    def __len__(self) -> int:
        return len(self.subsystems)

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _get_tag_observer(self, subsystem: System) -> Observer:

        def tag_observer(_subsystem: System, old_value: str | None, new_value: str | None) -> None:

            # print(f'{old_value} -> {new_value}')

            if _subsystem is not subsystem:
                return # make sure it only runs with the right instance! since it's set for the whole class

            if _subsystem not in self:
                return # make sure it only runs with subsystems registered in this system. If everything goes well, this should never happen.

            if new_value is None:
                self.remove(_subsystem)
                raise ValueError(f'Cannot keep subsystem with tag None')

            if new_value in self:
                if self[new_value] is _subsystem:
                    return # user is setting the tag to the same value, so it's already ok I guess

                self.remove(_subsystem)
                raise ValueError(f'Cannot keep subsystem with tag {new_value}: already registered in {self}.')

            if new_value in self.__dict__:
                raise ValueError(f'Cannot keep subsystem with tag {new_value}: already registered as attribute.')

            self.subsystems[new_value] = _subsystem

            if old_value is not None:
                del self.subsystems[old_value]

        return tag_observer

    def _title(self) -> Text:
        return Text(f'{type(self).__name__}', style='italic bold bright_yellow')

    def _content(self) -> RenderableType:

        contents = []

        # ---------- ---------- ---------- ---------- info
        info_data = {
            'name': self.name,
            'description': self.description,
            'tag': self.tag,
        }

        info_panel = Panel(self.format_as_form(info_data),
                           title='Info',
                           title_align='right')

        contents.append(info_panel)

        # ---------- ---------- ---------- ---------- ---------- ----------
        subsystem_data = {tag: sys.name for tag, sys in self.subsystems.items()}

        if subsystem_data:

            subsystem_panel = Panel(self.format_as_form(subsystem_data),
                                    title=Text('Subsystems', style='bold bright_yellow'),
                                    title_align='right',
                                    expand=True)

            contents.append(subsystem_panel)

        # ---------- ---------- ---------- ---------- ---------- ----------
        id_data = {
            'id': self.id,
        }

        id_panel = Panel(self.format_as_form(id_data))

        contents.append(id_panel)

        # ---------- ---------- ---------- ---------- ---------- ----------
        return Group(*contents)

    @property
    def _tag_observers(self) -> dict[System, Observer]:

        try:
            return self.__tag_observers
        except AttributeError:
            self.__tag_observers = {}
            return self.__tag_observers

    # ========== ========== ========== ========== ========== public methods
    def add(self, *subsystems: System) -> None:

        for subsystem in subsystems:
            subsystem.supersystem = self

    def remove(self, subsystem: System|str) -> None:
        try:
            subsystem = subsystem if isinstance(subsystem, System) else self[subsystem]

            if subsystem not in self:
                raise ValueError(f'Subsystem {subsystem} not registered in {self}.')

        except KeyError as error:
            raise ValueError(f'Subsystem {subsystem} not registered in {self}.') from error

        else:
            subsystem.supersystem = None

    def equal(self, system: System) -> bool:
        return Persistable.__eq__(self, system)

    # ---------- ---------- ---------- ---------- ---------- properties
    @property
    def supersystem(self) -> System|None:

        if self._supersystem_id is None:
            return None

        return Identifiable.get_instance(self._supersystem_id)

    @supersystem.setter
    def supersystem(self, value: System|None) -> None:
        if not isinstance(value, System) and value is not None:
            raise TypeError(f'Expected System, got {type(value)}')

        self._supersystem_id = value.id if value is not None else None

    @property
    def supersystem_chain(self) -> list[System]:
        chain = []

        system = self.supersystem

        while system is not None:
            chain.append(system)
            system = system.supersystem

        return chain
