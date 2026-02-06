#  -*- coding: utf-8 -*-
"""
Hierarchical namespace system for organizing domain objects.

This module provides the System class, a powerful abstraction for building
hierarchical structures with namespace-style access to subsystems. Systems
can be nested arbitrarily deep, with automatic parent-child relationship
management, tag-based lookup, and dynamic reorganization.
"""

from __future__ import annotations

from rich.console import RenderableType, Group
from rich.text import Text
from rich.panel import Panel

from jangada.serialization import Persistable, SerializableProperty, Observer
from jangada.display import Displayable
from jangada.mixin import Identifiable, Taggable, Nameable, Describable

from typing import Any, Iterator, Callable


class System(Persistable, Displayable, Identifiable, Taggable, Nameable, Describable):
    """
    Hierarchical namespace container for organizing subsystems.

    System provides a framework for building tree-structured hierarchies with
    automatic bidirectional parent-child relationship management. Subsystems
    can be accessed via tag using either dict-style (``system['tag']``) or
    attribute-style (``system.tag``) syntax.

    Key features:
    - Automatic parent-child relationship management
    - Tag-based namespace access (dict and attribute style)
    - Dynamic tag reorganization via observers
    - Circular dependency prevention
    - Persistence of entire hierarchies
    - Rich terminal display integration

    Class Attributes
    ----------------
    extension : str
        File extension for saved systems ('.sys').

    Attributes
    ----------
    subsystems : dict[str, System]
        Dictionary mapping tags to subsystem objects. Managed automatically.
    supersystem : System or None
        Parent system in hierarchy. Setting this registers/unregisters from parent.
    supersystem_chain : list[System]
        List of all ancestors from parent to root. Read-only.

    Inherited Attributes
    --------------------
    From Identifiable:
        id : str - Globally unique UUID v4 identifier
    From Taggable:
        tag : str or None - Symbolic identifier for namespace access
    From Nameable:
        name : str or None - Human-readable name
    From Describable:
        description : str or None - Free-form description
    From Displayable:
        display_settings : DisplaySettings - Terminal display configuration

    Examples
    --------
    Create simple hierarchy::

        root = System(tag='root', name='Root System')
        child = System(tag='child', name='Child System')

        root.add(child)

        # Access via dict or attribute
        assert root['child'] is child
        assert root.child is child

    Dict-style registration::

        root = System(tag='root')
        root['sensor'] = System(name='Temperature Sensor')

        # Tag is automatically set from key
        assert root['sensor'].tag == 'sensor'

    Build complex hierarchy::

        system = System(tag='system')

        # Using dict-style
        system['sensors'] = System(name='Sensors')
        system['sensors']['temperature'] = System(name='Temp Sensor')
        system['sensors']['pressure'] = System(name='Pressure Sensor')

        # Navigate
        temp = system.sensors.temperature
        assert temp.name == 'Temp Sensor'

    Dynamic reorganization::

        root = System(tag='root')
        child = System(tag='old_tag')
        root.add(child)

        # Change tag - dict keys update automatically
        child.tag = 'new_tag'
        assert 'new_tag' in root
        assert 'old_tag' not in root

    Save and load hierarchies::

        root = System(tag='root')
        root['child'] = System(tag='child', name='Child')

        root.save('hierarchy.sys')

        loaded = System.load('hierarchy.sys')
        assert loaded.child.name == 'Child'

    Content comparison::

        sys1 = System(tag='test', name='Name')
        sys2 = System(tag='test', name='Name')

        # Different IDs, so ID-based equality is False
        assert sys1 != sys2

        # But same content, so content equality is True
        assert sys1.equal(sys2)

    Notes
    -----
    **Hierarchy Structure**
        Systems form a tree structure (not DAG). Each system has at most one
        parent. Circular dependencies are prevented automatically.

    **Tag Requirements**
        Subsystems must have tags to be registered. Tags must be valid Python
        identifiers and cannot be Python keywords.

    **Tag Observers**
        When a subsystem's tag changes, the parent's subsystems dict is
        automatically updated. This is achieved through the observer pattern.

    **Identity vs Content Equality**
        - ``sys1 == sys2`` uses Identifiable (ID-based equality)
        - ``sys1.equal(sys2)`` uses Persistable (content-based equality)

    **Persistence**
        Saving a system saves its entire subtree. Parent references are stored
        as IDs, allowing proper reconstruction on load.

    **Reserved Attributes**
        Avoid using subsystem tags that conflict with System's methods and
        properties (e.g., 'add', 'remove', 'subsystems', 'tag', 'id').

    See Also
    --------
    Persistable : Serialization base class
    Displayable : Terminal display base class
    Identifiable : Unique identification mixin
    Taggable : Tag-based identification mixin
    """

    # ========== ========== ========== ========== ========== class attributes
    __hash__ = Identifiable.__hash__
    """Use Identifiable's hash implementation (ID-based)."""

    __eq__ = Identifiable.__eq__
    """Use Identifiable's equality implementation (ID-based)."""

    extension = '.sys'
    """File extension for saved System files."""

    # ---------- ---------- ---------- subsystems
    subsystems: dict[str, System] = SerializableProperty(default=lambda self: {}, doc = """
Dictionary mapping tags to subsystem objects.

Automatically managed - subsystems are added/removed via add(), remove(),
dict-style assignment, or by setting supersystem property. Manual
modification is discouraged as it bypasses validation.

Type
----
dict[str, System]

Default
-------
Empty dict (factory function)

Constraints
-----------
- Keys must match subsystem tags
- Values must be System instances
- Reconstructed on deserialization to ensure consistency

Examples
--------
Access subsystems::

    for tag, subsystem in system.subsystems.items():
        print(f"{tag}: {subsystem.name}")

Check contents::

    if 'sensor' in system.subsystems:
        sensor = system.subsystems['sensor']

Iterate::

    for subsystem in system.subsystems.values():
        process(subsystem)

Notes
-----
**Automatic Management**
    This dict is managed automatically. When you add a subsystem via
    ``system.add(child)`` or ``system['tag'] = child``, it is added to
    this dict. When tags change, dict keys update automatically via
    observers.

**Direct Modification**
    Avoid directly modifying this dict (e.g., ``system.subsystems['tag'] = child``)
    as it bypasses validation and observer setup. Use ``add()``, ``remove()``,
    or dict-style assignment (``system['tag'] = child``) instead.

**Serialization**
    When a System is saved, the subsystems dict is serialized, preserving
    the entire hierarchy. Parent references are stored as IDs to avoid
    circular references.

See Also
--------
add : Add subsystems
remove : Remove subsystems
__setitem__ : Dict-style subsystem registration
""")

    @subsystems.parser
    def subsystems(self, value: dict[str, System]) -> dict[str, System]:
        """
        Parse and validate subsystems dictionary.

        Parameters
        ----------
        value : dict[str, System]
            Dictionary mapping tags to System objects.

        Returns
        -------
        dict[str, System]
            Validated dictionary with correct tag-to-subsystem mapping.

        Raises
        ------
        TypeError
            If value is not a dict or contains non-System values.

        Notes
        -----
        Reconstructs the dict to ensure tags match keys. Each subsystem's
        supersystem is set to self during parsing.
        """
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
    _supersystem_id = SerializableProperty(copiable=False, doc = """
Parent system ID.

Stores the parent's ID rather than a direct reference to avoid circular
reference issues in serialization. The actual parent object is accessed
via the supersystem property, which looks up the ID in Identifiable's
global registry.

Type
----
str or None

Default
-------
None (no parent - this is a root system)

Constraints
-----------
- Must be a valid ID of an existing System instance
- Setting to None unregisters from parent
- Non-copiable (each copy gets independent parent relationship)
- Automatically manages bidirectional relationship

Examples
--------
Not typically accessed directly - use supersystem property instead::

    # Don't do this:
    system._supersystem_id = parent.id

    # Do this instead:
    system.supersystem = parent

Internal use::

    # Check if system has parent
    if system._supersystem_id is not None:
        print("Has parent")

Notes
-----
**Why Store ID Instead of Reference?**
    Storing the parent's ID instead of a direct reference solves several
    problems:
    1. Avoids circular references in serialization
    2. Allows garbage collection of parent (weak reference behavior)
    3. Enables proper save/load of hierarchies
    4. Works with Identifiable's weak reference registry

**Parser Logic**
    The parser for this property is complex and handles:
    - Unregistration from old parent (when set to None)
    - Validation of new parent
    - Circular dependency prevention
    - Tag observer setup/removal
    - Subsystems dict synchronization

**Serialization**
    This property is non-copiable. When a System is copied, the copy does
    not have a parent (is a new root). This ensures copies are independent.

See Also
--------
supersystem : Property for accessing parent
supersystem_chain : List of all ancestors
""")

    @_supersystem_id.parser
    def _supersystem_id(self, value: str|None) -> str|None:
        """
        Parse and validate supersystem ID with full relationship management.

        Handles:
        - Unregistration from old parent (if value is None)
        - Validation of new parent
        - Registration with new parent
        - Tag observer setup/cleanup

        Parameters
        ----------
        value : str or None
            ID of parent system, or None to unregister.

        Returns
        -------
        str or None
            Validated parent ID or None.

        Raises
        ------
        ValueError
            If ID doesn't exist, tag is None, circular dependency detected,
            or tag conflicts with existing subsystem.
        TypeError
            If ID doesn't refer to a System instance.

        Notes
        -----
        This is the core of the bidirectional relationship management. Setting
        supersystem automatically:
        - Removes from old parent's subsystems dict
        - Adds to new parent's subsystems dict
        - Sets up tag observer for dynamic reorganization
        - Validates hierarchy constraints
        """
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
        """
        Check if subsystem is registered.

        Supports checking by object (identity) or tag (string).

        Parameters
        ----------
        obj : System or str
            Subsystem object or tag string to check.

        Returns
        -------
        bool
            True if subsystem is registered, False otherwise.

        Notes
        -----
        When checking by object, uses identity (``is``) not equality (``==``).
        This ensures the actual registered instance is checked, not just an
        object with the same ID.

        Examples
        --------
        >>> root = System(tag='root')
        >>> child = System(tag='child')
        >>> root.add(child)
        >>> child in root
        True
        >>> 'child' in root
        True
        >>> 'nonexistent' in root
        False
        """

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
        """
        Get subsystem by tag (dict-style access).

        Parameters
        ----------
        tag : str
            Tag of subsystem to retrieve.

        Returns
        -------
        System
            Subsystem with specified tag.

        Raises
        ------
        KeyError
            If tag is None or no subsystem with that tag exists.

        Examples
        --------
        >>> root = System(tag='root')
        >>> root['child'] = System(name='Child')
        >>> root['child'].name
        'Child'
        """

        if tag is None:
            raise KeyError('Cannot retrieve a subsystem with a None tag.')

        if tag in self.subsystems:
            return self.subsystems[tag]

        raise KeyError(f"No subsystem registered with tag '{tag}'.")

    def __setitem__(self, tag: str, value: System) -> None:
        """
        Register subsystem via dict-style assignment.

        Automatically sets subsystem's tag to match the key and registers it.
        If a subsystem already exists with that tag, it is removed first.

        Parameters
        ----------
        tag : str
            Tag to assign to the subsystem.
        value : System
            Subsystem to register.

        Examples
        --------
        >>> root = System(tag='root')
        >>> root['sensor'] = System(name='Sensor')
        >>> root['sensor'].tag
        'sensor'
        >>> root['sensor'].supersystem is root
        True

        Notes
        -----
        This is equivalent to::

            value.tag = tag
            root.add(value)

        If a subsystem already exists with the given tag, it is removed before
        adding the new one.
        """

        if tag in self:
            self.remove(tag)

        value.tag = tag

        self.add(value)

    def __getattr__(self, tag: str) -> System:
        """
        Get subsystem by tag (attribute-style access).

        Called when normal attribute lookup fails. Attempts to retrieve
        subsystem from the subsystems dict.

        Parameters
        ----------
        tag : str
            Tag of subsystem to retrieve.

        Returns
        -------
        System
            Subsystem with specified tag.

        Raises
        ------
        AttributeError
            If no subsystem with that tag exists.

        Examples
        --------
        >>> root = System(tag='root')
        >>> root['child'] = System(name='Child')
        >>> root.child.name
        'Child'

        Notes
        -----
        Provides symmetry with dict-style access:
        - ``system['tag']`` - Dict-style
        - ``system.tag`` - Attribute-style (via this method)
        """
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
        """
        Iterate over direct subsystems.

        Yields
        ------
        System
            Each direct subsystem in order.

        Notes
        -----
        Only iterates over direct children, not entire subtree. For recursive
        iteration, implement a walk() method.

        Examples
        --------
        >>> root = System(tag='root')
        >>> for i in range(3):
        ...     root[f'child{i}'] = System()
        >>> len(list(root))
        3
        """
        return iter(self.subsystems.values())

    def __len__(self) -> int:
        """
        Return number of direct subsystems.

        Returns
        -------
        int
            Count of direct subsystems.

        Notes
        -----
        Only counts direct children, not entire subtree.

        Examples
        --------
        >>> root = System(tag='root')
        >>> root['child1'] = System()
        >>> root['child2'] = System()
        >>> len(root)
        2
        """
        return len(self.subsystems)

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    def _get_tag_observer(self, subsystem: System) -> Observer:
        """
        Create tag change observer for a subsystem.

        The observer handles automatic dict key updates when a subsystem's
        tag changes.

        Parameters
        ----------
        subsystem : System
            Subsystem to create observer for.

        Returns
        -------
        Observer
            Observer function that handles tag changes.

        Notes
        -----
        The observer:
        - Updates subsystems dict when tag changes
        - Removes subsystem if tag becomes None
        - Prevents tag conflicts
        - Maintains consistency with __dict__
        """
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
        """
        Generate display title.

        Returns
        -------
        Text
            Styled title showing class name.
        """
        return Text(f'{type(self).__name__}', style='italic bold bright_yellow')

    def _content(self) -> RenderableType:
        """
        Generate display content.

        Returns
        -------
        RenderableType
            Rich Group containing info, subsystems, and ID panels.
        """
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
        """
        Get tag observer registry.

        Returns
        -------
        dict[System, Observer]
            Dictionary mapping subsystems to their tag observers.

        Notes
        -----
        Lazy initialization - creates dict on first access. Not serialized
        since observers are reconstructed from subsystems dict on load.
        """
        try:
            return self.__tag_observers
        except AttributeError:
            self.__tag_observers = {}
            return self.__tag_observers

    # ========== ========== ========== ========== ========== public methods
    def add(self, *subsystems: System) -> None:
        """
        Add one or more subsystems.

        Parameters
        ----------
        *subsystems : System
            Subsystems to add.

        Raises
        ------
        ValueError
            If subsystem has no tag, tag conflicts exist, or circular
            dependency would be created.
        TypeError
            If argument is not a System.

        Examples
        --------
        >>> root = System(tag='root')
        >>> child1 = System(tag='child1')
        >>> child2 = System(tag='child2')
        >>> root.add(child1, child2)
        >>> len(root)
        2

        Notes
        -----
        Equivalent to setting ``subsystem.supersystem = self`` for each subsystem.
        """
        for subsystem in subsystems:
            subsystem.supersystem = self

    def remove(self, subsystem: System|str) -> None:
        """
        Remove a subsystem.

        Parameters
        ----------
        subsystem : System or str
            Subsystem object or tag to remove.

        Raises
        ------
        ValueError
            If subsystem is not registered in this system.

        Examples
        --------
        >>> root = System(tag='root')
        >>> child = System(tag='child')
        >>> root.add(child)
        >>> root.remove(child)
        >>> child in root
        False

        >>> root.add(child)
        >>> root.remove('child')
        >>> child in root
        False

        Notes
        -----
        Equivalent to setting ``subsystem.supersystem = None``.
        """
        try:
            subsystem = subsystem if isinstance(subsystem, System) else self[subsystem]

            if subsystem not in self:
                raise ValueError(f'Subsystem {subsystem} not registered in {self}.')

        except KeyError as error:
            raise ValueError(f'Subsystem {subsystem} not registered in {self}.') from error

        else:
            subsystem.supersystem = None

    def equal(self, system: System) -> bool:
        """
        Compare content equality with another system.

        Uses Persistable's equality which compares serialized content
        recursively, ignoring non-copiable properties like IDs.

        Parameters
        ----------
        system : System
            System to compare with.

        Returns
        -------
        bool
            True if systems have identical content, False otherwise.

        Examples
        --------
        >>> sys1 = System(tag='test', name='Name')
        >>> sys2 = System(tag='test', name='Name')
        >>> sys1 == sys2  # ID-based equality
        False
        >>> sys1.equal(sys2)  # Content-based equality
        True

        Notes
        -----
        Distinction from ``==``:
        - ``sys1 == sys2`` uses Identifiable (compares IDs)
        - ``sys1.equal(sys2)`` uses Persistable (compares content)

        This is useful for comparing copies or checking if two systems have
        the same structure and data, regardless of their identities.

        See Also
        --------
        Persistable.__eq__ : Content comparison implementation
        Identifiable.__eq__ : ID comparison implementation
        """
        return Persistable.__eq__(self, system)

    # ---------- ---------- ---------- ---------- ---------- properties
    @property
    def supersystem(self) -> System|None:
        """
        Get parent system.

        Returns
        -------
        System or None
            Parent system, or None if this is a root system.

        Examples
        --------
        >>> root = System(tag='root')
        >>> child = System(tag='child')
        >>> root.add(child)
        >>> child.supersystem is root
        True
        >>> root.supersystem is None
        True
        """
        if self._supersystem_id is None:
            return None

        return Identifiable.get_instance(self._supersystem_id)

    @supersystem.setter
    def supersystem(self, value: System|None) -> None:
        """
        Set parent system.

        Parameters
        ----------
        value : System or None
            Parent system to register with, or None to unregister.

        Raises
        ------
        TypeError
            If value is not a System or None.
        ValueError
            If registration would create circular dependency or tag conflicts.

        Examples
        --------
        >>> parent = System(tag='parent')
        >>> child = System(tag='child')
        >>> child.supersystem = parent
        >>> child in parent
        True

        Notes
        -----
        Setting supersystem automatically:
        - Removes from old parent (if any)
        - Adds to new parent's subsystems dict
        - Sets up tag observer
        - Validates constraints
        """
        if not isinstance(value, System) and value is not None:
            raise TypeError(f'Expected System, got {type(value)}')

        self._supersystem_id = value.id if value is not None else None

    @property
    def supersystem_chain(self) -> list[System]:
        """
        Get list of all ancestors.

        Returns list of all parent systems from immediate parent to root,
        in order.

        Returns
        -------
        list[System]
            List of ancestors, nearest first.

        Examples
        --------
        >>> root = System(tag='root')
        >>> level1 = System(tag='level1')
        >>> level2 = System(tag='level2')
        >>> root.add(level1)
        >>> level1.add(level2)
        >>> level2.supersystem_chain
        [<System level1>, <System root>]

        Notes
        -----
        Walks the hierarchy upwards until reaching a root (supersystem is None).
        Used for circular dependency detection and hierarchy navigation.
        """
        chain = []

        system = self.supersystem

        while system is not None:
            chain.append(system)
            system = system.supersystem

        return chain
