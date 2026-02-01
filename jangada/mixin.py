#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import weakref

import uuid

from jangada.serialization import SerializableProperty

# ---------- ---------- ---------- ---------- ---------- ---------- typing
from typing import Any


# ========== ========== ========== ========== ========== Identifiable
class Identifiable:
    """
    Mixin class for objects with unique identifiers.

    Provides automatic UUID v4 generation and management for objects that
    require unique identification. IDs are write-once (immutable after first
    assignment), not copied when copying objects, and tracked in a global
    registry for instance lookup.

    The ID is lazily generated on first access - objects don't receive an ID
    until the `id` property is accessed or explicitly set. All instances are
    tracked in a weak reference registry to enable lookup by ID without
    preventing garbage collection.

    Attributes
    ----------
    id : str
        Unique identifier as a 32-character hexadecimal UUID v4 string.
        Automatically generated on first access if not explicitly set.
        Write-once: cannot be modified after initial assignment.
        Not copied: each copy receives a new ID.

    Examples
    --------
    Basic usage with automatic ID generation:

    >>> obj = Identifiable()
    >>> obj.id
    '02248d1fd3c14f3aa16cb1eb61d0d68e'
    >>> obj.id  # Same ID on subsequent access
    '02248d1fd3c14f3aa16cb1eb61d0d68e'

    Setting a custom ID before first access:

    >>> obj = Identifiable()
    >>> obj.id = 'f2efab5c19ee4df09d4baa541dc3436c'
    >>> obj.id
    'f2efab5c19ee4df09d4baa541dc3436c'

    IDs are immutable after assignment:

    >>> obj = Identifiable()
    >>> obj.id
    'f014267e9e28416290437e8e6c81cfd5'
    >>> obj.id = 'c83951b43bdb4b85ab3774dafa24de1a'
    Traceback (most recent call last):
        ...
    AttributeError: id is a write-once property and has already been set

    Looking up instances by ID:

    >>> obj = Identifiable()
    >>> obj_id = obj.id
    >>> retrieved = Identifiable.get_instance(obj_id)
    >>> retrieved is obj
    True

    Using in sets and dictionaries (hashable by ID):

    >>> obj1 = Identifiable()
    >>> obj2 = Identifiable()
    >>> unique_objects = {obj1, obj2}
    >>> len(unique_objects)
    2

    Notes
    -----
    This class uses :py:class:`SerializableProperty` with the following flags:

    - ``writeonce=True``: ID can only be set once, then becomes immutable
    - ``copiable=False``: When copying an object, a new ID is generated

    The instance registry uses :py:class:`weakref.WeakValueDictionary`, which
    means objects can be garbage collected even when in the registry. If an
    object is garbage collected, its ID will no longer be retrievable via
    :meth:`get_instance`.

    Warnings
    --------
    Although this class is not abstract, it is primarily designed to be used
    as a mixin rather than instantiated directly. It provides the ``id``
    property to any class that inherits from it.

    When using with serialization frameworks, ensure the ID is included in
    serialization but excluded from copying (which is handled automatically
    by the ``copiable=False`` flag).

    See Also
    --------
    Nameable : Mixin for objects with human-readable names
    Taggable : Mixin for objects with mnemonic tags
    Describable : Mixin for objects with descriptions
    SerializableProperty : Property descriptor used for ID management

    References
    ----------
    .. [1] RFC 4122 - A Universally Unique IDentifier (UUID) URN Namespace
           https://tools.ietf.org/html/rfc4122
    """

    _instances = weakref.WeakValueDictionary()

    id = SerializableProperty(copiable=False, writeonce=True, doc="""
    Unique identifier for the object.
    
    A 32-character hexadecimal string representing a UUID v4. Automatically
    generated on first access if not explicitly set. Once set (either
    automatically or manually), the ID cannot be changed.
    
    Type
    ----
    str
    
    Properties
    ----------
    - **Write-once**: Can be set only once, then becomes immutable
    - **Not copiable**: Copies receive new IDs rather than copying the original
    - **Lazy**: Generated only when first accessed, not at object creation
    - **Validated**: Must be a valid UUID v4 format if set manually
    
    Raises
    ------
    AttributeError
        If attempting to modify an already-set ID.
    ValueError
        If setting to an invalid UUID v4 format.
    
    Examples
    --------
    Automatic generation:
    
    >>> obj = Identifiable()
    >>> obj.id  # Generated on first access
    '550e8400e29b41d4a716446655440000'
    
    Manual setting (must be valid UUID v4):
    
    >>> obj = Identifiable()
    >>> obj.id = '123e4567e89b12d3a456426614174000'
    >>> obj.id
    '123e4567e89b12d3a456426614174000'
    
    Attempting to change ID fails:
    
    >>> obj.id = 'different-id'
    Traceback (most recent call last):
        ...
    AttributeError: id is a write-once property and has already been set
    """)

    @id.default
    def id(self) -> str:
        return uuid.uuid4().hex

    @id.parser
    def id(self, value: Any) -> str:
        try:
            return uuid.UUID(str(value), version=4).hex

        except ValueError as error:
            msg = f"Invalid UUID for id: {value!r}. Must be a valid UUID v4."
            raise ValueError(msg) from error

    @id.observer
    def id(self, old_value: str, new_value: str) -> None:
        Identifiable._instances[new_value] = self

    def __hash__(self) -> int:
        """
        Return hash of the object based on its ID.

        Allows Identifiable objects to be used in sets and as dictionary keys.
        Two objects with the same ID will have the same hash (though in
        practice, IDs should be unique across all instances).

        The hash is computed from the ID string. Note that accessing the hash
        will trigger ID generation if the ID has not yet been set.

        Returns
        -------
        int
            Hash value based on the object's ID.

        Examples
        --------
        Using Identifiable objects in sets:

        >>> obj1 = Identifiable()
        >>> obj2 = Identifiable()
        >>> obj3 = obj1  # Same object
        >>>
        >>> objects = {obj1, obj2, obj3}
        >>> len(objects)  # obj1 and obj3 are the same
        2

        Using as dictionary keys:

        >>> obj = Identifiable()
        >>> registry = {obj: "some data"}
        >>> registry[obj]
        'some data'

        Notes
        -----
        Calling this method will trigger ID generation if the ID has not
        already been accessed or set. This ensures objects always have
        consistent hash values.

        The hash is based solely on the ID, not on any other object
        attributes. This means the hash remains stable even if other
        attributes change (which is required for hashable objects).

        See Also
        --------
        __eq__ : Equality comparison (should be implemented by subclasses)
        """
        return hash(self.id)

    @classmethod
    def get_instance(cls, id_: str) -> Identifiable | None:
        """
        Retrieve an Identifiable instance by its ID.

        Looks up an object in the global instance registry using its unique
        identifier. Returns the object if it exists and has not been garbage
        collected, otherwise returns None.

        Parameters
        ----------
        id_ : str
            The 32-character hexadecimal UUID string to look up.

        Returns
        -------
        Identifiable or None
            The instance with the given ID, or None if no such instance
            exists or has been garbage collected.

        Examples
        --------
        Basic lookup:

        >>> obj = Identifiable()
        >>> obj_id = obj.id
        >>> retrieved = Identifiable.get_instance(obj_id)
        >>> retrieved is obj
        True

        Lookup of non-existent ID:

        >>> Identifiable.get_instance('00000000000000000000000000000000')
        None

        Object garbage collected:

        >>> obj = Identifiable()
        >>> obj_id = obj.id
        >>> del obj  # Object can be garbage collected
        >>> Identifiable.get_instance(obj_id)
        None

        Notes
        -----
        This method searches the global registry maintained by the
        Identifiable class. The registry uses weak references, so objects
        can be garbage collected even if they're in the registry.

        If an object has been garbage collected, its ID will no longer
        be in the registry and this method will return None.

        The lookup is O(1) as the registry is implemented as a dictionary.

        See Also
        --------
        id : The unique identifier property
        """
        return Identifiable._instances.get(id_)