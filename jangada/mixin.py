#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import keyword
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


# ========== ========== ========== ========== ========== Taggable
class Taggable:
    """
    Mixin class providing a user-defined symbolic tag.

    ``Taggable`` represents an object that can be assigned a symbolic label
    (``tag``) intended to be used as an external reference by container or
    registry objects (e.g. ``TagNamespace``).

    The tag itself has no intrinsic behavior beyond validation and storage.
    In particular, a ``Taggable`` does not automatically register itself
    anywhere, nor does it manage uniqueness or lookup.

    The tag is designed to be compatible with Python attribute access, so that
    container objects may expose tagged members using ``container.<tag>``
    syntax.

    Attributes
    ----------
    tag : str or None
        Optional symbolic tag identifying the object. If None, the object is
        considered untagged and cannot be resolved by tag-based lookup.

    Tag validation rules
    --------------------
    When assigned, the tag value is validated as follows:

    - ``None`` is accepted and represents an untagged object.
    - The value is coerced to ``str`` and stripped of leading/trailing
      whitespace.
    - The resulting string must be a valid Python identifier
      (``str.isidentifier()``).
    - The string must not be a Python keyword (``keyword.iskeyword``).

    These rules ensure that valid tags may be safely exposed via attribute
    access on container objects.

    Notes
    -----
    - ``None`` is treated as a special value meaning "no tag".
    - The empty string is not allowed.
    - Unicode identifiers are allowed if they satisfy
      ``str.isidentifier()`` and are not keywords.

    Tag lifecycle
    -------------
    ``Taggable`` does not enforce immutability of the tag. The tag may be
    changed at any time.

    However, container objects such as ``TagNamespace`` are *not required*
    to observe or react to tag changes. Changing the tag of a registered
    object does not automatically update any external registries or views.

    If a container requires consistency after a tag change, the object must
    be explicitly re-registered or refreshed according to that container's
    rules.

    Examples
    --------
    >>> obj = Taggable()
    >>> obj.tag = "satellite_1"
    >>> obj.tag
    'satellite_1'

    >>> obj.tag = None
    >>> obj.tag is None
    True

    >>> obj.tag = "class"
    Traceback (most recent call last):
        ...
    ValueError: Invalid tag: 'class'. Must not be a Python keyword.
    """

    tag = SerializableProperty(doc="""
        Symbolic identifier used for external referencing.

        The ``tag`` is intended to act as a stable, human-chosen name for an
        object, allowing other objects or containers to refer to it by a
        meaningful string. Most commonly, a container such as ``TagNamespace``
        uses this value to expose members by attribute access:

        - ``namespace[tag]`` (authoritative lookup)
        - ``namespace.<tag>`` (convenience lookup for identifier-safe tags)

        Parameters
        ----------
        value : str or None
            The proposed tag value.

        Returns
        -------
        str or None
            Normalized tag, or None if untagged.

        Validation Rules
        ----------------
        - ``None`` is accepted and represents an untagged object.
        - Non-None values are coerced to ``str`` and stripped of whitespace.
        - The resulting string must satisfy ``str.isidentifier()``.
        - The string must not be a Python keyword (``keyword.iskeyword``).

        Notes
        -----
        - Tags are validated to ensure they are compatible with Python
          attribute access, which enables idioms like ``namespace.<tag>``.
        - Objects with ``tag is None`` may still be registered in a namespace,
          but are not resolvable by tag-based lookup until a non-None tag is set.
        - Changing the tag does not automatically update external containers
          unless those containers explicitly handle tag updates.
        """)

    @tag.parser
    def tag(self, value: str|None) -> str|None:
        if value is None:
            return None

        value = str(value).strip()

        if not value.isidentifier():
            raise ValueError(f"Invalid tag: {value!r}. Must be a valid Python identifier.")

        if keyword.iskeyword(value):
            raise ValueError(f"Invalid tag: {value!r}. Must not be a Python keyword.")

        return value


class TagNamespace:
    """
    Namespace-like container exposing ``Taggable`` objects by their tags.

    ``TagNamespace`` provides a lightweight, dynamic view over a collection
    of ``Taggable`` objects, allowing them to be accessed by their tag using
    either dictionary-style or attribute-style lookup.

    This class does not own the objects it references, nor does it impose
    constraints on tag mutation after registration. It is intentionally
    passive and non-reactive.

    Conceptual model
    ----------------
    ``TagNamespace`` acts as a *resolver*, not an index:

    - It stores references to ``Taggable`` objects.
    - Lookup by tag is performed dynamically at access time.
    - No internal indexing is maintained.
    - Tag changes on registered objects are not tracked automatically.

    This design prioritizes correctness, simplicity, and explicit lifecycle
    control over lookup performance.

    Lookup semantics
    ----------------
    Two equivalent lookup mechanisms are provided:

    - ``namespace[tag]`` (authoritative, always works)
    - ``namespace.<tag>`` (convenience, best-effort)

    Attribute-based access is only possible for tags that are valid Python
    identifiers and do not collide with existing attributes or methods.

    Storage model
    -------------
    Internally, registered objects are stored in insertion order in a list.
    Lookup by tag is therefore O(n).

    This choice avoids issues with mutable tags and hashing invariants, and
    ensures predictable behavior even when tags are assigned or changed
    after registration.

    Notes
    -----
    - Multiple objects may share the same tag. In such cases, lookup returns
      the first matching object in registration order.
    - Objects with ``tag is None`` may be registered, but are not resolvable
      by tag-based lookup.
    - Changing the tag of a registered object does not update the namespace.
      Re-registration is required if consistency is needed.

    Examples
    --------
    >>> ns = TagNamespace()
    >>> a = Taggable()
    >>> b = Taggable()
    >>> a.tag = "alpha"
    >>> b.tag = "beta"
    >>> ns.register(a)
    >>> ns.register(b)

    >>> ns["alpha"] is a
    True

    >>> ns.beta is b
    True
    """
    # ========== ========== ========== ========== ========== class attributes
    ...

    # ========== ========== ========== ========== ========== special methods
    def __init__(self) -> None:
        """
        Initialize an empty TagNamespace.

        The namespace starts with no registered objects.
        """
        self._taggable: list[Taggable] = []

    def __getitem__(self, tag: str) -> Taggable:
        """
        Retrieve a registered object by its tag.

        Parameters
        ----------
        tag : str
         Tag identifying the desired object.

        Returns
        -------
        Taggable
         The first registered object whose ``tag`` equals the given value.

        Raises
        ------
        KeyError
         If no registered object has the given tag.

        Notes
        -----
        This is the authoritative lookup mechanism and works for all tags,
        including those that are not valid Python identifiers.
        """
        for taggable in self._taggable:
            if taggable.tag == tag:
                return taggable

        raise KeyError(f"No taggable registered with tag '{tag}'.")

    def __getattr__(self, tag: str) -> Taggable:
        """
        Retrieve a registered object by attribute-style tag lookup.

        Parameters
        ----------
        tag : str
            Attribute name corresponding to a tag.

        Returns
        -------
        Taggable
            The resolved object.

        Raises
        ------
        AttributeError
            If no object with the given tag is registered.

        Notes
        -----
        This method is only invoked if normal attribute lookup fails.
        It provides syntactic convenience and mirrors the behavior of
        libraries such as pandas.

        If a tag collides with an existing attribute or method name,
        attribute access resolves to the attribute, not the tag.
        """
        try:
            return self[tag]
        except KeyError as error:
            raise AttributeError(f"No tag '{tag}' registered.") from error

    def __contains__(self, reference: Taggable|str) -> bool:
        """
        Test whether an object or tag is registered.

        Parameters
        ----------
        reference : Taggable or str
            Either a Taggable instance or a tag string.

        Returns
        -------
        bool
            True if the object is registered or if a matching tag exists.

        Notes
        -----
        When testing membership by tag string, the result is True if at least
        one registered object has the given tag.
        """
        if isinstance(reference, Taggable):
            return reference in self._taggable

        return reference in (taggable.tag for taggable in self._taggable)

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    ...

    # ========== ========== ========== ========== ========== public methods
    def register(self, taggable: Taggable) -> None:
        """
        Register a Taggable object in the namespace.

        Parameters
        ----------
        taggable : Taggable
            Object to register.

        Raises
        ------
        ValueError
            If the object is already registered.

        Notes
        -----
        - Objects may be registered even if ``taggable.tag is None``.
        - No validation of tag uniqueness is performed.
        - Registration does not freeze or lock the object's tag.
        """
        if taggable in self:
            raise ValueError(f"Taggable object already registered: {taggable!r}.")

        self._taggable.append(taggable)

    def unregister(self, reference: Taggable|str) -> None:
        """
        Unregister a Taggable object from the namespace.

        Parameters
        ----------
        reference : Taggable or str
            Either the object itself or its tag.

        Raises
        ------
        ValueError
            If the object is not registered or if no object matches the tag.

        Notes
        -----
        When a string is provided, the first object with the matching tag is
        removed.
        """
        if isinstance(reference, str):
            reference = self[reference]

        try:
            self._taggable.remove(reference)

        except ValueError as error:
            raise ValueError(f"Taggable object not registered: {reference!r}.") from error

    # ---------- ---------- ---------- ---------- ---------- properties
    ...


# ========== ========== ========== ========== ========== Nameable
class Nameable:

    name = SerializableProperty()

