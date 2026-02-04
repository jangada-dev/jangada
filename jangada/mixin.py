#  -*- coding: utf-8 -*-
"""
Mixins providing orthogonal identity, metadata, and state capabilities.

This module defines a collection of lightweight mixin classes that add
well-scoped, reusable properties to domain objects. Each mixin introduces
exactly one conceptual capability (identity, naming, tagging, description,
color, activation state) and is designed to compose cleanly with others.

The mixins in this module are intentionally:

- Orthogonal: each mixin addresses a single concern.
- Declarative: behavior is expressed via ``SerializableProperty`` descriptors.
- Non-invasive: no assumptions are made about object lifecycle, ownership,
  persistence strategy, or container semantics.
- Composable: mixins may be combined freely in user-defined classes.

Design philosophy
-----------------
These mixins are *capabilities*, not full abstractions. They do not enforce
global policies such as uniqueness, registration, or indexing. Instead,
they expose normalized, well-defined attributes that higher-level components
may interpret and organize as needed.

For example:

- ``Taggable`` provides a validated symbolic tag, but does not manage lookup.
- ``Identifiable`` provides a globally unique ID, but does not define equality
  semantics beyond hashing.
- ``Colorable`` provides a canonical color representation, but does not impose
  any rendering or styling logic.

This separation keeps the mixins predictable and avoids hidden coupling.

SerializableProperty integration
--------------------------------
All properties in this module are implemented using
:class:`jangada.serialization.SerializableProperty`. This ensures:

- Consistent normalization and validation on assignment.
- Explicit control over copy semantics (e.g. IDs are not copied).
- First-class support for serialization and persistence.

Each property documents its own validation rules and guarantees.

Overview of provided mixins
---------------------------

Identifiable
    Adds a globally unique, write-once UUID v4 identifier (``id``). Instances
    are tracked in a weak-reference registry for lookup by ID.

Taggable
    Adds a symbolic tag (``tag``) validated as a Python identifier, suitable
    for attribute-based access in registries or namespaces.

Nameable
    Adds a short, human-readable name (``name``) intended for display and
    labeling purposes.

Describable
    Adds a free-form description text (``description``) for extended context
    or documentation.

Colorable
    Adds a color attribute (``color``) stored as a canonical HTML hex string
    (``#RRGGBB``), suitable for visualization and UI styling.

Activatable
    Adds a boolean activation flag (``active``) indicating whether an object
    is enabled or in effect.

Composition
-----------
These mixins are designed to be freely combined:

>>> class Asset(Identifiable, Taggable, Nameable, Colorable, Activatable):
...     pass

No ordering constraints are imposed, and no mixin assumes the presence of
another unless explicitly documented.

Notes
-----
- None of the mixins are abstract, but they are primarily intended to be used
  via inheritance rather than instantiated directly.
- Validation logic is intentionally conservative and explicit; higher-level
  policies (e.g. uniqueness of tags, strict boolean parsing) should be enforced
  by containers or application logic.

See Also
--------
jangada.serialization.SerializableProperty
    Descriptor used to implement all mixin properties.

TagNamespace
    Namespace-like container that can expose Taggable objects by their tags.
"""

from __future__ import annotations

import keyword
import uuid

from weakref import WeakValueDictionary

from matplotlib.colors import to_hex

from jangada.serialization import SerializableProperty

# ---------- ---------- ---------- ---------- ---------- ---------- typing
from typing import Any


# ========== ========== ========== ========== ========== Identifiable
class Identifiable:
    """
    Mixin that adds a globally unique, write-once UUID identifier.

    Provides each instance with a unique ID that cannot be changed after
    initialization. Instances are tracked in a weak-reference registry,
    allowing lookup by ID while permitting garbage collection.

    The ID is a UUID v4 (randomly generated) stored as a 32-character
    hexadecimal string. It is not included when copying objects (copiable=False).

    Attributes
    ----------
    id : str
        A unique 32-character hexadecimal UUID v4 identifier. Write-once,
        auto-generated if not provided, and not copied.

    Class Attributes
    ----------------
    _instances : WeakValueDictionary[str, Identifiable]
        Global registry of instances indexed by ID. Uses weak references
        to allow garbage collection.

    Examples
    --------
    Basic usage::

        class Component(Identifiable):
            pass

        comp = Component()
        print(comp.id)  # '3a5f8e2c1b9d4f7a...'

    Lookup by ID::

        comp_id = comp.id
        retrieved = Identifiable.get_instance(comp_id)
        assert retrieved is comp

    Hash and equality based on ID::

        comp1 = Component()
        comp2 = Component()

        # Different IDs
        assert comp1 != comp2
        assert hash(comp1) != hash(comp2)

        # Can use in sets and dicts
        components = {comp1, comp2}

    Write-once protection::

        comp = Component()
        original_id = comp.id

        # Cannot change ID
        comp.id = 'different-id'  # Raises AttributeError

    Notes
    -----
    **Equality Semantics**
        Two Identifiable objects are equal if they have the same ID.
        This implements value equality; for identity equality use ``is``.
        Subclasses can override ``__eq__`` for content-based equality.

    **Hash Consistency**
        Hash is based solely on ID, ensuring objects can be used in sets
        and as dictionary keys. Hash remains consistent even if other
        attributes change.

    **Weak References**
        The registry uses weak references, so instances can be garbage
        collected even while registered. After garbage collection,
        ``get_instance()`` will return None for that ID.

    **Copy Behavior**
        The ID is not copied when using ``copy.copy()`` or serialization
        with ``is_copy=True``. Each copy gets a new, unique ID.

    **Thread Safety**
        The registry is not thread-safe. Concurrent access from multiple
        threads may require external synchronization.

    See Also
    --------
    Taggable : Symbolic identifier for namespace access
    SerializableProperty : Property descriptor used for id
    """

    _instances: WeakValueDictionary[str, Identifiable] = WeakValueDictionary()

    id: str = SerializableProperty(copiable=False, writeonce=True, doc="""
        Globally unique identifier (UUID v4).
        
        A 32-character hexadecimal UUID v4 string that uniquely identifies this instance.
        Automatically generated on first access if not explicitly provided. Write-once
        (cannot be changed after initialization) and non-copiable (each copy gets a new ID).
        
        Type
        ----
        str
        
        Default
        -------
        Auto-generated UUID v4 hex string (e.g., '3a5f8e2c1b9d4f7a...')
        
        Constraints
        -----------
        - Must be a valid UUID v4
        - Write-once: cannot be changed after first set
        - Non-copiable: excluded from copy/serialization
        - Automatically registers instance in global weak-reference registry
        
        Examples
        --------
        Automatic generation::
        
            obj = Identifiable()
            print(obj.id)  # '3a5f8e2c1b9d4f7a0b1c2d3e4f5a6b7c'
        
        Explicit setting (during deserialization)::
        
            # ID is validated and normalized
            obj.id = 'A1B2C3D4-E5F6-4789-ABCD-EF0123456789'
        
        Immutability::
        
            obj = Identifiable()
            obj.id = 'different-id'  # Raises AttributeError
        
        Lookup by ID::
        
            obj_id = obj.id
            retrieved = Identifiable.get_instance(obj_id)
            assert retrieved is obj
        
        Notes
        -----
        The ID serves as the basis for ``__hash__()`` and ``__eq__()``, enabling
        instances to be used in sets and as dictionary keys. Objects with the same
        ID are considered equal.
        
        See Also
        --------
        Identifiable.get_instance : Retrieve instance by ID
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

    @id.postinitializer
    def id(self) -> None:
        Identifiable._instances[self.id] = self

    def __hash__(self) -> int:
        """
        Compute hash based on ID.

        Returns
        -------
        int
            Hash of the ID string.

        Notes
        -----
        Hash is consistent and based only on ID, allowing objects to be
        used in sets and as dictionary keys even if other attributes change.
        """
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        """
        Compare equality based on ID.

        Two Identifiable objects are equal if they have the same ID.

        Parameters
        ----------
        other : Any
            Object to compare with.

        Returns
        -------
        bool
            True if other is Identifiable with same ID, False otherwise.
            Returns NotImplemented for non-Identifiable objects.

        Notes
        -----
        This implements value equality. For identity equality (same object),
        use the ``is`` operator.

        Subclasses may override this for content-based equality while still
        inheriting ID-based hashing.
        """
        if not isinstance(other, Identifiable):
            return NotImplemented
        return self.id == other.id

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
    Mixin that adds a symbolic tag for namespace access.

    Provides a validated tag attribute that must be a valid Python identifier
    (but not a keyword). Tags are intended for programmatic, attribute-style
    access in namespaces or registries, similar to how pandas DataFrames expose
    columns.

    Unlike names, tags have strict validation rules to ensure they can be
    used as Python identifiers. They are mutable, allowing dynamic reorganization.

    Attributes
    ----------
    tag : str or None
        A symbolic identifier string. Must be a valid Python identifier
        (alphanumeric and underscore, not starting with digit) and not
        a Python keyword. Can be None.

    Examples
    --------
    Valid tags::

        obj = Taggable()
        obj.tag = "sensor_a"
        obj.tag = "temp_sensor_01"
        obj.tag = "_private"

    Invalid tags (will raise ValueError)::

        obj.tag = "123invalid"      # Starts with digit
        obj.tag = "invalid-tag"     # Contains hyphen
        obj.tag = "invalid tag"     # Contains space
        obj.tag = "if"              # Python keyword

    Namespace-style access (intended use case)::

        class System:
            def __init__(self):
                self._components = {}

            def add(self, component):
                self._components[component.tag] = component

            def __getattr__(self, tag):
                return self._components.get(tag)

        system = System()

        sensor = Taggable()
        sensor.tag = "temp_sensor"
        system.add(sensor)

        # Attribute-style access
        assert system.temp_sensor is sensor

    Mutability::

        obj = Taggable()
        obj.tag = "first_tag"
        obj.tag = "second_tag"  # Can change
        obj.tag = None          # Can reset

    Notes
    -----
    **Validation Rules**
        - Must be a valid Python identifier: ``str.isidentifier()``
        - Cannot be a Python keyword (``if``, ``for``, ``class``, etc.)
        - Leading/trailing whitespace is stripped before validation
        - Empty string (or whitespace-only) is invalid

    **Uniqueness**
        Tags are not globally unique. Uniqueness should be enforced at the
        container level (e.g., within a specific namespace or system).

    **Mutability**
        Tags are mutable to allow dynamic reorganization. Containers that
        index by tag must handle tag changes appropriately.

    **Use Cases**
        - Component identification in hierarchical systems
        - Attribute-style namespace access
        - Symbolic references in configuration files
        - Human-readable identifiers in logs and debugging

    See Also
    --------
    Identifiable : Globally unique identifier
    Nameable : Human-readable display name
    """

    tag: str|None = SerializableProperty(doc="""
        Symbolic identifier for namespace-style access.
        
        A validated string that must be a valid Python identifier (but not a keyword).
        Intended for programmatic, attribute-style access in containers and namespaces,
        similar to how pandas DataFrames expose columns.
        
        Type
        ----
        str or None
        
        Default
        -------
        None
        
        Constraints
        -----------
        - Must be a valid Python identifier: ``str.isidentifier()``
        - Cannot be a Python keyword (``if``, ``for``, ``class``, etc.)
        - Leading/trailing whitespace is automatically stripped
        - Empty strings are invalid (ValueError)
        - Case-sensitive
        
        Examples
        --------
        Valid tags::
        
            obj.tag = "sensor_a"
            obj.tag = "temp_sensor_01"
            obj.tag = "_private"
        
        Invalid tags::
        
            obj.tag = "123invalid"      # Starts with digit
            obj.tag = "invalid-tag"     # Contains hyphen
            obj.tag = "invalid tag"     # Contains space
            obj.tag = "if"              # Python keyword
            obj.tag = ""                # Empty string
        
        Namespace access pattern::
        
            # Container provides attribute-style access by tag
            system.sensor_a  # Returns object with tag='sensor_a'
        
        Whitespace normalization::
        
            obj.tag = "  my_tag  "
            assert obj.tag == "my_tag"
        
        Notes
        -----
        Tags are mutable (can be changed) to support dynamic reorganization of
        namespaces. Uniqueness is not enforced globally - containers should manage
        tag uniqueness within their scope.
        
        Use tags for programmatic access, not display. For human-readable labels,
        use the ``name`` property from Nameable.
        
        See Also
        --------
        Nameable.name : Human-readable display name
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


# ========== ========== ========== ========== ========== Nameable
class Nameable:
    """
    Mixin that adds a human-readable name.

    Provides a name attribute for display and labeling purposes. Names are
    less restrictive than tags - they can contain spaces, special characters,
    and non-ASCII characters. They are intended for user-facing contexts like
    UI labels, reports, and logs.

    Names are normalized by stripping whitespace and converting empty strings
    to None.

    Attributes
    ----------
    name : str or None
        A human-readable name string. Can contain any characters.
        Leading/trailing whitespace is stripped. Empty strings become None.

    Examples
    --------
    Basic usage::

        obj = Nameable()
        obj.name = "Temperature Sensor"
        print(obj.name)  # 'Temperature Sensor'

    Special characters allowed::

        obj.name = "Sensor #1 (Main)"
        obj.name = "Test-Case-A"
        obj.name = "Tëst Nämé"
        obj.name = "测试名称"

    Normalization::

        obj.name = "  Name  "
        print(obj.name)  # 'Name' (whitespace stripped)

        obj.name = "   "
        print(obj.name)  # None (empty after stripping)

        obj.name = ""
        print(obj.name)  # None

    Type conversion::

        obj.name = 123
        print(obj.name)  # '123'

        obj.name = 3.14
        print(obj.name)  # '3.14'

    Notes
    -----
    **Validation**
        Minimal validation is performed. Any value is converted to string
        and stripped. Empty results become None.

    **Use Cases**
        - UI labels and display text
        - Report headers
        - Log messages
        - User-facing identifiers
        - Documentation titles

    **Comparison with Tag**
        Tags are for programmatic access (strict validation), names are
        for human display (loose validation). Objects often have both::

            component.tag = "temp_sensor_01"  # For code
            component.name = "Temperature Sensor #1"  # For UI

    **Length**
        No length restrictions are imposed. Very long names are allowed
        but may need truncation for display purposes.

    See Also
    --------
    Taggable : Symbolic identifier for namespace access
    Describable : Extended description text
    """
    name: str = SerializableProperty(doc="""
        Human-readable display name.
        
        A string for labeling and display purposes. Less restrictive than tags -
        can contain spaces, special characters, and Unicode. Intended for user-facing
        contexts like UI labels, reports, and logs.
        
        Type
        ----
        str or None
        
        Default
        -------
        None
        
        Constraints
        -----------
        - Any characters allowed (spaces, punctuation, Unicode)
        - Leading/trailing whitespace is automatically stripped
        - Empty strings (or whitespace-only) normalize to None
        - Converted to string if other type provided
        - No length restrictions
        
        Examples
        --------
        Basic usage::
        
            obj.name = "Temperature Sensor"
            obj.name = "Sensor #1 (Main)"
            obj.name = "Test-Case-A"
        
        Unicode support::
        
            obj.name = "Tëst Nämé"
            obj.name = "测试名称"
            obj.name = "📊 Data Dashboard"
        
        Normalization::
        
            obj.name = "  Name  "
            assert obj.name == "Name"
            
            obj.name = "   "
            assert obj.name is None
            
            obj.name = ""
            assert obj.name is None
        
        Type conversion::
        
            obj.name = 123
            assert obj.name == "123"
        
        Notes
        -----
        Use names for display and human consumption. For programmatic identifiers
        that work in attribute access, use tags from Taggable.
        
        Names are mutable and have no uniqueness constraints. Very long names
        may need truncation for display purposes.
        
        See Also
        --------
        Taggable.tag : Symbolic identifier for namespace access
        Describable.description : Extended description text
        """)

    @name.parser
    def name(self, value: Any) -> str|None:
        if value is None:
            return None

        value = str(value).strip()

        return value if value else None


# ========== ========== ========== ========== ========== Describable
class Describable:
    """
    Mixin that adds a free-form description.

    Provides a description attribute for extended context, documentation, or
    explanatory text. Descriptions can be multiline and have no length
    restrictions. They are intended for detailed information that doesn't
    fit in a short name.

    Like names, descriptions are normalized by stripping whitespace and
    converting empty strings to None.

    Attributes
    ----------
    description : str or None
        Free-form description text. Can be multiline and any length.
        Leading/trailing whitespace is stripped. Empty strings become None.

    Examples
    --------
    Basic usage::

        obj = Describable()
        obj.description = "This sensor monitors ambient temperature."
        print(obj.description)

    Multiline descriptions::

        obj.description = '''
        This is a multiline description.
        It can span multiple lines.
        Useful for detailed documentation.
        '''

    Normalization (same as Nameable)::

        obj.description = "  Text  "
        print(obj.description)  # 'Text'

        obj.description = ""
        print(obj.description)  # None

    Long descriptions::

        obj.description = "A" * 10000  # No length limit
        print(len(obj.description))  # 10000

    Notes
    -----
    **Use Cases**
        - Detailed documentation
        - Usage instructions
        - Configuration notes
        - Context for debugging
        - Help text in UIs

    **Formatting**
        No formatting is applied. If you need formatted text (markdown,
        HTML, etc.), store it as a string and format at display time.

    **Length**
        No restrictions. Very long descriptions are allowed. Consider
        pagination or truncation in UI contexts.

    See Also
    --------
    Nameable : Short display name
    """
    description: str = SerializableProperty(doc="""
        Free-form descriptive text.
        
        Extended context, documentation, or explanatory information. Can be multiline
        and arbitrarily long. Intended for detailed information that doesn't fit in
        a short name.
        
        Type
        ----
        str or None
        
        Default
        -------
        None
        
        Constraints
        -----------
        - Any characters allowed (including newlines)
        - Leading/trailing whitespace is automatically stripped
        - Empty strings (or whitespace-only) normalize to None
        - Converted to string if other type provided
        - No length restrictions
        
        Examples
        --------
        Basic usage::
        
            obj.description = "This sensor monitors ambient temperature."
        
        Multiline text::
        
            obj.description = '''
            This is a detailed description.
            It can span multiple lines.
            Useful for documentation.
            '''
        
        Very long text::
        
            obj.description = "Long documentation..." * 1000  # No limit
        
        Normalization (same as name)::
        
            obj.description = "  Text  "
            assert obj.description == "Text"
            
            obj.description = ""
            assert obj.description is None
        
        Notes
        -----
        Descriptions are for extended information - use name for short labels.
        No formatting is applied - if you need formatted text (markdown, HTML),
        store it as a string and format at display time.
        
        Common use cases: detailed documentation, usage instructions, configuration
        notes, debugging context, help text in UIs.
        
        See Also
        --------
        Nameable.name : Short display name
        """)

    @description.parser
    def description(self, value: Any) -> str|None:
        if value is None:
            return None

        value = str(value).strip()

        return value if value else None


# ========== ========== ========== ========== ========== Colorable
class Colorable:
    """
    Mixin that adds a color attribute for visualization.

    Provides a color attribute stored as a canonical HTML hex string
    (``#RRGGBB`` format). Accepts various color formats via matplotlib's
    color parsing, including hex strings, RGB tuples, and named colors.

    The default color is matplotlib's C0 (``#1F77B4``), a pleasant blue.

    Attributes
    ----------
    color : str
        Color as uppercase HTML hex string (``#RRGGBB`` format).
        Default is ``'#1F77B4'`` (matplotlib C0 blue).

    Properties
    ----------
    color_rgb : tuple[int, int, int]
        Read-only property returning color as RGB tuple with values 0-255.

    Examples
    --------
    Default color::

        obj = Colorable()
        print(obj.color)  # '#1F77B4'

    Set from hex string::

        obj.color = '#FF0000'
        print(obj.color)  # '#FF0000' (red)

    Set from RGB tuple::

        obj.color = (1.0, 0.0, 0.0)  # Float format (0-1)
        print(obj.color)  # '#FF0000'

    Set from color name::

        obj.color = 'red'
        print(obj.color)  # '#FF0000'

    Get as RGB tuple::

        obj.color = '#FF0000'
        print(obj.color_rgb)  # (255, 0, 0)

    Case normalization::

        obj.color = '#ff0000'
        print(obj.color)  # '#FF0000' (uppercase)

    Notes
    -----
    **Color Formats**
        Input accepts any format that matplotlib's ``to_hex()`` understands:

        - Hex strings: ``'#FF0000'``, ``'#F00'`` (short form)
        - RGB tuples: ``(1.0, 0.0, 0.0)`` or ``(255, 0, 0)``
        - Named colors: ``'red'``, ``'blue'``, etc.
        - Matplotlib colors: ``'C0'``, ``'C1'``, etc.

    **Output Format**
        Always stored as uppercase ``#RRGGBB`` (6 hex digits).
        This canonical format ensures consistency.

    **Matplotlib Dependency**
        Requires matplotlib for color parsing. Consider making this
        optional or providing a lightweight fallback in future versions.

    **Use Cases**
        - Visualization and plotting
        - UI styling and themes
        - Legend entries
        - Color-coded categories

    See Also
    --------
    matplotlib.colors.to_hex : Color conversion function used
    """
    color: str = SerializableProperty(default='#1F77B4', doc="""
        Color for visualization and styling.
        
        Canonical HTML hex string (``#RRGGBB`` format) for visual representation.
        Accepts various input formats via matplotlib's color parsing, including hex
        strings, RGB tuples, and named colors. Always stored in uppercase hex format.
        
        Type
        ----
        str
        
        Default
        -------
        '#1F77B4' (matplotlib C0 blue)
        
        Constraints
        -----------
        - Output always uppercase ``#RRGGBB`` format (6 hex digits)
        - Input accepts any matplotlib-compatible color format
        - Validated via matplotlib's color parser
        - Alpha/transparency not supported
        
        Input Formats
        -------------
        - Hex strings: ``'#FF0000'``, ``'#F00'`` (short form)
        - RGB tuples: ``(1.0, 0.0, 0.0)`` (floats 0-1) or ``(255, 0, 0)`` (ints 0-255)
        - Color names: ``'red'``, ``'blue'``, ``'green'``
        - Matplotlib colors: ``'C0'``, ``'C1'``, ``'tab:blue'``
        - Grayscale: ``'0.5'`` (float string)
        
        Examples
        --------
        Hex string::
        
            obj.color = '#FF0000'
            assert obj.color == '#FF0000'
        
        RGB tuple::
        
            obj.color = (1.0, 0.0, 0.0)  # Float format
            assert obj.color == '#FF0000'
        
        Color name::
        
            obj.color = 'red'
            assert obj.color == '#FF0000'
        
        Case normalization::
        
            obj.color = '#ff0000'
            assert obj.color == '#FF0000'  # Uppercase
        
        Get as RGB tuple::
        
            obj.color = '#FF0000'
            assert obj.color_rgb == (255, 0, 0)
        
        Notes
        -----
        Requires matplotlib for color parsing. The canonical uppercase ``#RRGGBB``
        format ensures consistency across the system and compatibility with most
        graphics APIs and web standards.
        
        Use ``color_rgb`` property for integer RGB values (0-255) when needed by
        graphics libraries.
        
        See Also
        --------
        Colorable.color_rgb : Read-only RGB tuple property
        matplotlib.colors.to_hex : Underlying color conversion function
        """)

    @color.parser
    def color(self, value: Any) -> str:
        return to_hex(value).upper()

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        """
        Get color as RGB tuple.

        Returns
        -------
        tuple[int, int, int]
            RGB color with values in range 0-255.

        Examples
        --------
        >>> obj = Colorable()
        >>> obj.color = '#FF0000'
        >>> obj.color_rgb
        (255, 0, 0)
        """
        hex_color = self.color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# ========== ========== ========== ========== ========== Activatable
class Activatable:
    """
    Mixin that adds an activation state flag.

    Provides a boolean ``active`` attribute indicating whether an object is
    enabled, in effect, or otherwise "active". This is useful for objects
    that can be toggled on/off without destroying them.

    The attribute only accepts strict boolean values (``True`` or ``False``),
    not truthy/falsy values like integers or strings.

    Attributes
    ----------
    active : bool
        Boolean flag indicating activation state. Default is ``True``.
        Only accepts actual ``bool`` instances, not truthy/falsy values.

    Examples
    --------
    Basic usage::

        obj = Activatable()
        print(obj.active)  # True (default)

        obj.active = False
        print(obj.active)  # False

    Toggling state::

        obj.active = not obj.active  # Toggle

    Strict boolean validation::

        obj = Activatable()
        obj.active = True   # ✓ OK
        obj.active = False  # ✓ OK
        obj.active = 1      # ✗ TypeError
        obj.active = 0      # ✗ TypeError
        obj.active = "true" # ✗ TypeError

    Conditional logic::

        if obj.active:
            process(obj)
        else:
            skip(obj)

    Notes
    -----
    **Strict Validation**
        Only ``bool`` instances (``True``/``False``) are accepted. This
        prevents bugs from truthy/falsy values.

    **Use Cases**
        - Enable/disable components
        - Toggle features on/off
        - Mark objects as temporarily inactive
        - Filtering active vs inactive items

    **State vs Deletion**
        Use ``active`` for temporary enable/disable. For permanent removal,
        delete the object instead.

    See Also
    --------
    SerializableProperty : Property descriptor used for active
    """
    active: bool = SerializableProperty(default=True, doc="""
        Activation state flag.
        
        Boolean indicating whether the object is enabled, active, or in effect.
        Useful for objects that can be toggled on/off without being destroyed.
        Strictly validates boolean type - does not accept truthy/falsy values.
        
        Type
        ----
        bool
        
        Default
        -------
        True
        
        Constraints
        -----------
        - Must be exactly ``True`` or ``False`` (strict boolean)
        - No truthy/falsy coercion (1, 0, "", [], None are rejected)
        - Raises TypeError for non-boolean values
        
        Examples
        --------
        Basic usage::
        
            obj.active = True
            obj.active = False
        
        Toggling::
        
            obj.active = not obj.active
        
        Strict validation::
        
            obj.active = True   # ✓ OK
            obj.active = False  # ✓ OK
            obj.active = 1      # ✗ TypeError
            obj.active = 0      # ✗ TypeError
            obj.active = "true" # ✗ TypeError
            obj.active = None   # ✗ TypeError
            obj.active = []     # ✗ TypeError
        
        Conditional logic::
        
            if obj.active:
                process(obj)
            
            active_objects = [o for o in objects if o.active]
        
        Notes
        -----
        Strict boolean validation prevents subtle bugs from truthy/falsy values.
        For example, ``"false"`` would be truthy in Python, causing unexpected
        behavior if coercion were allowed.
        
        Use ``active`` for temporary enable/disable. For permanent removal, delete
        the object instead.
        
        The active state persists through serialization, so disabled objects remain
        disabled after save/load cycles.
        """)

    @active.parser
    def active(self, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("Active must be a boolean.")

        return value


__all__ = [
    'Identifiable',
    'Taggable',
    'Nameable',
    'Describable',
    'Activatable',
    'Colorable',
]
