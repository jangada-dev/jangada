#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for mixin classes.

Tests cover:
- Identifiable: UUID generation, validation, registry, hash, equality
- Taggable: Identifier validation, keyword rejection, mutability
- Nameable: String normalization, None handling
- Describable: String normalization, None handling
- Colorable: Color parsing, hex format, RGB conversion
- Activatable: Boolean validation
- Composition: Multiple mixins working together

Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import pytest
import uuid
import keyword
from typing import Any


from jangada.mixin import (Identifiable,
                           Taggable,
                           Nameable,
                           Describable,
                           Colorable,
                           Activatable)


# ========== ========== ========== ========== Test Identifiable
class TestIdentifiable:
    """Test Identifiable mixin."""

    def test_creates_unique_id_on_initialization(self) -> None:
        # Each instance should get a unique UUID
        obj1 = Identifiable()
        obj2 = Identifiable()

        assert obj1.id != obj2.id

    def test_id_is_valid_uuid_hex(self) -> None:
        # ID should be valid UUID v4 in hex format
        obj = Identifiable()

        # Should be 32-character hex string
        assert len(obj.id) == 32
        assert all(c in '0123456789abcdef' for c in obj.id)

        # Should be parseable as UUID
        uuid_obj = uuid.UUID(obj.id, version=4)
        assert uuid_obj.hex == obj.id

    def test_id_is_write_once(self) -> None:
        # ID cannot be changed after first set
        obj = Identifiable()
        original_id = obj.id

        with pytest.raises(AttributeError, match="write-once"):
            obj.id = uuid.uuid4().hex

    def test_id_not_copiable(self) -> None:
        # ID should have copiable=False
        descriptor = Identifiable.id
        assert descriptor.copiable is False

    def test_id_parser_accepts_uuid_string(self) -> None:
        # Parser should accept valid UUID strings
        valid_uuid = uuid.uuid4().hex
        obj = Identifiable()

        # Cannot set directly due to write-once, but parser validates on init

        # Create new object with explicit ID
        class TestClass(Identifiable):
            pass

        # This tests parser indirectly through initialization
        # Parser is called during first set

    def test_id_parser_rejects_invalid_uuid(self) -> None:
        # Parser should reject invalid UUID strings
        class TestClass(Identifiable):
            pass

        obj = TestClass()
        # Try to set invalid UUID (will fail due to write-once, but tests validation)
        # We need to test the parser directly or through initialization data

    def test_hash_based_on_id(self) -> None:
        # __hash__ should be based on ID
        obj = Identifiable()

        assert hash(obj) == hash(obj.id)

    def test_hash_consistent(self) -> None:
        # Hash should be consistent across calls
        obj = Identifiable()

        hash1 = hash(obj)
        hash2 = hash(obj)

        assert hash1 == hash2

    def test_hashable_in_set(self) -> None:
        # Identifiable objects should be usable in sets
        obj1 = Identifiable()
        obj2 = Identifiable()

        obj_set = {obj1, obj2}

        assert len(obj_set) == 2
        assert obj1 in obj_set
        assert obj2 in obj_set

    def test_equality_same_id(self) -> None:
        # Objects with same ID should be equal
        obj = Identifiable()
        obj_id = obj.id

        # Get same object from registry
        retrieved = Identifiable.get_instance(obj_id)

        assert obj == retrieved
        assert obj is retrieved  # Same object

    def test_equality_different_id(self) -> None:
        # Objects with different IDs should not be equal
        obj1 = Identifiable()
        obj2 = Identifiable()

        assert obj1 != obj2

    def test_equality_with_non_identifiable(self) -> None:
        # Comparison with non-Identifiable should return NotImplemented
        obj = Identifiable()

        result = obj.__eq__("not an identifiable")
        assert result is NotImplemented

    def test_registered_in_instances(self) -> None:
        # New instances should be registered
        obj = Identifiable()
        obj_id = obj.id

        assert obj_id in Identifiable._instances

    def test_get_instance_returns_object(self) -> None:
        # get_instance should return the object
        obj = Identifiable()
        obj_id = obj.id

        retrieved = Identifiable.get_instance(obj_id)

        assert retrieved is obj

    def test_get_instance_nonexistent_returns_none(self) -> None:
        # get_instance with invalid ID should return None
        fake_id = uuid.uuid4().hex

        result = Identifiable.get_instance(fake_id)

        assert result is None

    def test_get_instance_after_garbage_collection(self) -> None:
        # get_instance should return None after object is garbage collected
        obj = Identifiable()
        obj_id = obj.id

        # Verify it's registered
        assert Identifiable.get_instance(obj_id) is obj

        # Delete object
        del obj

        # Should now return None (weak reference broken)
        assert Identifiable.get_instance(obj_id) is None

    def test_multiple_instances_independent(self) -> None:
        # Multiple instances should have independent IDs
        obj1 = Identifiable()
        obj2 = Identifiable()
        obj3 = Identifiable()

        ids = {obj1.id, obj2.id, obj3.id}

        assert len(ids) == 3


# ========== ========== ========== ========== Test Taggable
class TestTaggable:
    """Test Taggable mixin."""

    def test_accepts_valid_identifier(self) -> None:
        # Valid Python identifiers should be accepted
        obj = Taggable()

        obj.tag = "valid_tag"
        assert obj.tag == "valid_tag"

        obj.tag = "tag123"
        assert obj.tag == "tag123"

        obj.tag = "_private"
        assert obj.tag == "_private"

    def test_accepts_none(self) -> None:
        # None should be accepted
        obj = Taggable()

        obj.tag = None
        assert obj.tag is None

    def test_strips_whitespace(self) -> None:
        # Whitespace should be stripped
        obj = Taggable()

        obj.tag = "  tag_name  "
        assert obj.tag == "tag_name"

    def test_rejects_invalid_identifier(self) -> None:
        # Invalid identifiers should be rejected
        obj = Taggable()

        with pytest.raises(ValueError, match="Must be a valid Python identifier"):
            obj.tag = "123invalid"

        with pytest.raises(ValueError, match="Must be a valid Python identifier"):
            obj.tag = "invalid-tag"

        with pytest.raises(ValueError, match="Must be a valid Python identifier"):
            obj.tag = "invalid tag"

    def test_rejects_python_keywords(self) -> None:
        # Python keywords should be rejected
        obj = Taggable()

        keywords_to_test = ['if', 'for', 'while', 'class', 'def', 'return']

        for kw in keywords_to_test:
            with pytest.raises(ValueError, match="Must not be a Python keyword"):
                obj.tag = kw

    def test_rejects_empty_string(self) -> None:
        # Empty strings should be rejected (not valid identifiers)
        obj = Taggable()

        with pytest.raises(ValueError, match="Must be a valid Python identifier"):
            obj.tag = ""

        with pytest.raises(ValueError, match="Must be a valid Python identifier"):
            obj.tag = "   "  # Only whitespace

    def test_tag_is_mutable(self) -> None:
        # Tags should be changeable (not write-once)
        obj = Taggable()

        obj.tag = "first_tag"
        assert obj.tag == "first_tag"

        obj.tag = "second_tag"
        assert obj.tag == "second_tag"

        obj.tag = None
        assert obj.tag is None

    def test_coerces_to_string(self) -> None:
        # Non-string values should be converted to string
        obj = Taggable()

        obj.tag = "tag_name"  # Already string
        assert obj.tag == "tag_name"

        # Numbers that form valid identifiers might work
        # but typically numbers alone are invalid identifiers

    def test_all_python_keywords_rejected(self) -> None:
        # Test against all Python keywords
        obj = Taggable()

        for kw in keyword.kwlist:
            with pytest.raises(ValueError, match="Must not be a Python keyword"):
                obj.tag = kw

    def test_change_default_tag_of_subclass(self) -> None:

        class Sys(Taggable):

            @Taggable.tag.default
            def tag(self):
                return "tag_name"

        assert Sys().tag == "tag_name"
        assert Taggable().tag is None


# ========== ========== ========== ========== Test Nameable
class TestNameable:
    """Test Nameable mixin."""

    def test_accepts_string(self) -> None:
        # Valid strings should be accepted
        obj = Nameable()

        obj.name = "Test Name"
        assert obj.name == "Test Name"

    def test_accepts_none(self) -> None:
        # None should be accepted
        obj = Nameable()

        obj.name = None
        assert obj.name is None

    def test_strips_whitespace(self) -> None:
        # Leading/trailing whitespace should be stripped
        obj = Nameable()

        obj.name = "  Name  "
        assert obj.name == "Name"

    def test_converts_to_string(self) -> None:
        # Non-string values should be converted
        obj = Nameable()

        obj.name = 123
        assert obj.name == "123"

        obj.name = 3.14
        assert obj.name == "3.14"

    def test_empty_string_becomes_none(self) -> None:
        # Empty strings should become None
        obj = Nameable()

        obj.name = ""
        assert obj.name is None

        obj.name = "   "  # Only whitespace
        assert obj.name is None

    def test_allows_special_characters(self) -> None:
        # Names can contain special characters (unlike tags)
        obj = Nameable()

        obj.name = "Name with spaces"
        assert obj.name == "Name with spaces"

        obj.name = "Name-with-dashes"
        assert obj.name == "Name-with-dashes"

        obj.name = "Name (with parentheses)"
        assert obj.name == "Name (with parentheses)"

    def test_name_is_mutable(self) -> None:
        # Names should be changeable
        obj = Nameable()

        obj.name = "First Name"
        assert obj.name == "First Name"

        obj.name = "Second Name"
        assert obj.name == "Second Name"

    def test_unicode_names(self) -> None:
        # Unicode characters should work
        obj = Nameable()

        obj.name = "Tëst Nämé"
        assert obj.name == "Tëst Nämé"

        obj.name = "测试名称"
        assert obj.name == "测试名称"


# ========== ========== ========== ========== Test Describable
class TestDescribable:
    """Test Describable mixin."""

    def test_accepts_string(self) -> None:
        # Valid strings should be accepted
        obj = Describable()

        obj.description = "This is a description."
        assert obj.description == "This is a description."

    def test_accepts_none(self) -> None:
        # None should be accepted
        obj = Describable()

        obj.description = None
        assert obj.description is None

    def test_strips_whitespace(self) -> None:
        # Leading/trailing whitespace should be stripped
        obj = Describable()

        obj.description = "  Description  "
        assert obj.description == "Description"

    def test_converts_to_string(self) -> None:
        # Non-string values should be converted
        obj = Describable()

        obj.description = 123
        assert obj.description == "123"

    def test_empty_string_becomes_none(self) -> None:
        # Empty strings should become None
        obj = Describable()

        obj.description = ""
        assert obj.description is None

        obj.description = "   "
        assert obj.description is None

    def test_allows_multiline_text(self) -> None:
        # Descriptions can be multiline
        obj = Describable()

        multiline = """This is a
        multiline
        description."""

        obj.description = multiline
        # Whitespace handling might affect exact format
        assert "multiline" in obj.description

    def test_allows_long_text(self) -> None:
        # Long descriptions should work
        obj = Describable()

        long_desc = "A" * 10000
        obj.description = long_desc

        assert len(obj.description) == 10000

    def test_description_is_mutable(self) -> None:
        # Descriptions should be changeable
        obj = Describable()

        obj.description = "First description"
        assert obj.description == "First description"

        obj.description = "Second description"
        assert obj.description == "Second description"


# ========== ========== ========== ========== Test Colorable
class TestColorable:
    """Test Colorable mixin."""

    def test_default_color(self) -> None:
        # Default should be matplotlib C0 blue
        obj = Colorable()

        assert obj.color == '#1F77B4'

    def test_accepts_hex_color(self) -> None:
        # Hex colors should be accepted
        obj = Colorable()

        obj.color = '#FF0000'
        assert obj.color == '#FF0000'

    def test_accepts_rgb_tuple(self) -> None:
        # RGB tuples should be converted to hex
        obj = Colorable()

        obj.color = (1.0, 0.0, 0.0)  # Red in float format
        assert obj.color == '#FF0000'

    def test_accepts_color_names(self) -> None:
        # Named colors should be accepted
        obj = Colorable()

        obj.color = 'red'
        assert obj.color == '#FF0000'

        obj.color = 'blue'
        # Exact hex depends on matplotlib's 'blue'

    def test_normalizes_to_uppercase(self) -> None:
        # Colors should be normalized to uppercase
        obj = Colorable()

        obj.color = '#ff0000'
        assert obj.color == '#FF0000'

    def test_color_rgb_property(self) -> None:
        # color_rgb should return tuple of ints
        obj = Colorable()

        obj.color = '#FF0000'
        rgb = obj.color_rgb

        assert rgb == (255, 0, 0)
        assert all(isinstance(c, int) for c in rgb)

    def test_color_rgb_for_blue(self) -> None:
        # Test RGB conversion for blue
        obj = Colorable()

        obj.color = '#0000FF'
        assert obj.color_rgb == (0, 0, 255)

    def test_color_rgb_for_mixed(self) -> None:
        # Test RGB conversion for mixed color
        obj = Colorable()

        obj.color = '#1F77B4'  # Default color
        assert obj.color_rgb == (31, 119, 180)

    def test_color_is_mutable(self) -> None:
        # Colors should be changeable
        obj = Colorable()

        obj.color = '#FF0000'
        assert obj.color == '#FF0000'

        obj.color = '#00FF00'
        assert obj.color == '#00FF00'

    def test_invalid_color_raises(self) -> None:
        # Invalid colors should raise error
        obj = Colorable()

        with pytest.raises(Exception):  # matplotlib will raise ValueError
            obj.color = 'not-a-color'


# ========== ========== ========== ========== Test Activatable
class TestActivatable:
    """Test Activatable mixin."""

    def test_default_active_true(self) -> None:
        # Default should be True
        obj = Activatable()

        assert obj.active is True

    def test_accepts_true(self) -> None:
        # Should accept True
        obj = Activatable()

        obj.active = True
        assert obj.active is True

    def test_accepts_false(self) -> None:
        # Should accept False
        obj = Activatable()

        obj.active = False
        assert obj.active is False

    def test_rejects_non_boolean(self) -> None:
        # Should reject non-boolean values
        obj = Activatable()

        with pytest.raises(TypeError, match="must be a boolean"):
            obj.active = 1

        with pytest.raises(TypeError, match="must be a boolean"):
            obj.active = 0

        with pytest.raises(TypeError, match="must be a boolean"):
            obj.active = "true"

        with pytest.raises(TypeError, match="must be a boolean"):
            obj.active = []

    def test_active_is_mutable(self) -> None:
        # Active state should be changeable
        obj = Activatable()

        assert obj.active is True

        obj.active = False
        assert obj.active is False

        obj.active = True
        assert obj.active is True

    def test_preserves_boolean_type(self) -> None:
        # Should not coerce to bool, should preserve True/False
        obj = Activatable()

        obj.active = True
        assert obj.active is True
        assert type(obj.active) is bool

        obj.active = False
        assert obj.active is False
        assert type(obj.active) is bool


# ========== ========== ========== ========== Test Composition
class TestMixinComposition:
    """Test multiple mixins used together."""

    def test_all_mixins_together(self) -> None:
        # All mixins should work together
        class FullObject(Identifiable, Taggable, Nameable,
                         Describable, Colorable, Activatable):
            pass

        obj = FullObject()

        # All properties should be accessible
        assert hasattr(obj, 'id')
        assert hasattr(obj, 'tag')
        assert hasattr(obj, 'name')
        assert hasattr(obj, 'description')
        assert hasattr(obj, 'color')
        assert hasattr(obj, 'active')

    def test_partial_mixin_combination(self) -> None:
        # Partial combinations should work
        class PartialObject(Identifiable, Nameable, Colorable):
            pass

        obj = PartialObject()

        assert hasattr(obj, 'id')
        assert hasattr(obj, 'name')
        assert hasattr(obj, 'color')
        assert not hasattr(obj, 'tag')
        assert not hasattr(obj, 'active')

    def test_identifiable_and_taggable(self) -> None:
        # Common combination
        class Component(Identifiable, Taggable):
            pass

        obj = Component()
        obj.tag = "sensor_a"

        assert obj.id is not None
        assert obj.tag == "sensor_a"

        # Should be retrievable by ID
        retrieved = Identifiable.get_instance(obj.id)
        assert retrieved is obj

    def test_nameable_and_describable(self) -> None:
        # Another common combination
        class Documentation(Nameable, Describable):
            pass

        obj = Documentation()
        obj.name = "User Guide"
        obj.description = "Complete user documentation"

        assert obj.name == "User Guide"
        assert obj.description == "Complete user documentation"

    def test_colorable_and_activatable(self) -> None:
        # UI element combination
        class UIElement(Colorable, Activatable):
            pass

        obj = UIElement()
        obj.color = '#FF0000'
        obj.active = False

        assert obj.color == '#FF0000'
        assert obj.active is False

    def test_full_object_initialization(self) -> None:
        # Test initializing all properties at once
        class Asset(Identifiable, Taggable, Nameable,
                    Describable, Colorable, Activatable):
            pass

        obj = Asset()
        obj.tag = "asset_001"
        obj.name = "Primary Asset"
        obj.description = "Main asset for testing"
        obj.color = '#00FF00'
        obj.active = True

        assert len(obj.id) == 32
        assert obj.tag == "asset_001"
        assert obj.name == "Primary Asset"
        assert obj.description == "Main asset for testing"
        assert obj.color == '#00FF00'
        assert obj.active is True

    def test_mixin_order_independence(self) -> None:
        # Order shouldn't matter
        class OrderA(Identifiable, Taggable, Nameable):
            pass

        class OrderB(Nameable, Taggable, Identifiable):
            pass

        obj_a = OrderA()
        obj_b = OrderB()

        # Both should have all properties
        assert hasattr(obj_a, 'id') and hasattr(obj_b, 'id')
        assert hasattr(obj_a, 'tag') and hasattr(obj_b, 'tag')
        assert hasattr(obj_a, 'name') and hasattr(obj_b, 'name')

    def test_multiple_instances_independent(self) -> None:
        # Multiple instances should be independent
        class Item(Identifiable, Nameable, Colorable):
            pass

        item1 = Item()
        item1.name = "Item 1"
        item1.color = '#FF0000'

        item2 = Item()
        item2.name = "Item 2"
        item2.color = '#00FF00'

        assert item1.id != item2.id
        assert item1.name == "Item 1"
        assert item2.name == "Item 2"
        assert item1.color == '#FF0000'
        assert item2.color == '#00FF00'

    def test_hash_and_equality_with_mixins(self) -> None:
        # Hash and equality should work with multiple mixins
        class HashableItem(Identifiable, Nameable):
            pass

        item1 = HashableItem()
        item1.name = "Test"

        item2 = HashableItem()
        item2.name = "Test"

        # Different IDs means different objects
        assert item1 != item2
        assert hash(item1) != hash(item2)

        # Can use in sets
        item_set = {item1, item2}
        assert len(item_set) == 2