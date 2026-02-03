#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for Serializable and SerializableMetatype.

Tests cover:
- Metaclass registration and discovery
- Serialization/deserialization of simple objects
- Serialization/deserialization of nested objects
- Collections (list, dict, tuple, set)
- Primitive types
- Dataset types (numpy arrays, pandas timestamps)
- Copy functionality
- Equality comparison
- Type registry management
- Edge cases

Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from unittest.mock import Mock


from jangada.serialization import (SerializableProperty,
                                   SerializableMetatype,
                                   Serializable,
                                   get_full_qualified_name,
                                   check_types)


# ========== ========== ========== ========== Fixtures
@pytest.fixture
def simple_serializable_class() -> type:
    """A simple Serializable class for testing."""

    class SimpleClass(Serializable):
        value = SerializableProperty(default=0)
        name = SerializableProperty(default="")

    return SimpleClass


@pytest.fixture
def nested_serializable_class(simple_serializable_class: type) -> type:
    """A Serializable class with nested Serializable properties."""

    class NestedClass(Serializable):
        child = SerializableProperty(default=None)
        data = SerializableProperty(default=0)

    return NestedClass


# ========== ========== ========== ========== Helper Functions
class TestHelperFunctions:
    """Test utility functions."""

    def test_get_full_qualified_name_regular_class(self) -> None:
        # Should return module.qualname for regular classes
        class MyClass:
            pass

        result = get_full_qualified_name(MyClass)
        assert result.endswith('.MyClass')
        assert 'test_Serializable' in result

    def test_get_full_qualified_name_builtin(self) -> None:
        # Builtin types should return just the qualname
        result = get_full_qualified_name(int)
        assert result == 'int'

    def test_check_types_valid(self) -> None:
        # Should return True for valid types
        assert check_types(5, int) is True
        assert check_types("hello", str) is True
        assert check_types([1, 2], list) is True

    def test_check_types_multiple_types(self) -> None:
        # Should accept tuple of types
        assert check_types(5, (int, str)) is True
        assert check_types("hello", (int, str)) is True

    def test_check_types_invalid_raises(self) -> None:
        # Should raise TypeError for invalid types
        with pytest.raises(TypeError, match="Expected instance"):
            check_types(5, str)

    def test_check_types_invalid_no_raise(self) -> None:
        # Should return False when raise_error=False
        result = check_types(5, str, raise_error=False)
        assert result is False

    def test_check_types_none_allowed(self) -> None:
        # Should accept None when can_be_none=True
        assert check_types(None, int, can_be_none=True) is True
        assert check_types(5, int, can_be_none=True) is True

    def test_check_types_none_not_allowed(self) -> None:
        # Should reject None when can_be_none=False
        with pytest.raises(TypeError):
            check_types(None, int, can_be_none=False)


# ========== ========== ========== ========== Metaclass
class TestSerializableMetatype:
    """Test metaclass registration and discovery."""

    def test_Serializable_base_has_registries(self) -> None:
        # Serializable base class should have registry attributes
        assert hasattr(Serializable, '_subclasses')
        assert hasattr(Serializable, '_primitive_types')
        assert hasattr(Serializable, '_dataset_types')

    def test_subclass_registration(self) -> None:
        # Subclasses should be automatically registered
        class TestClass(Serializable):
            pass

        qualname = get_full_qualified_name(TestClass)
        assert qualname in Serializable._subclasses
        assert Serializable._subclasses[qualname] is TestClass

    def test_subscript_access_to_subclass(self) -> None:
        # Should be able to access subclass by qualified name
        class TestClass(Serializable):
            pass

        qualname = get_full_qualified_name(TestClass)
        retrieved = Serializable[qualname]
        assert retrieved is TestClass

    def test_subscript_on_non_base_raises(self) -> None:
        # Subscripting non-Serializable classes should raise KeyError
        class TestClass(Serializable):
            pass

        with pytest.raises(KeyError, match="not subscriptable"):
            TestClass["something"]

    def test_contains_with_string(self) -> None:
        # Should check if qualified name is registered
        class TestClass(Serializable):
            pass

        qualname = get_full_qualified_name(TestClass)
        assert qualname in Serializable
        assert "nonexistent.Class" not in Serializable

    def test_contains_with_type(self) -> None:
        # Should check if type is registered
        class TestClass(Serializable):
            pass

        assert TestClass in Serializable

    def test_contains_invalid_type_raises(self) -> None:
        # Should raise TypeError for invalid types
        with pytest.raises(TypeError, match="Expected the class full qualified name"):
            123 in Serializable

    def test_Serializable_properties_collected(self) -> None:
        # Metaclass should collect SerializableProperty descriptors
        class TestClass(Serializable):
            prop1 = SerializableProperty()
            prop2 = SerializableProperty()
            regular_attr = 42

        assert 'prop1' in TestClass._serializable_properties
        assert 'prop2' in TestClass._serializable_properties
        assert 'regular_attr' not in TestClass._serializable_properties

    def test_inherited_properties_collected(self) -> None:
        # Should collect properties from base classes
        class Base(Serializable):
            base_prop = SerializableProperty()

        class Derived(Base):
            derived_prop = SerializableProperty()

        assert 'base_prop' in Derived._serializable_properties
        assert 'derived_prop' in Derived._serializable_properties

    def test_Serializable_properties_property(self) -> None:
        # Should provide read-only access to properties
        class TestClass(Serializable):
            prop = SerializableProperty()

        props = TestClass.serializable_properties
        assert 'prop' in props
        assert isinstance(props, dict)

    def test_copiable_properties_property(self) -> None:
        # Should filter properties by copiable flag
        class TestClass(Serializable):
            copiable_prop = SerializableProperty(copiable=True)
            non_copiable_prop = SerializableProperty(copiable=False)

        copiable = TestClass.copiable_properties
        assert 'copiable_prop' in copiable
        assert 'non_copiable_prop' not in copiable


# ========== ========== ========== ========== Type Registration
class TestTypeRegistration:
    """Test primitive and dataset type registration."""

    def test_register_primitive_type(self) -> None:
        # Should add type to primitive types registry
        class CustomType:
            pass

        Serializable.register_primitive_type(CustomType)
        assert Serializable.is_primitive_type(CustomType)

        # Cleanup
        Serializable.remove_primitive_type(CustomType)

    def test_register_primitive_rejects_collections(self) -> None:
        # Should reject list, dict, tuple as primitives
        with pytest.raises(TypeError, match="Cannot register"):
            Serializable.register_primitive_type(list)

        with pytest.raises(TypeError, match="Cannot register"):
            Serializable.register_primitive_type(dict)

    def test_remove_primitive_type(self) -> None:
        # Should remove type from registry
        class CustomType:
            pass

        Serializable.register_primitive_type(CustomType)
        Serializable.remove_primitive_type(CustomType)
        assert not Serializable.is_primitive_type(CustomType)

    def test_primitive_types_property(self) -> None:
        # Should return list of primitive types
        types = Serializable.primitive_types
        assert isinstance(types, list)
        assert str in types
        # Number is registered by default

    def test_register_dataset_type(self) -> None:
        # Should register type with disassemble/assemble functions
        class CustomDataset:
            pass

        def disassemble(obj):
            return np.array([1, 2, 3]), {}

        def assemble(arr, attrs):
            return CustomDataset()

        Serializable.register_dataset_type(CustomDataset, disassemble, assemble)

        assert Serializable.is_dataset_type(CustomDataset)
        assert Serializable.is_primitive_type(CustomDataset)

        # Cleanup
        Serializable.remove_dataset_type(CustomDataset)

    def test_dataset_type_registered_by_name_and_type(self) -> None:
        # Should be accessible by both type and qualified name
        class CustomDataset:
            pass

        def disassemble(obj):
            return np.array([]), {}

        def assemble(arr, attrs):
            return CustomDataset()

        Serializable.register_dataset_type(CustomDataset, disassemble, assemble)

        qualname = get_full_qualified_name(CustomDataset)
        assert CustomDataset in Serializable._dataset_types
        assert qualname in Serializable._dataset_types

        # Cleanup
        Serializable.remove_dataset_type(CustomDataset)

    def test_remove_dataset_type(self) -> None:
        # Should remove dataset type from all registries
        class CustomDataset:
            pass

        Serializable.register_dataset_type(
            CustomDataset,
            lambda obj: (np.array([]), {}),
            lambda arr, attrs: CustomDataset()
        )

        Serializable.remove_dataset_type(CustomDataset)

        assert not Serializable.is_dataset_type(CustomDataset)
        assert not Serializable.is_primitive_type(CustomDataset)

    def test_dataset_types_property(self) -> None:
        # Should return list of dataset types (not including string keys)
        types = Serializable.dataset_types
        assert isinstance(types, list)
        assert np.ndarray in types
        # Should not include string qualified names
        assert all(isinstance(t, type) for t in types)


# ========== ========== ========== ========== Initialization
class TestSerializableInitialization:
    """Test Serializable object construction."""

    def test_init_with_kwargs(self, simple_serializable_class: type) -> None:
        # Should initialize from keyword arguments
        obj = simple_serializable_class(value=42, name="test")
        assert obj.value == 42
        assert obj.name == "test"

    def test_init_with_partial_kwargs(self, simple_serializable_class: type) -> None:
        # Should use defaults for missing kwargs
        obj = simple_serializable_class(value=100)
        assert obj.value == 100
        assert obj.name == ""  # Default

    def test_init_copy_constructor(self, simple_serializable_class: type) -> None:
        # Should create a copy from another instance
        original = simple_serializable_class(value=42, name="original")
        copy = simple_serializable_class(original)

        assert copy.value == original.value
        assert copy.name == original.name
        assert copy is not original

    def test_init_invalid_args_raises(self, simple_serializable_class: type) -> None:
        # Should raise ValueError for invalid arguments
        with pytest.raises(ValueError, match="do not match any available signature"):
            simple_serializable_class("invalid", "args")

    def test_init_with_class_in_data(self, simple_serializable_class: type) -> None:
        # Should handle __class__ key in data
        qualname = get_full_qualified_name(simple_serializable_class)
        data = {'__class__': qualname, 'value': 99, 'name': 'test'}
        obj = simple_serializable_class(**data)

        assert obj.value == 99
        assert obj.name == 'test'

    def test_init_extra_keys_raises(self, simple_serializable_class: type) -> None:
        # Should raise ValueError for unknown keys
        with pytest.raises(ValueError, match="no signature with the keys"):
            simple_serializable_class(value=1, unknown_key=2)


# ========== ========== ========== ========== Serialization
class TestSerialization:
    """Test serialization of objects to dictionaries."""

    def test_serialize_none(self) -> None:
        # None should serialize to None
        result = Serializable.serialize(None)
        assert result is None

    def test_serialize_primitive(self) -> None:
        # Primitives should serialize as-is
        assert Serializable.serialize(42) == 42
        assert Serializable.serialize("hello") == "hello"
        assert Serializable.serialize(3.14) == 3.14

    def test_serialize_path(self) -> None:
        # Path should serialize as-is (registered primitive)
        path = Path("/tmp/test.txt")
        result = Serializable.serialize(path)
        assert result == path

    def test_serialize_list(self) -> None:
        # Lists should serialize recursively
        data = [1, 2, "three", 4.0]
        result = Serializable.serialize(data)
        assert result == [1, 2, "three", 4.0]

    def test_serialize_dict(self) -> None:
        # Dicts should serialize recursively
        data = {"a": 1, "b": "two", "c": 3.0}
        result = Serializable.serialize(data)
        assert result == {"a": 1, "b": "two", "c": 3.0}

    def test_serialize_tuple(self) -> None:
        # Tuples should serialize to lists
        data = (1, 2, 3)
        result = Serializable.serialize(data)
        assert result == [1, 2, 3]

    def test_serialize_set(self) -> None:
        # Sets should serialize to lists
        data = {1, 2, 3}
        result = Serializable.serialize(data)
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_serialize_simple_object(self, simple_serializable_class: type) -> None:
        # Should serialize Serializable objects to dicts with __class__
        obj = simple_serializable_class(value=42, name="test")
        result = Serializable.serialize(obj)

        assert isinstance(result, dict)
        assert '__class__' in result
        assert result['value'] == 42
        assert result['name'] == "test"

    def test_serialize_nested_object(self, simple_serializable_class: type,
                                     nested_serializable_class: type) -> None:
        # Should recursively serialize nested objects
        child = simple_serializable_class(value=10, name="child")
        parent = nested_serializable_class(child=child, data=100)

        result = Serializable.serialize(parent)

        assert isinstance(result, dict)
        assert '__class__' in result
        assert isinstance(result['child'], dict)
        assert result['child']['value'] == 10
        assert result['data'] == 100

    def test_serialize_with_collections(self, simple_serializable_class: type) -> None:
        # Should handle objects in collections
        obj1 = simple_serializable_class(value=1, name="one")
        obj2 = simple_serializable_class(value=2, name="two")

        data = [obj1, obj2]
        result = Serializable.serialize(data)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all('__class__' in item for item in result)

    def test_serialize_copy_mode(self, simple_serializable_class: type) -> None:
        # Should only serialize copiable properties in copy mode
        class TestClass(Serializable):
            copiable = SerializableProperty(default=1, copiable=True)
            non_copiable = SerializableProperty(default=2, copiable=False)

        obj = TestClass(copiable=10, non_copiable=20)
        result = Serializable.serialize(obj, is_copy=True)

        assert 'copiable' in result
        assert 'non_copiable' not in result

    def test_serialize_full_mode(self, simple_serializable_class: type) -> None:
        # Should serialize all properties in full mode
        class TestClass(Serializable):
            copiable = SerializableProperty(default=1, copiable=True)
            non_copiable = SerializableProperty(default=2, copiable=False)

        obj = TestClass(copiable=10, non_copiable=20)
        result = Serializable.serialize(obj, is_copy=False)

        assert 'copiable' in result
        assert 'non_copiable' in result

    def test_serialize_numpy_array(self) -> None:
        # NumPy arrays should serialize (registered as dataset type)
        arr = np.array([1, 2, 3, 4, 5])
        result = Serializable.serialize(arr)
        # Result depends on dataset type registration
        assert result is not None

    def test_serialize_unregistered_type_raises(self) -> None:
        # Should raise TypeError for unregistered types
        class UnknownType:
            pass

        obj = UnknownType()
        with pytest.raises(TypeError, match="No serialisation process"):
            Serializable.serialize(obj)


# ========== ========== ========== ========== Deserialization
class TestDeserialization:
    """Test deserialization from dictionaries to objects."""

    def test_deserialize_none(self) -> None:
        # None should deserialize to None
        result = Serializable.deserialize(None)
        assert result is None

    def test_deserialize_primitive(self) -> None:
        # Primitives should deserialize as-is
        assert Serializable.deserialize(42) == 42
        assert Serializable.deserialize("hello") == "hello"
        assert Serializable.deserialize(3.14) == 3.14

    def test_deserialize_list(self) -> None:
        # Lists should deserialize recursively
        data = [1, 2, "three", 4.0]
        result = Serializable.deserialize(data)
        assert result == [1, 2, "three", 4.0]

    def test_deserialize_dict_without_class(self) -> None:
        # Plain dicts should deserialize recursively
        data = {"a": 1, "b": "two"}
        result = Serializable.deserialize(data)
        assert result == {"a": 1, "b": "two"}

    def test_deserialize_simple_object(self, simple_serializable_class: type) -> None:
        # Should reconstruct objects from serialized data
        qualname = get_full_qualified_name(simple_serializable_class)
        data = {'__class__': qualname, 'value': 42, 'name': "test"}

        result = Serializable.deserialize(data)

        assert isinstance(result, simple_serializable_class)
        assert result.value == 42
        assert result.name == "test"

    def test_deserialize_nested_object(self, simple_serializable_class: type,
                                       nested_serializable_class: type) -> None:
        # Should recursively deserialize nested objects
        child_qualname = get_full_qualified_name(simple_serializable_class)
        parent_qualname = get_full_qualified_name(nested_serializable_class)

        data = {
            '__class__': parent_qualname,
            'child': {
                '__class__': child_qualname,
                'value': 10,
                'name': "child"
            },
            'data': 100
        }

        result = Serializable.deserialize(data)

        assert isinstance(result, nested_serializable_class)
        assert isinstance(result.child, simple_serializable_class)
        assert result.child.value == 10
        assert result.data == 100

    def test_deserialize_unknown_class_creates_generic(self) -> None:
        # Should create generic class for unknown types
        data = {
            '__class__': 'unknown.module.UnknownClass',
            'prop1': 1,
            'prop2': "test"
        }

        result = Serializable.deserialize(data)

        assert isinstance(result, Serializable)
        assert result.prop1 == 1
        assert result.prop2 == "test"

    def test_deserialize_roundtrip(self, simple_serializable_class: type) -> None:
        # Serialize then deserialize should preserve data
        original = simple_serializable_class(value=42, name="test")
        serialized = Serializable.serialize(original)
        deserialized = Serializable.deserialize(serialized)

        assert isinstance(deserialized, simple_serializable_class)
        assert deserialized.value == original.value
        assert deserialized.name == original.name

    def test_deserialize_unregistered_type_raises(self) -> None:
        # Should raise TypeError for unregistered primitives
        class UnknownType:
            pass

        obj = UnknownType()
        with pytest.raises(TypeError, match="No serialisation method"):
            Serializable.deserialize(obj)


# ========== ========== ========== ========== Copy
class TestCopy:
    """Test object copying functionality."""

    def test_copy_method(self, simple_serializable_class: type) -> None:
        # copy() should create independent copy
        original = simple_serializable_class(value=42, name="original")
        copied = original.copy()

        assert isinstance(copied, simple_serializable_class)
        assert copied is not original
        assert copied.value == original.value
        assert copied.name == original.name

    def test_copy_preserves_copiable_only(self) -> None:
        # copy() should only copy copiable properties
        class TestClass(Serializable):
            copiable = SerializableProperty(default=0, copiable=True)
            non_copiable = SerializableProperty(default=0, copiable=False)

        original = TestClass(copiable=10, non_copiable=20)
        copied = original.copy()

        assert copied.copiable == 10
        assert copied.non_copiable == 0  # Default, not copied

    def test_copy_nested_objects(self, simple_serializable_class: type,
                                 nested_serializable_class: type) -> None:
        # Should recursively copy nested objects
        child = simple_serializable_class(value=10, name="child")
        parent = nested_serializable_class(child=child, data=100)

        copied = parent.copy()

        assert copied is not parent
        assert copied.child is not parent.child
        assert copied.child.value == parent.child.value

    def test_dunder_copy(self, simple_serializable_class: type) -> None:
        # __copy__ should work with copy module
        import copy

        original = simple_serializable_class(value=42, name="test")
        copied = copy.copy(original)

        assert copied is not original
        assert copied.value == original.value


# ========== ========== ========== ========== Equality
class TestEquality:
    """Test equality comparison."""

    def test_eq_same_type_same_values(self, simple_serializable_class: type) -> None:
        # Objects with same values should be equal
        obj1 = simple_serializable_class(value=42, name="test")
        obj2 = simple_serializable_class(value=42, name="test")

        assert obj1 == obj2

    def test_eq_same_type_different_values(self, simple_serializable_class: type) -> None:
        # Objects with different values should not be equal
        obj1 = simple_serializable_class(value=42, name="test")
        obj2 = simple_serializable_class(value=99, name="test")

        assert obj1 != obj2

    def test_eq_different_types(self, simple_serializable_class: type) -> None:
        # Objects of different types should not be equal
        class OtherClass(Serializable):
            value = SerializableProperty(default=0)

        obj1 = simple_serializable_class(value=42, name="test")
        obj2 = OtherClass(value=42)

        assert obj1 != obj2

    def test_eq_with_numpy_arrays(self) -> None:
        # Should handle numpy arrays in comparison
        class ArrayClass(Serializable):
            data = SerializableProperty(default=None)

        obj1 = ArrayClass(data=np.array([1, 2, 3]))
        obj2 = ArrayClass(data=np.array([1, 2, 3]))
        obj3 = ArrayClass(data=np.array([4, 5, 6]))

        assert obj1 == obj2
        assert obj1 != obj3

    def test_eq_nested_objects(self, simple_serializable_class: type,
                               nested_serializable_class: type) -> None:
        # Should recursively compare nested objects
        child1 = simple_serializable_class(value=10, name="child")
        parent1 = nested_serializable_class(child=child1, data=100)

        child2 = simple_serializable_class(value=10, name="child")
        parent2 = nested_serializable_class(child=child2, data=100)

        assert parent1 == parent2

    def test_eq_only_compares_copiable(self) -> None:
        # Should only compare copiable properties
        class TestClass(Serializable):
            copiable = SerializableProperty(default=0, copiable=True)
            non_copiable = SerializableProperty(default=0, copiable=False)

        obj1 = TestClass(copiable=10, non_copiable=20)
        obj2 = TestClass(copiable=10, non_copiable=999)

        # Should be equal because non_copiable is ignored
        assert obj1 == obj2


# ========== ========== ========== ========== Pandas Integration
class TestPandasIntegration:
    """Test pandas Timestamp and DatetimeIndex serialization."""

    def test_serialize_timestamp(self) -> None:
        # Should serialize pandas Timestamp
        timestamp = pd.Timestamp('2024-01-15 12:30:00')
        result = Serializable.serialize(timestamp)
        assert result is not None

    def test_serialize_timestamp_with_timezone(self) -> None:
        # Should handle timezone-aware timestamps
        timestamp = pd.Timestamp('2024-01-15 12:30:00', tz='UTC')
        result = Serializable.serialize(timestamp)
        assert result is not None

    def test_deserialize_timestamp_roundtrip(self) -> None:
        # Should roundtrip timestamp serialization
        original = pd.Timestamp('2024-01-15 12:30:00')
        serialized = Serializable.serialize(original)
        deserialized = Serializable.deserialize(serialized)

        # May not be exact same object but should represent same time
        assert isinstance(deserialized, pd.Timestamp)

    def test_serialize_datetimeindex(self) -> None:
        # Should serialize DatetimeIndex
        index = pd.date_range('2024-01-01', periods=5, freq='D')
        result = Serializable.serialize(index)
        assert result is not None


# ========== ========== ========== ========== Integration Tests
class TestIntegration:
    """Test complex scenarios combining multiple features."""

    def test_complete_workflow(self, simple_serializable_class: type) -> None:
        # Create, serialize, deserialize, compare
        original = simple_serializable_class(value=42, name="test")

        # Serialize
        data = Serializable.serialize(original)
        assert isinstance(data, dict)
        assert '__class__' in data

        # Deserialize
        restored = Serializable.deserialize(data)
        assert isinstance(restored, simple_serializable_class)

        # Compare
        assert original == restored

    def test_complex_nested_structure(self) -> None:
        # Test deeply nested objects with collections
        class Inner(Serializable):
            value = SerializableProperty(default=0)

        class Middle(Serializable):
            items = SerializableProperty(default=None)

        class Outer(Serializable):
            children = SerializableProperty(default=None)

        # Create complex structure
        inner1 = Inner(value=1)
        inner2 = Inner(value=2)
        middle = Middle(items=[inner1, inner2])
        outer = Outer(children=[middle])

        # Serialize and deserialize
        data = Serializable.serialize(outer)
        restored = Serializable.deserialize(data)

        # Verify structure
        assert isinstance(restored, Outer)
        assert isinstance(restored.children[0], Middle)
        assert len(restored.children[0].items) == 2
        assert restored.children[0].items[0].value == 1

    def test_mixed_data_types(self) -> None:
        # Test object with various property types
        class MixedClass(Serializable):
            number = SerializableProperty(default=0)
            text = SerializableProperty(default="")
            array = SerializableProperty(default=None)
            path = SerializableProperty(default=None)
            nested = SerializableProperty(default=None)

        class SimpleNested(Serializable):
            val = SerializableProperty(default=0)

        obj = MixedClass(
            number=42,
            text="hello",
            array=np.array([1, 2, 3]),
            path=Path("/tmp/test"),
            nested=SimpleNested(val=99)
        )

        # Roundtrip
        data = Serializable.serialize(obj)
        restored = Serializable.deserialize(data)

        assert restored.number == 42
        assert restored.text == "hello"
        assert isinstance(restored.array, np.ndarray)
        assert restored.path == Path("/tmp/test")
        assert restored.nested.val == 99

    def test_inheritance_hierarchy(self) -> None:
        # Test serialization with class inheritance
        class Base(Serializable):
            base_prop = SerializableProperty(default=0)

        class Derived(Base):
            derived_prop = SerializableProperty(default="")

        obj = Derived(base_prop=10, derived_prop="test")

        # Serialize and deserialize
        data = Serializable.serialize(obj)
        restored = Serializable.deserialize(data)

        assert isinstance(restored, Derived)
        assert restored.base_prop == 10
        assert restored.derived_prop == "test"

    def test_list_of_mixed_objects(self) -> None:
        # Test list containing different Serializable types
        class TypeA(Serializable):
            a = SerializableProperty(default=0)

        class TypeB(Serializable):
            b = SerializableProperty(default="")

        objects = [
            TypeA(a=1),
            TypeB(b="hello"),
            TypeA(a=2),
            42,  # primitive
            "string",  # primitive
        ]

        # Serialize and deserialize
        data = Serializable.serialize(objects)
        restored = Serializable.deserialize(data)

        assert len(restored) == 5
        assert isinstance(restored[0], TypeA)
        assert isinstance(restored[1], TypeB)
        assert isinstance(restored[2], TypeA)
        assert restored[3] == 42
        assert restored[4] == "string"