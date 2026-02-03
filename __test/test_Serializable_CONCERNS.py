#  -*- coding: utf-8 -*-
"""
Test suite for edge cases, known issues, and potential concerns in Serializable.

These tests document behaviors that may need attention in future versions.
Tests marked with @pytest.mark.xfail are expected to fail with current implementation.
Tests marked with @pytest.mark.skip are documented but not yet implemented.

Categories:
- Circular references
- Thread safety
- Observer performance during deserialization
- Malformed data handling
- Memory concerns
- Complex edge cases
"""

from __future__ import annotations

import pytest
import numpy as np
import threading
import time
from typing import Any
from unittest.mock import Mock

from jangada.serialization import (SerializableProperty,
                                   SerializableMetatype,
                                   Serializable,
                                   get_full_qualified_name,
                                   check_types)


# ========== ========== ========== ========== Circular References
class TestCircularReferences:
    """Test handling of circular references in object graphs."""

    @pytest.mark.xfail(reason="Circular references cause infinite recursion")
    def test_self_reference_serialization(self) -> None:
        # Object that references itself should not cause infinite recursion
        class Node(Serializable):
            value = SerializableProperty(default=0)
            next = SerializableProperty(default=None)

        node = Node(value=1)
        node.next = node  # Self-reference

        # This will currently hang/crash
        try:
            result = Serializable.serialize(node)
            assert '__class__' in result
        except RecursionError:
            pytest.fail("RecursionError raised for self-reference")

    @pytest.mark.xfail(reason="Circular references cause infinite recursion")
    def test_mutual_reference_serialization(self) -> None:
        # Two objects referencing each other should not cause infinite recursion
        class Node(Serializable):
            value = SerializableProperty(default=0)
            partner = SerializableProperty(default=None)

        node1 = Node(value=1)
        node2 = Node(value=2)
        node1.partner = node2
        node2.partner = node1  # Circular reference

        # This will currently hang/crash
        try:
            result = Serializable.serialize(node1)
            assert result is not None
        except RecursionError:
            pytest.fail("RecursionError raised for circular reference")

    @pytest.mark.xfail(reason="Circular references cause infinite recursion")
    def test_circular_list_serialization(self) -> None:
        # List containing objects that reference back to the list
        class Container(Serializable):
            items = SerializableProperty(default=None)

        class Item(Serializable):
            parent = SerializableProperty(default=None)

        container = Container()
        item = Item()
        container.items = [item]
        item.parent = container  # Circular through list

        try:
            result = Serializable.serialize(container)
            assert result is not None
        except RecursionError:
            pytest.fail("RecursionError raised for circular reference through list")

    @pytest.mark.skip(reason="Feature not implemented: cycle detection")
    def test_cycle_detection_in_serialize(self) -> None:
        # Serialization should detect cycles and handle gracefully
        class Node(Serializable):
            next = SerializableProperty(default=None)

        node = Node()
        node.next = node

        # Expected behavior: either raise meaningful error or use object IDs
        with pytest.raises(ValueError, match="Circular reference detected"):
            Serializable.serialize(node)

    @pytest.mark.skip(reason="Feature not implemented: reference tracking")
    def test_shared_reference_preservation(self) -> None:
        # Shared references should be preserved, not duplicated
        class Node(Serializable):
            child = SerializableProperty(default=None)

        shared = Node()
        parent1 = Node(child=shared)
        parent2 = Node(child=shared)

        container = [parent1, parent2]
        serialized = Serializable.serialize(container)
        deserialized = Serializable.deserialize(serialized)

        # Ideally, deserialized[0].child and deserialized[1].child
        # should be the same object
        assert deserialized[0].child is deserialized[1].child


# ========== ========== ========== ========== Thread Safety
class TestThreadSafety:
    """Test thread safety of class-level registries."""

    @pytest.mark.skip(reason="Thread safety not guaranteed")
    def test_concurrent_subclass_registration(self) -> None:
        # Multiple threads registering subclasses simultaneously
        results = []

        def create_subclass(name: str) -> None:
            class DynamicClass(Serializable):
                pass
            DynamicClass.__name__ = name
            DynamicClass.__qualname__ = name
            results.append(name)

        threads = [
            threading.Thread(target=create_subclass, args=(f"Class{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All classes should be registered
        assert len(results) == 10

    @pytest.mark.skip(reason="Thread safety not guaranteed")
    def test_concurrent_type_registration(self) -> None:
        # Multiple threads registering primitive types
        def register_type() -> None:
            class CustomType:
                pass
            Serializable.register_primitive_type(CustomType)
            time.sleep(0.001)  # Simulate work
            Serializable.remove_primitive_type(CustomType)

        threads = [threading.Thread(target=register_type) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash or corrupt registry

    @pytest.mark.skip(reason="Thread safety not guaranteed")
    def test_concurrent_serialization(self) -> None:
        # Multiple threads serializing objects simultaneously
        class TestClass(Serializable):
            value = SerializableProperty(default=0)

        results = []

        def serialize_object(value: int) -> None:
            obj = TestClass(value=value)
            data = Serializable.serialize(obj)
            results.append(data)

        threads = [
            threading.Thread(target=serialize_object, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        # All should have correct values
        values = {r['value'] for r in results}
        assert values == set(range(20))


# ========== ========== ========== ========== Observer Performance
class TestObserverPerformance:
    """Test observer behavior during deserialization."""

    def test_observers_fire_during_deserialization(self) -> None:
        # Document that observers fire during deserialization
        call_count = []

        def observer(instance: object, old: Any, new: Any) -> None:
            call_count.append((old, new))

        class TestClass(Serializable):
            value = SerializableProperty(default=0, observers={observer})

        # Serialize
        obj = TestClass(value=42)
        call_count.clear()  # Reset

        data = Serializable.serialize(obj)

        # Deserialize - observers will fire
        restored = Serializable.deserialize(data)

        # Observer was called during deserialization
        assert len(call_count) > 0
        assert call_count[-1] == (None, 42)  # First set

    @pytest.mark.skip(reason="Performance concern not addressed")
    def test_large_object_graph_deserialization_performance(self) -> None:
        # Deserializing large graphs with many observers may be slow
        call_count = []

        def observer(instance: object, old: Any, new: Any) -> None:
            call_count.append(1)

        class Node(Serializable):
            value = SerializableProperty(default=0, observers={observer})
            children = SerializableProperty(default=None)

        # Create deep tree (100 nodes)
        def make_tree(depth: int) -> Node:
            if depth == 0:
                return Node(value=depth)
            return Node(value=depth, children=[make_tree(depth-1), make_tree(depth-1)])

        tree = make_tree(5)  # 2^6-1 = 63 nodes
        data = Serializable.serialize(tree)
        call_count.clear()

        # Deserialization triggers observer for each node
        import time
        start = time.time()
        restored = Serializable.deserialize(data)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # Should be fast
        # But many observer calls happened
        assert len(call_count) > 50

    @pytest.mark.skip(reason="Feature not implemented: silent deserialization")
    def test_deserialize_without_observers(self) -> None:
        # Option to deserialize without triggering observers
        observer = Mock()

        class TestClass(Serializable):
            value = SerializableProperty(default=0, observers={observer})

        obj = TestClass(value=42)
        data = Serializable.serialize(obj)

        # Hypothetical API
        restored = Serializable.deserialize(data, fire_observers=False)

        # Observer should not have been called during deserialization
        observer.assert_not_called()


# ========== ========== ========== ========== Malformed Data
class TestMalformedData:
    """Test handling of malformed or unexpected data."""

    def test_deserialize_missing_class_key(self) -> None:
        # Data without __class__ but looking like object data
        data = {'value': 42, 'name': "test"}

        # Should be treated as plain dict
        result = Serializable.deserialize(data)
        assert isinstance(result, dict)
        assert result == data

    @pytest.mark.xfail(reason="No validation of __class__ format")
    def test_deserialize_malformed_class_name(self) -> None:
        # __class__ with invalid format
        data = {'__class__': 'not-a-valid-name!!!', 'value': 42}

        # Should raise meaningful error, not generic KeyError
        with pytest.raises(ValueError, match="Invalid class name"):
            Serializable.deserialize(data)

    @pytest.mark.xfail(reason="No validation during deserialization")
    def test_deserialize_wrong_property_types(self) -> None:
        # Data with property values of wrong type
        class TestClass(Serializable):
            number = SerializableProperty(
                default=0,
                parser=lambda self, v: int(v)
            )

        data = {
            '__class__': get_full_qualified_name(TestClass),
            'number': 'not-a-number'
        }

        # Parser should catch this
        with pytest.raises(ValueError):
            Serializable.deserialize(data)

    def test_deserialize_extra_properties_in_data(self) -> None:
        # Data contains properties not in class definition
        class TestClass(Serializable):
            value = SerializableProperty(default=0)

        qualname = get_full_qualified_name(TestClass)
        data = {
            '__class__': qualname,
            'value': 42,
            'extra_prop': 999  # Not in TestClass
        }

        # Should raise ValueError about unknown keys
        with pytest.raises(ValueError, match="no signature with the keys"):
            Serializable.deserialize(data)

    @pytest.mark.skip(reason="Feature not implemented: version compatibility")
    def test_deserialize_old_version_data(self) -> None:
        # Data from old version missing new required properties
        class TestClass(Serializable):
            old_prop = SerializableProperty(default=0)
            new_prop = SerializableProperty(default=0)  # Added in v2

        # Old data without new_prop
        data = {
            '__class__': get_full_qualified_name(TestClass),
            'old_prop': 42
        }

        # Should use default for missing new_prop
        result = Serializable.deserialize(data)
        assert result.old_prop == 42
        assert result.new_prop == 0

    def test_deserialize_none_in_collection(self) -> None:
        # Collections containing None values
        data = [1, None, 3, None, 5]
        result = Serializable.deserialize(data)
        assert result == [1, None, 3, None, 5]


# ========== ========== ========== ========== Memory Concerns
class TestMemoryConcerns:
    """Test memory-related edge cases."""

    @pytest.mark.skip(reason="Memory optimization not implemented")
    def test_large_array_serialization_memory(self) -> None:
        # Serializing very large arrays shouldn't duplicate memory
        class ArrayHolder(Serializable):
            data = SerializableProperty(default=None)

        # 100MB array
        large_array = np.zeros((10_000, 1_000), dtype=np.float64)
        obj = ArrayHolder(data=large_array)

        import sys
        before = sys.getsizeof(obj)
        serialized = Serializable.serialize(obj)
        after = sys.getsizeof(serialized)

        # Should not significantly increase memory
        # (This test is conceptual - actual behavior depends on implementation)
        assert after < before * 2

    @pytest.mark.skip(reason="Weak reference support not implemented")
    def test_weak_references_not_serialized(self) -> None:
        # Weak references should be handled gracefully
        import weakref

        class Node(Serializable):
            child = SerializableProperty(default=None)

        child = Node()
        parent = Node()
        parent.child = weakref.ref(child)

        # Should either skip weak refs or raise clear error
        try:
            result = Serializable.serialize(parent)
            # If it succeeds, weak ref should be None or skipped
            assert result['child'] is None
        except TypeError as e:
            assert 'weakref' in str(e).lower()

    def test_property_initialization_creates_instance_attributes(self) -> None:
        # Document that properties create instance attributes
        class TestClass(Serializable):
            prop1 = SerializableProperty(default=0)
            prop2 = SerializableProperty(default=0)

        obj = TestClass(prop1=1, prop2=2)

        # Instance has mangled names in __dict__
        assert '_serializable_property__prop1' in obj.__dict__
        assert '_serializable_property__prop2' in obj.__dict__


# ========== ========== ========== ========== Complex Edge Cases
class TestComplexEdgeCases:
    """Test complex scenarios and corner cases."""

    def test_serialize_then_modify_class_definition(self) -> None:
        # Serialize, modify class, then deserialize
        class TestClass(Serializable):
            value = SerializableProperty(default=0)

        obj = TestClass(value=42)
        data = Serializable.serialize(obj)

        # Add new property to class
        TestClass.new_prop = SerializableProperty(default=99)
        TestClass._serializable_properties['new_prop'] = TestClass.new_prop
        TestClass.new_prop.__set_name__(TestClass, 'new_prop')

        # Deserialize old data
        restored = Serializable.deserialize(data)

        # Old property preserved, new property uses default
        assert restored.value == 42
        assert restored.new_prop == 99

    @pytest.mark.skip(reason="Class versioning not implemented")
    def test_class_version_mismatch(self) -> None:
        # Data from different version of same class
        class TestClass(Serializable):
            __version__ = "2.0"
            value = SerializableProperty(default=0)

        # Data from version 1.0
        data = {
            '__class__': get_full_qualified_name(TestClass),
            '__version__': "1.0",
            'value': 42
        }

        # Should either migrate or warn
        result = Serializable.deserialize(data)
        assert result.value == 42

    def test_equality_with_nan_values(self) -> None:
        # NaN values in arrays affect equality
        class TestClass(Serializable):
            data = SerializableProperty(default=None)

        obj1 = TestClass(data=np.array([1.0, np.nan, 3.0]))
        obj2 = TestClass(data=np.array([1.0, np.nan, 3.0]))

        # NaN != NaN, so objects with NaN should not be equal
        # Current implementation uses np.all() which may not handle this correctly
        result = obj1 == obj2
        # Behavior depends on implementation

    def test_property_name_conflicts(self) -> None:
        # Property names that could conflict with internal attributes
        class TestClass(Serializable):
            # These might conflict with internal state
            _serializable_properties = SerializableProperty(default=None)
            __dict__ = SerializableProperty(default=None)

        # Should still work (uses mangled names internally)
        obj = TestClass(_serializable_properties=42, __dict__=99)
        assert obj._serializable_properties == 42
        assert obj.__dict__ == 99

    @pytest.mark.xfail(reason="Dynamic class creation may not preserve all metadata")
    def test_generic_class_creation_preserves_behavior(self) -> None:
        # Generic classes created for unknown types should behave normally
        data = {
            '__class__': 'unknown.Module.UnknownClass',
            'value': 42,
            'name': "test"
        }

        obj = Serializable.deserialize(data)

        # Should support all Serializable operations
        copy = obj.copy()
        assert copy == obj

        # Should be re-serializable
        reserialized = Serializable.serialize(obj)
        assert reserialized['value'] == 42

    def test_multiple_inheritance_property_resolution(self) -> None:
        # Multiple inheritance with properties in multiple bases
        class Base1(Serializable):
            prop1 = SerializableProperty(default=1)

        class Base2(Serializable):
            prop2 = SerializableProperty(default=2)

        class Derived(Base1, Base2):
            prop3 = SerializableProperty(default=3)

        # All properties should be collected
        assert 'prop1' in Derived._serializable_properties
        assert 'prop2' in Derived._serializable_properties
        assert 'prop3' in Derived._serializable_properties

        obj = Derived(prop1=10, prop2=20, prop3=30)
        data = Serializable.serialize(obj)
        restored = Serializable.deserialize(data)

        assert restored.prop1 == 10
        assert restored.prop2 == 20
        assert restored.prop3 == 30

    @pytest.mark.skip(reason="Metaclass conflict handling not tested")
    def test_multiple_inheritance_with_other_metaclasses(self) -> None:
        # Combining Serializable with classes using other metaclasses
        from abc import ABCMeta, abstractmethod

        # This might cause metaclass conflicts
        class AbstractSerializable(Serializable, metaclass=ABCMeta):
            @abstractmethod
            def process(self) -> None:
                pass

        class Concrete(AbstractSerializable):
            value = SerializableProperty(default=0)

            def process(self) -> None:
                return self.value * 2

        obj = Concrete(value=5)
        assert obj.process() == 10


# ========== ========== ========== ========== Dataset Type Edge Cases
class TestDatasetTypeEdgeCases:
    """Test edge cases with dataset type registration."""

    def test_dataset_disassemble_returns_wrong_format(self) -> None:
        # Disassemble function returns wrong format
        class BadDataset:
            pass

        def bad_disassemble(obj: Any) -> Any:
            return "not-a-tuple"  # Should return (array, attrs)

        def assemble(arr: Any, attrs: Any) -> BadDataset:
            return BadDataset()

        Serializable.register_dataset_type(BadDataset, bad_disassemble, assemble)

        obj = BadDataset()

        # Should fail when trying to serialize
        with pytest.raises(Exception):  # Type depends on implementation
            Serializable.serialize(obj)

        # Cleanup
        Serializable.remove_dataset_type(BadDataset)

    @pytest.mark.skip(reason="Dataset type validation not implemented")
    def test_register_dataset_type_validates_signatures(self) -> None:
        # Registration should validate disassemble/assemble signatures
        class CustomDataset:
            pass

        def bad_disassemble(wrong_param_name: Any) -> tuple:
            return (np.array([]), {})

        def bad_assemble() -> CustomDataset:  # Missing parameters
            return CustomDataset()

        # Should raise TypeError for invalid signatures
        with pytest.raises(TypeError, match="Invalid disassemble signature"):
            Serializable.register_dataset_type(CustomDataset, bad_disassemble, bad_assemble)

    def test_dataset_type_unregistered_after_removal(self) -> None:
        # Ensure dataset type is fully removed
        class TempDataset:
            pass

        Serializable.register_dataset_type(
            TempDataset,
            lambda obj: (np.array([]), {}),
            lambda arr, attrs: TempDataset()
        )

        assert Serializable.is_dataset_type(TempDataset)
        assert Serializable.is_primitive_type(TempDataset)

        Serializable.remove_dataset_type(TempDataset)

        assert not Serializable.is_dataset_type(TempDataset)
        assert not Serializable.is_primitive_type(TempDataset)

        # Trying to serialize should fail
        with pytest.raises(TypeError):
            Serializable.serialize(TempDataset())


# ========== ========== ========== ========== Summary Report
@pytest.fixture(scope="session", autouse=True)
def print_concern_summary(request) -> None:
    """Print summary of concerns at end of test session."""
    yield

    # This runs after all tests
    print("\n" + "="*70)
    print("EDGE CASE & CONCERN TEST SUMMARY")
    print("="*70)
    print("\nKnown Issues (xfail):")
    print("  - Circular references cause infinite recursion")
    print("  - Some malformed data cases not handled gracefully")
    print("\nFuture Features (skip):")
    print("  - Thread safety for concurrent operations")
    print("  - Cycle detection in serialization")
    print("  - Silent deserialization without observers")
    print("  - Class versioning and migration")
    print("  - Weak reference support")
    print("\nDocumented Behaviors:")
    print("  - Observers fire during deserialization")
    print("  - Properties create mangled instance attributes")
    print("  - Generic classes created for unknown types")
    print("\nSee test_serializable_concerns.py for details")
    print("="*70 + "\n")
