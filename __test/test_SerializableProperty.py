#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for SerializableProperty descriptor.

Tests cover:
- Basic descriptor protocol (get/set/delete)
- Default values (static and callable)
- Parsers (validation and transformation)
- Observers (change notifications)
- Post-initializers (setup hooks)
- Write-once semantics
- Edge cases and error conditions

Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import pytest
from typing import Any, Callable
from unittest.mock import Mock


from jangada import SerializableProperty

# ========== ========== ========== ========== Fixtures


@pytest.fixture
def simple_class() -> type:
    """A simple class with a basic SerializableProperty for testing."""

    class Simple:
        value = SerializableProperty()

    return Simple


@pytest.fixture
def observer_spy() -> Mock:
    """Mock observer that records calls for verification."""
    return Mock()


# ========== ========== ========== ========== Basic Descriptor Protocol
class TestDescriptorProtocol:
    """Test basic descriptor get/set/delete operations."""

    def test_class_access_returns_descriptor(self) -> None:
        # Accessing from class should return descriptor itself, not value
        class MyClass:
            prop = SerializableProperty()

        assert isinstance(MyClass.prop, SerializableProperty)

    def test_instance_access_returns_value(self) -> None:
        # Accessing from instance should return the stored value
        class MyClass:
            prop = SerializableProperty(default=42)

        obj = MyClass()
        assert obj.prop == 42

    def test_set_and_get_value(self) -> None:
        # Setting a value should store it and get should retrieve it
        class MyClass:
            prop = SerializableProperty()

        obj = MyClass()
        obj.prop = 100
        assert obj.prop == 100

    def test_multiple_instances_independent(self) -> None:
        # Each instance should have independent property values
        class MyClass:
            prop = SerializableProperty(default=0)

        obj1 = MyClass()
        obj2 = MyClass()

        obj1.prop = 10
        obj2.prop = 20

        assert obj1.prop == 10
        assert obj2.prop == 20

    def test_delete_raises_error(self) -> None:
        # Deleting a SerializableProperty should raise AttributeError
        class MyClass:
            prop = SerializableProperty()

        obj = MyClass()
        with pytest.raises(AttributeError, match="can't delete attribute"):
            del obj.prop

    def test_first_access_initializes_with_none_then_default(self) -> None:
        # First access should trigger initialization with default value
        class MyClass:
            prop = SerializableProperty(default=5)

        obj = MyClass()
        # First access triggers __set__(instance, None) which uses default
        assert obj.prop == 5


# ========== ========== ========== ========== Default Values
class TestDefaultValues:
    """Test static and callable default value behavior."""

    def test_static_default_value(self) -> None:
        # Static default should be used when property is set to None
        class MyClass:
            prop = SerializableProperty(default=42)

        obj = MyClass()
        assert obj.prop == 42

    def test_callable_default_receives_instance(self) -> None:
        # Callable default should receive the instance as argument
        instance_received = None

        def default_func(instance: object) -> int:
            nonlocal instance_received
            instance_received = instance
            return 99

        class MyClass:
            prop = SerializableProperty(default=default_func)

        obj = MyClass()
        _ = obj.prop

        assert instance_received is obj

    def test_callable_default_returns_value(self) -> None:
        # Callable default's return value should be used
        class MyClass:
            prop = SerializableProperty(default=lambda self: [1, 2, 3])

        obj = MyClass()
        assert obj.prop == [1, 2, 3]

    def test_mutable_default_with_callable(self) -> None:
        # Callable defaults should create new instances for each object
        class MyClass:
            items = SerializableProperty(default=lambda self: [])

        obj1 = MyClass()
        obj2 = MyClass()

        obj1.items.append(1)
        obj2.items.append(2)

        assert obj1.items == [1]
        assert obj2.items == [2]

    def test_none_resets_to_default(self) -> None:
        # Setting to None should reset to default value
        class MyClass:
            prop = SerializableProperty(default=10)

        obj = MyClass()
        obj.prop = 50
        assert obj.prop == 50

        obj.prop = None
        assert obj.prop == 10

    def test_default_decorator_sets_callable_default(self) -> None:
        # Using .default() decorator should set callable default
        class MyClass:
            prop = SerializableProperty()

            @prop.default
            def prop(self) -> int:
                return 123

        obj = MyClass()
        assert obj.prop == 123

    def test_none_default_stays_none(self) -> None:
        # If default is None, value should be None
        class MyClass:
            prop = SerializableProperty(default=None)

        obj = MyClass()
        assert obj.prop is None


# ========== ========== ========== ========== Parsers
class TestParsers:
    """Test parser validation and transformation."""

    def test_parser_transforms_value(self) -> None:
        # Parser should transform the input value
        class MyClass:
            prop = SerializableProperty(parser=lambda self, val: val * 2)

        obj = MyClass()
        obj.prop = 5
        assert obj.prop == 10

    def test_parser_receives_instance_and_value(self) -> None:
        # Parser should receive both instance and value
        received_instance = None
        received_value = None

        def parser(instance: object, value: Any) -> Any:
            nonlocal received_instance, received_value
            received_instance = instance
            received_value = value
            return value

        class MyClass:
            prop = SerializableProperty(parser=parser)

        obj = MyClass()
        obj.prop = 42

        assert received_instance is obj
        assert received_value == 42

    def test_parser_applied_to_defaults(self) -> None:
        # Parser should also transform default values
        class MyClass:
            prop = SerializableProperty(
                default=5,
                parser=lambda self, val: val * 10
            )

        obj = MyClass()
        assert obj.prop == 50

    def test_parser_decorator(self) -> None:
        # Using .parser() decorator should set parser function
        class MyClass:
            prop = SerializableProperty(default=0)

            @prop.parser
            def prop(self, value: Any) -> int:
                return int(value)

        obj = MyClass()
        obj.prop = "123"
        assert obj.prop == 123
        assert isinstance(obj.prop, int)

    def test_parser_can_validate_and_reject(self) -> None:
        # Parser can raise exceptions for invalid values
        def validator(instance: object, value: Any) -> int:
            if value < 0:
                raise ValueError("Value must be non-negative")
            return value

        class MyClass:
            prop = SerializableProperty(parser=validator)

        obj = MyClass()
        obj.prop = 10
        assert obj.prop == 10

        with pytest.raises(ValueError, match="must be non-negative"):
            obj.prop = -5

    def test_parser_with_type_conversion(self) -> None:
        # Parser can perform type conversions
        class MyClass:
            prop = SerializableProperty(parser=lambda self, val: str(val).upper())

        obj = MyClass()
        obj.prop = "hello"
        assert obj.prop == "HELLO"

        obj.prop = 123
        assert obj.prop == "123"

    def test_parser_chaining_via_new_descriptor(self) -> None:
        # Creating new descriptor with parser should preserve behavior
        base_prop = SerializableProperty(default=1)
        parsed_prop = base_prop.parser(lambda self, val: val + 10)

        class MyClass:
            prop = parsed_prop

        obj = MyClass()
        obj.prop = 5
        assert obj.prop == 15


# ========== ========== ========== ========== Observers
class TestObservers:
    """Test observer notification on value changes."""

    def test_observer_called_on_change(self, observer_spy: Mock) -> None:
        # Observer should be called when value changes
        class MyClass:
            prop = SerializableProperty(observers={observer_spy})

        obj = MyClass()
        obj.prop = 42

        observer_spy.assert_called()

    def test_observer_receives_instance_old_new(self, observer_spy: Mock) -> None:
        # Observer should receive (instance, old_value, new_value)
        class MyClass:
            prop = SerializableProperty(observers={observer_spy})

        obj = MyClass()
        obj.prop = 10
        obj.prop = 20

        # Second call (first is initialization)
        call_args = observer_spy.call_args_list[1]
        assert call_args[0][0] is obj  # instance
        assert call_args[0][1] == 10  # old value
        assert call_args[0][2] == 20  # new value

    def test_observer_on_first_access(self, observer_spy: Mock) -> None:
        # Observer should be called on first access with (obj, None, default)
        class MyClass:
            prop = SerializableProperty(default=5, observers={observer_spy})

        obj = MyClass()
        _ = obj.prop  # First access triggers initialization

        observer_spy.assert_called_once()
        call_args = observer_spy.call_args[0]
        assert call_args[0] is obj
        assert call_args[1] is None  # old value (not set yet)
        assert call_args[2] == 5  # new value (default)

    def test_multiple_observers_all_called(self) -> None:
        # All observers should be called on change
        observer1 = Mock()
        observer2 = Mock()
        observer3 = Mock()

        class MyClass:
            prop = SerializableProperty(observers={observer1, observer2, observer3})

        obj = MyClass()
        obj.prop = 42

        observer1.assert_called_once()
        observer2.assert_called_once()
        observer3.assert_called_once()

    def test_add_observer_creates_new_descriptor(self, observer_spy: Mock) -> None:
        # add_observer should create new descriptor with added observer
        base_prop = SerializableProperty()
        new_prop = base_prop.add_observer(observer_spy)

        assert base_prop is not new_prop
        assert observer_spy not in base_prop._observers
        assert observer_spy in new_prop._observers

    def test_remove_observer_creates_new_descriptor(self, observer_spy: Mock) -> None:
        # remove_observer should create new descriptor without the observer
        base_prop = SerializableProperty(observers={observer_spy})
        new_prop = base_prop.remove_observer(observer_spy)

        assert base_prop is not new_prop
        assert observer_spy in base_prop._observers
        assert observer_spy not in new_prop._observers

    def test_observer_sees_parsed_value(self, observer_spy: Mock) -> None:
        # Observer should see the parsed value, not the raw input
        class MyClass:
            prop = SerializableProperty(
                parser=lambda self, val: val * 2,
                observers={observer_spy}
            )

        obj = MyClass()
        obj.prop = 5

        call_args = observer_spy.call_args[0]
        assert call_args[2] == 10  # Parsed value, not 5

    def test_observer_can_access_other_properties(self) -> None:
        # Observer can safely access other properties of the instance
        class MyClass:
            prop1 = SerializableProperty()
            prop2 = SerializableProperty(default=0)

            @prop1.add_observer
            def prop1(self, old: Any, new: Any) -> None:
                self.prop2 = new * 2

        obj = MyClass()
        obj.prop1 = 5

        assert obj.prop1 == 5
        assert obj.prop2 == 10


# ========== ========== ========== ========== Post-Initializers
class TestPostInitializers:
    """Test post-initialization hooks."""

    def test_postinitializer_called_on_first_set(self) -> None:
        # Post-initializer should be called after first set
        called = []

        def init(instance: object) -> None:
            called.append(instance)

        class MyClass:
            prop = SerializableProperty(postinitializer=init)

        obj = MyClass()
        obj.prop = 42

        assert len(called) == 1
        assert called[0] is obj

    def test_postinitializer_called_on_first_access(self) -> None:
        # Post-initializer should be called when first accessed (via __get__)
        called = []

        def init(instance: object) -> None:
            called.append(True)

        class MyClass:
            prop = SerializableProperty(default=5, postinitializer=init)

        obj = MyClass()
        _ = obj.prop  # First access

        assert len(called) == 1

    def test_postinitializer_only_called_once(self) -> None:
        # Post-initializer should only be called once per instance
        call_count = []

        def init(instance: object) -> None:
            call_count.append(1)

        class MyClass:
            prop = SerializableProperty(postinitializer=init)

        obj = MyClass()
        obj.prop = 1
        obj.prop = 2
        obj.prop = 3
        _ = obj.prop

        assert len(call_count) == 1

    def test_postinitializer_runs_after_value_set(self) -> None:
        # Post-initializer should run after value is set
        class MyClass:
            prop = SerializableProperty(default=10)

            @prop.postinitializer
            def prop(self) -> None:
                # At this point, self.prop should already be set
                assert self.prop == 10

        obj = MyClass()
        _ = obj.prop  # Should not raise

    def test_postinitializer_can_modify_other_properties(self) -> None:
        # Post-initializer can set up related properties
        class MyClass:
            prop1 = SerializableProperty(default=5)
            prop2 = SerializableProperty(default=0)

            @prop1.postinitializer
            def prop1(self) -> None:
                self.prop2 = self.prop1 * 2

        obj = MyClass()
        _ = obj.prop1  # Triggers initialization

        assert obj.prop1 == 5
        assert obj.prop2 == 10

    def test_postinitializer_can_modify_own_property(self) -> None:
        # Post-initializer can modify its own property value
        class MyClass:
            prop = SerializableProperty(default=1)

            @prop.postinitializer
            def prop(self) -> None:
                self.prop = self.prop + 10

        obj = MyClass()
        _ = obj.prop

        assert obj.prop == 11

    def test_postinitializer_no_infinite_recursion(self) -> None:
        # Modifying property in post-initializer should not cause recursion
        class MyClass:
            prop = SerializableProperty(default=1)

            @prop.postinitializer
            def prop(self) -> None:
                # This sets the value again, but shouldn't trigger another init
                self.prop = self.prop + 1

        obj = MyClass()
        obj.prop = 5  # Should not hang or recurse infinitely

        assert obj.prop == 6

    def test_postinitializer_decorator(self) -> None:
        # Using .postinitializer() decorator should set the hook
        class MyClass:
            prop = SerializableProperty(default=0)
            other = SerializableProperty(default=0)

            @prop.postinitializer
            def prop(self) -> None:
                self.other = 999

        obj = MyClass()
        _ = obj.prop

        assert obj.other == 999

    def test_postinitializer_independent_per_instance(self) -> None:
        # Post-initializer should run independently for each instance
        instances_initialized = []

        def init(instance: object) -> None:
            instances_initialized.append(instance)

        class MyClass:
            prop = SerializableProperty(postinitializer=init)

        obj1 = MyClass()
        obj2 = MyClass()

        obj1.prop = 1
        obj2.prop = 2

        assert len(instances_initialized) == 2
        assert instances_initialized[0] is obj1
        assert instances_initialized[1] is obj2

    def test_postinitializer_with_parser_and_observer(self) -> None:
        # Post-initializer should work together with parsers and observers
        init_called = []
        observer_called = []

        def init(instance: object) -> None:
            init_called.append(True)

        def observer(instance: object, old: Any, new: Any) -> None:
            observer_called.append((old, new))

        class MyClass:
            prop = SerializableProperty(
                default=5,
                parser=lambda self, val: val * 2,
                observers={observer},
                postinitializer=init
            )

        obj = MyClass()
        _ = obj.prop

        assert len(init_called) == 1
        assert len(observer_called) == 1
        assert observer_called[0] == (None, 10)  # Parser applied to default


# ========== ========== ========== ========== Write-Once
class TestWriteOnce:
    """Test write-once property semantics."""

    def test_writeonce_allows_first_set(self) -> None:
        # Write-once property should allow first assignment
        class MyClass:
            prop = SerializableProperty(writeonce=True)

        obj = MyClass()
        obj.prop = 42
        assert obj.prop == 42

    def test_writeonce_prevents_second_set(self) -> None:
        # Write-once property should prevent reassignment
        class MyClass:
            prop = SerializableProperty(writeonce=True)

        obj = MyClass()
        obj.prop = 42

        with pytest.raises(AttributeError, match="write-once property"):
            obj.prop = 100

    def test_writeonce_allows_none_as_first_value(self) -> None:
        # Write-once should allow None as the first value
        class MyClass:
            prop = SerializableProperty(writeonce=True, default=10)

        obj = MyClass()
        # First access sets to default
        _ = obj.prop
        assert obj.prop == 10

        # Cannot change after first access
        with pytest.raises(AttributeError, match="write-once property"):
            obj.prop = 20

    def test_writeonce_with_explicit_none_set(self) -> None:
        # Explicitly setting to None should count as "set" for write-once
        class MyClass:
            prop = SerializableProperty(writeonce=True, default=5)

        obj = MyClass()
        obj.prop = None  # Explicitly set to None (which becomes default)

        with pytest.raises(AttributeError, match="write-once property"):
            obj.prop = 10

    def test_writeonce_independent_per_instance(self) -> None:
        # Write-once should be enforced independently per instance
        class MyClass:
            prop = SerializableProperty(writeonce=True)

        obj1 = MyClass()
        obj2 = MyClass()

        obj1.prop = 10
        obj2.prop = 20

        assert obj1.prop == 10
        assert obj2.prop == 20

        with pytest.raises(AttributeError):
            obj1.prop = 30

        with pytest.raises(AttributeError):
            obj2.prop = 40

    def test_writeonce_property_is_readonly(self) -> None:
        # The writeonce property attribute should be readable
        class MyClass:
            prop = SerializableProperty(writeonce=True)

        assert MyClass.prop.writeonce is True

    def test_non_writeonce_allows_multiple_sets(self) -> None:
        # Non-write-once properties should allow reassignment
        class MyClass:
            prop = SerializableProperty(writeonce=False)

        obj = MyClass()
        obj.prop = 1
        obj.prop = 2
        obj.prop = 3
        assert obj.prop == 3


# ========== ========== ========== ========== Copiable
class TestCopiable:
    """Test copiable flag (for future serialization features)."""

    def test_copiable_flag_stored(self) -> None:
        # Copiable flag should be stored and accessible
        class MyClass:
            prop1 = SerializableProperty(copiable=True)
            prop2 = SerializableProperty(copiable=False)

        assert MyClass.prop1.copiable is True
        assert MyClass.prop2.copiable is False

    def test_copiable_default_is_true(self) -> None:
        # Default value of copiable should be True
        class MyClass:
            prop = SerializableProperty()

        assert MyClass.prop.copiable is True


# ========== ========== ========== ========== Edge Cases
class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_descriptor_immutability(self) -> None:
        # Decorator methods should create new descriptors, not mutate
        base = SerializableProperty(default=1)
        modified = base.parser(lambda self, val: val * 2)

        assert base is not modified
        assert base._parser is None
        assert modified._parser is not None

    def test_multiple_properties_on_same_class(self) -> None:
        # Multiple SerializableProperties should work independently
        class MyClass:
            prop1 = SerializableProperty(default=10)
            prop2 = SerializableProperty(default=20)
            prop3 = SerializableProperty(default=30)

        obj = MyClass()
        assert obj.prop1 == 10
        assert obj.prop2 == 20
        assert obj.prop3 == 30

        obj.prop1 = 100
        obj.prop2 = 200

        assert obj.prop1 == 100
        assert obj.prop2 == 200
        assert obj.prop3 == 30

    def test_inheritance_properties_work(self) -> None:
        # Properties should work correctly with inheritance
        class Base:
            prop1 = SerializableProperty(default=1)

        class Derived(Base):
            prop2 = SerializableProperty(default=2)

        obj = Derived()
        assert obj.prop1 == 1
        assert obj.prop2 == 2

    def test_property_override_in_subclass(self) -> None:
        # Subclass can override property definition
        class Base:
            prop = SerializableProperty(default=10)

        class Derived(Base):
            prop = SerializableProperty(default=20)

        base_obj = Base()
        derived_obj = Derived()

        assert base_obj.prop == 10
        assert derived_obj.prop == 20

    def test_observer_exceptions_propagate(self) -> None:
        # Exceptions in observers should propagate to caller
        def bad_observer(instance: object, old: Any, new: Any) -> None:
            raise RuntimeError("Observer error")

        class MyClass:
            prop = SerializableProperty(observers={bad_observer})

        obj = MyClass()

        with pytest.raises(RuntimeError, match="Observer error"):
            obj.prop = 42

    def test_parser_exceptions_propagate(self) -> None:
        # Exceptions in parsers should propagate to caller
        def bad_parser(instance: object, value: Any) -> Any:
            raise ValueError("Parser error")

        class MyClass:
            prop = SerializableProperty(parser=bad_parser)

        obj = MyClass()

        with pytest.raises(ValueError, match="Parser error"):
            obj.prop = 42

    def test_postinitializer_exceptions_propagate(self) -> None:
        # Exceptions in post-initializers should propagate
        def bad_init(instance: object) -> None:
            raise RuntimeError("Init error")

        class MyClass:
            prop = SerializableProperty(postinitializer=bad_init)

        obj = MyClass()

        with pytest.raises(RuntimeError, match="Init error"):
            obj.prop = 42

    def test_complex_default_factory(self) -> None:
        # Complex default factory should work correctly
        class MyClass:
            data = SerializableProperty()

            @data.default
            def data(self) -> dict[str, list[int]]:
                return {"items": [], "count": [0]}

        obj1 = MyClass()
        obj2 = MyClass()

        obj1.data["items"].append(1)
        obj2.data["items"].append(2)

        assert obj1.data["items"] == [1]
        assert obj2.data["items"] == [2]

    def test_set_name_attributes_exist(self) -> None:
        # __set_name__ should set name, owner, private_name
        class MyClass:
            my_property = SerializableProperty()

        descriptor = MyClass.my_property
        assert descriptor.name == "my_property"
        assert descriptor.owner is MyClass
        assert descriptor.private_name == "_serializable_property__my_property"

    def test_documentation_preserved(self) -> None:
        # Documentation string should be preserved
        doc_string = "This is a documented property"

        class MyClass:
            prop = SerializableProperty(doc=doc_string)

        assert MyClass.prop.__doc__ == doc_string

    def test_none_documentation(self) -> None:
        # None documentation should be allowed
        class MyClass:
            prop = SerializableProperty(doc=None)

        assert MyClass.prop.__doc__ is None


# ========== ========== ========== ========== Integration Tests
class TestIntegration:
    """Test complex scenarios combining multiple features."""

    def test_full_featured_property(self) -> None:
        # Property with all features working together
        observer_calls = []
        init_calls = []

        def observer(instance: object, old: Any, new: Any) -> None:
            observer_calls.append((old, new))

        def init(instance: object) -> None:
            init_calls.append(instance)

        class MyClass:
            prop = SerializableProperty(
                default=lambda self: 10,
                parser=lambda self, val: val * 2 if val is not None else None,
                observers={observer},
                postinitializer=init,
                writeonce=False,
                copiable=True,
                doc="A fully-featured property"
            )

        obj = MyClass()

        # First access
        assert obj.prop == 20  # default 10 * 2
        assert len(observer_calls) == 1
        assert observer_calls[0] == (None, 20)
        assert len(init_calls) == 1

        # Set new value
        obj.prop = 5
        assert obj.prop == 10  # 5 * 2
        assert len(observer_calls) == 2
        assert observer_calls[1] == (20, 10)
        assert len(init_calls) == 1  # Still only called once

    def test_scientific_use_case_temperature(self) -> None:
        # Realistic scientific use case: temperature in Kelvin
        class Experiment:
            # Temperature must be positive and in Kelvin
            temperature = SerializableProperty(
                default=293.15,  # Room temperature
                parser=lambda self, val: max(0.0, float(val))
            )

        exp = Experiment()
        assert exp.temperature == 293.15

        exp.temperature = 373.15  # Boiling water
        assert exp.temperature == 373.15

        exp.temperature = -100  # Invalid, clamps to 0
        assert exp.temperature == 0.0

    def test_lazy_loaded_data_property(self) -> None:
        # Use case: lazy-loaded data with post-initialization
        class DataContainer:
            data = SerializableProperty(default=None)
            _loaded = False

            @data.postinitializer
            def data(self) -> None:
                # Simulate expensive data loading
                if self.data is None:
                    self.data = list(range(1000))
                self._loaded = True

        container = DataContainer()
        assert not container._loaded

        # First access triggers loading
        _ = container.data
        assert container._loaded
        assert len(container.data) == 1000

    def test_cascading_properties(self) -> None:
        # Use case: properties that update each other via observers
        class System:
            celsius = SerializableProperty(default=0.0)
            fahrenheit = SerializableProperty(default=32.0)

            @celsius.add_observer
            def celsius(self, old: float, new: float) -> None:
                # Update fahrenheit when celsius changes
                # Avoid infinite loop by checking if update needed
                expected_f = new * 9 / 5 + 32
                if abs(self.fahrenheit - expected_f) > 0.01:
                    self.fahrenheit = expected_f

            @fahrenheit.add_observer
            def fahrenheit(self, old: float, new: float) -> None:
                # Update celsius when fahrenheit changes
                expected_c = (new - 32) * 5 / 9
                if abs(self.celsius - expected_c) > 0.01:
                    self.celsius = expected_c

        sys = System()
        sys.celsius = 100  # Boiling point
        assert abs(sys.fahrenheit - 212) < 0.1

        sys.fahrenheit = 32  # Freezing point
        assert abs(sys.celsius - 0) < 0.1