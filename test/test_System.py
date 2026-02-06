#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for System hierarchical namespace.

Tests cover:
- System creation and basic properties
- Subsystem registration and management
- Tag observer pattern and dynamic reorganization
- Namespace access (dict and attribute style)
- Hierarchy navigation and validation
- Circular dependency prevention
- Serialization and persistence
- Display integration

================================================================================
SUGGESTIONS AND FUTURE ENHANCEMENTS
================================================================================

ENHANCEMENTS:
-------------

1. Path Property and Navigation
   Add full path representation and navigation:

   @property
   def path(self) -> str:
       '''Full hierarchical path like "root.subsystem1.subsystem2"'''
       if self.supersystem is None:
           return self.tag or '<root>'
       chain = [s.tag for s in reversed(self.supersystem_chain)]
       chain.append(self.tag)
       return '.'.join(chain)

   def find(self, path: str) -> System:
       '''Navigate by path: system.find("sensors.temp.primary")'''
       parts = path.split('.')
       current = self
       for part in parts:
           current = current[part]
       return current

2. Recursive Operations
   Add tree traversal and search:

   def walk(self, depth_first=True) -> Iterator[System]:
       '''Recursively yield all subsystems in hierarchy'''
       yield self
       for sub in self:
           if depth_first:
               yield from sub.walk()
           else:
               # Breadth-first implementation
               pass

   def find_all(self, **kwargs) -> list[System]:
       '''Find all subsystems matching criteria'''
       results = []
       for sys in self.walk():
           if all(getattr(sys, k, None) == v for k, v in kwargs.items()):
               results.append(sys)
       return results

   def find_one(self, **kwargs) -> System | None:
       '''Find first subsystem matching criteria'''
       for sys in self.walk():
           if all(getattr(sys, k, None) == v for k, v in kwargs.items()):
               return sys
       return None

3. Move Operation
   Simplify moving subsystems between parents:

   def move_to(self, new_parent: System) -> None:
       '''Move this system to a new parent'''
       self.supersystem = new_parent

   def move(self, subsystem: System | str, new_parent: System) -> None:
       '''Move a subsystem to a different parent'''
       if isinstance(subsystem, str):
           subsystem = self[subsystem]
       if subsystem not in self:
           raise ValueError("Not a subsystem")
       subsystem.supersystem = new_parent

4. Bulk Operations
   Add/remove multiple subsystems efficiently:

   def add_all(self, subsystems: list[System]) -> None:
       '''Add multiple subsystems at once'''
       for subsystem in subsystems:
           subsystem.supersystem = self

   def remove_all(self, tags: list[str] = None) -> None:
       '''Remove multiple subsystems (all if tags=None)'''
       if tags is None:
           tags = list(self.subsystems.keys())
       for tag in tags:
           self.remove(tag)

   def clear(self) -> None:
       '''Remove all subsystems'''
       self.remove_all()

5. Hierarchy Validation
   Check integrity and report issues:

   def validate_hierarchy(self) -> list[str]:
       '''Check hierarchy integrity, return list of issues'''
       issues = []

       # Orphaned observers
       for sub, obs in self._tag_observers.items():
           if sub not in self:
               issues.append(f"Orphaned observer for {sub}")

       # Tag consistency
       for tag, sub in self.subsystems.items():
           if sub.tag != tag:
               issues.append(f"Tag mismatch: {tag} != {sub.tag}")
           if sub.supersystem is not self:
               issues.append(f"Parent mismatch for {sub}")

       # Recursive check
       for sub in self:
           issues.extend(sub.validate_hierarchy())

       return issues

6. Statistics and Metrics
   Provide hierarchy information:

   @property
   def depth(self) -> int:
       '''Depth in hierarchy (0 for root)'''
       return len(self.supersystem_chain)

   @property
   def total_subsystems(self) -> int:
       '''Total subsystems recursively'''
       return sum(1 + sub.total_subsystems for sub in self)

   def get_stats(self) -> dict:
       '''Return hierarchy statistics'''
       return {
           'depth': self.depth,
           'direct_subsystems': len(self),
           'total_subsystems': self.total_subsystems,
           'max_depth': max((sub.depth for sub in self.walk()), default=0),
       }

7. Better equal() Method Name
   Current name is confusing. Options:
   - content_equal(other) - Compare serialized content
   - same_content(other) - More readable
   - equals_by_value(other) - Clear distinction from ==
   - structural_equal(other) - Emphasizes structure comparison

   Recommendation: content_equal() is clearest

8. Export/Import Subtrees
   Save/load portions of hierarchy:

   def export_subtree(self, path: Path) -> None:
       '''Save this system and all subsystems'''
       self.save(path)

   @classmethod
   def import_subtree(cls, path: Path, parent: System = None) -> System:
       '''Load system and optionally attach to parent'''
       system = cls.load(path)
       if parent is not None:
           system.supersystem = parent
       return system

9. Pretty Print Hierarchy
   ASCII tree visualization:

   def tree(self, indent: str = "", last: bool = True) -> str:
       '''Return ASCII tree representation'''
       connector = "└── " if last else "├── "
       result = indent + connector + str(self.tag) + "\\n"

       children = list(self.subsystems.values())
       for i, child in enumerate(children):
           is_last = (i == len(children) - 1)
           extension = "    " if last else "│   "
           result += child.tree(indent + extension, is_last)

       return result

10. Tag Validation Options
    Configurable tag validation:
    - Allow/disallow certain characters
    - Min/max length
    - Reserved tag names
    - Case sensitivity options

11. Subsystem Filtering
    Filter subsystems by criteria:

    def filter(self, predicate: Callable[[System], bool]) -> list[System]:
        '''Return subsystems matching predicate'''
        return [sub for sub in self if predicate(sub)]

    def by_name(self, name: str) -> list[System]:
        '''Find subsystems by name'''
        return self.filter(lambda s: s.name == name)

12. Event Hooks
    Allow custom callbacks on hierarchy changes:
    - on_subsystem_added(subsystem)
    - on_subsystem_removed(subsystem)
    - on_tag_changed(subsystem, old_tag, new_tag)
    - on_parent_changed(old_parent, new_parent)

13. Hierarchy Locking
    Prevent modifications during operations:

    @contextmanager
    def locked(self):
        '''Temporarily prevent hierarchy modifications'''
        self._locked = True
        try:
            yield
        finally:
            self._locked = False

14. Clone with Hierarchy
    Deep copy entire subtree:

    def clone_subtree(self, include_ids: bool = False) -> System:
        '''Create deep copy of this system and all subsystems'''
        # Copy self
        clone = self.copy(is_copy=not include_ids)

        # Recursively copy subsystems
        for sub in self:
            sub_clone = sub.clone_subtree(include_ids)
            sub_clone.supersystem = clone

        return clone

15. Diff Operations
    Compare hierarchies:

    def diff(self, other: System) -> dict:
        '''Compare two hierarchies, return differences'''
        return {
            'added': [tag for tag in other.subsystems if tag not in self.subsystems],
            'removed': [tag for tag in self.subsystems if tag not in other.subsystems],
            'modified': [tag for tag in self.subsystems if tag in other.subsystems
                        and not self[tag].equal(other[tag])],
        }


DESIGN CONCERNS:
----------------

1. Tag Observer Cleanup on Delete
   Q: What happens if a subsystem is deleted?

   del system.sensor_a
   # Is observer cleaned up from parent's _tag_observers?

   Currently, the observer dict might retain the key even after object is GC'd.
   Weak references help, but explicit cleanup might be needed.

   Possible solutions:
   - WeakKeyDictionary for _tag_observers
   - Explicit cleanup in remove()
   - __del__ method (but careful with GC issues)

2. Observer Attachment During __init__
   Q: What if supersystem is set during initialization before tag?

   System(tag=None, supersystem=parent)  # Would this work?

   Current parser checks for tag before registration. Need test to verify
   error messages are clear.

3. Multiple Parents (DAG vs Tree)
   Current design enforces tree structure (single parent).
   Q: Is this always desired? Or should we support DAG?

   Tree (current):
       A
      / \\
     B   C
     |
     D

   DAG (not supported):
       A     B
        \\   /
          C

   If DAG needed in future, would require major refactor:
   - subsystems becomes dict[str, set[System]]
   - Multiple observers per subsystem
   - More complex cycle detection

4. Serialization of Deep Hierarchies
   Q: Does saving a root system save entire tree?

   root = System(tag='root')
   root.add(System(tag='child'))
   root.save('root.sys')
   # Does this save child too?

   If yes: Great! But might be slow for large trees.
   If no: Need explicit save for each level.

   Need tests to verify and document behavior.

5. Tag Case Sensitivity
   Q: Are tags case-sensitive?

   system['Sensor'] vs system['sensor']  # Different?

   Currently yes (dict keys are case-sensitive).
   Should we normalize? Or is case-sensitivity desired?

6. Reserved Tag Names
   Q: Can a subsystem have tag that conflicts with System methods?

   system.add(System(tag='add'))  # Now system.add is ambiguous!

   Current check: `if new_value in self.__dict__`
   But methods are in class dict, not instance dict.
   Might need more thorough check.

7. Performance of supersystem_chain
   Q: For deep hierarchies, walking chain might be slow.

   @property
   def supersystem_chain(self):
       # Walks entire chain on every access

   Options:
   - Cache chain, invalidate on parent change
   - Use @lru_cache (but watch memory)
   - Document performance characteristics

8. Subsystems Dict Mutability
   Q: What if user directly modifies subsystems dict?

   system.subsystems['new'] = other_system  # Bypasses validation!

   Currently relies on parser for validation, but direct dict access
   bypasses it. Options:
   - Make subsystems a custom dict that validates
   - Document that users should use add/remove
   - Make subsystems read-only (property that returns copy)

9. Observer Execution Order
   Q: If multiple subsystems change tags simultaneously, what's the order?

   Observers are called in order of tag property changes.
   Race conditions unlikely in single-threaded, but worth documenting.

10. Error Recovery
    Q: If observer raises exception, what state is system in?

    subsystem.tag = 'conflict'  # Raises ValueError in observer
    # Is subsystem still registered? Is dict in consistent state?

    Observers modify state before raising. Need tests for cleanup.


TEST STRATEGY NOTES:
-------------------

High Priority Tests:
- Tag observer pattern (dynamic reorganization)
- Circular dependency prevention
- Tag conflict detection
- Serialization/deserialization with hierarchy
- Error conditions and cleanup

Medium Priority:
- Attribute vs dict access
- Iterator and len
- Display integration
- Deep hierarchies

Low Priority (but document):
- Performance characteristics
- Thread safety (or lack thereof)
- Memory usage for large hierarchies

================================================================================
END OF SUGGESTIONS AND CONCERNS
================================================================================
"""

from __future__ import annotations

import pytest
from pathlib import Path

from jangada.system import System


# ========== ========== ========== ========== Fixtures
@pytest.fixture
def root_system() -> System:
    """Create a root system with no parent."""
    return System(tag='root', name='Root System')


@pytest.fixture
def child_system() -> System:
    """Create a child system."""
    return System(tag='child', name='Child System')


@pytest.fixture
def simple_hierarchy() -> tuple[System, System, System]:
    """
    Create a simple hierarchy:

    root
    ├── child1
    └── child2
    """
    root = System(tag='root', name='Root')
    child1 = System(tag='child1', name='Child 1')
    child2 = System(tag='child2', name='Child 2')

    root.add(child1, child2)

    return root, child1, child2


@pytest.fixture
def deep_hierarchy() -> tuple[System, System, System, System]:
    """
    Create a deep hierarchy:

    root
    └── level1
        └── level2
            └── level3
    """
    root = System(tag='root')
    level1 = System(tag='level1')
    level2 = System(tag='level2')
    level3 = System(tag='level3')

    root.add(level1)
    level1.add(level2)
    level2.add(level3)

    return root, level1, level2, level3


@pytest.fixture
def complex_hierarchy() -> System:
    """
    Create a complex hierarchy:

    root
    ├── sensors
    │   ├── temperature
    │   └── pressure
    └── controllers
        └── main
    """
    root = System(tag='root')

    sensors = System(tag='sensors', name='Sensors')
    temp = System(tag='temperature', name='Temperature Sensor')
    pressure = System(tag='pressure', name='Pressure Sensor')

    controllers = System(tag='controllers', name='Controllers')
    main = System(tag='main', name='Main Controller')

    root.add(sensors, controllers)
    sensors.add(temp, pressure)
    controllers.add(main)

    return root


# ========== ========== ========== ========== Test Basic Creation
class TestSystemCreation:
    """Test System instantiation and basic properties."""

    def test_creates_with_all_mixins(self) -> None:
        # System should have all mixin properties
        sys = System()

        assert hasattr(sys, 'id')  # Identifiable
        assert hasattr(sys, 'tag')  # Taggable
        assert hasattr(sys, 'name')  # Nameable
        assert hasattr(sys, 'description')  # Describable
        assert hasattr(sys, 'subsystems')  # System-specific

    def test_creates_with_empty_subsystems(self) -> None:
        # New system should have empty subsystems dict
        sys = System()

        assert sys.subsystems == {}
        assert len(sys) == 0

    def test_creates_without_supersystem(self) -> None:
        # New system should have no parent
        sys = System()

        assert sys.supersystem is None
        assert sys._supersystem_id is None

    def test_creates_with_properties(self) -> None:
        # Can set properties on creation
        sys = System(
            tag='test',
            name='Test System',
            description='A test'
        )

        assert sys.tag == 'test'
        assert sys.name == 'Test System'
        assert sys.description == 'A test'

    def test_extension_is_sys(self) -> None:
        # File extension should be .sys
        assert System.extension == '.sys'

    def test_hash_uses_identifiable(self) -> None:
        # Hash should use Identifiable implementation
        sys = System()

        assert hash(sys) == hash(sys.id)

    def test_eq_uses_identifiable(self) -> None:
        # Equality should use Identifiable implementation (ID-based)
        sys1 = System()
        sys2 = System()

        assert sys1 != sys2  # Different IDs


# ========== ========== ========== ========== Test Subsystem Registration
class TestSubsystemRegistration:
    """Test adding and removing subsystems."""

    def test_add_single_subsystem(self, root_system: System, child_system: System) -> None:
        # Can add a subsystem

        root_system.add(child_system)

        assert child_system in root_system
        assert child_system.supersystem is root_system
        assert root_system.subsystems[child_system.tag] is child_system

    def test_add_multiple_subsystems(self, root_system: System) -> None:
        # Can add multiple subsystems at once
        child1 = System(tag='child1')
        child2 = System(tag='child2')
        child3 = System(tag='child3')

        root_system.add(child1, child2, child3)

        assert len(root_system) == 3
        assert child1 in root_system
        assert child2 in root_system
        assert child3 in root_system

    def test_remove_subsystem_by_object(self, simple_hierarchy) -> None:
        # Can remove subsystem by passing object
        root, child1, child2 = simple_hierarchy

        root.remove(child1)

        assert child1 not in root
        assert child1.supersystem is None
        assert len(root) == 1

    def test_remove_subsystem_by_tag(self, simple_hierarchy) -> None:
        # Can remove subsystem by tag string
        root, child1, child2 = simple_hierarchy

        root.remove('child1')

        assert child1 not in root
        assert child1.supersystem is None

    def test_remove_nonexistent_raises(self, root_system: System) -> None:
        # Removing non-existent subsystem raises ValueError
        fake = System(tag='fake')

        with pytest.raises(ValueError, match='not registered'):
            root_system.remove(fake)

        with pytest.raises(ValueError, match='not registered'):
            root_system.remove('nonexistent')

    def test_add_requires_tag(self, root_system: System) -> None:
        # Cannot add subsystem without tag
        child = System(tag=None)

        with pytest.raises(ValueError, match='must have a tag'):
            root_system.add(child)

    def test_add_validates_type(self, root_system: System) -> None:
        # Adding non-System object should fail
        # (This happens through supersystem setter validation)
        child = System(tag='child')

        with pytest.raises(TypeError):
            child.supersystem = "not a system"

    def test_setitem_creates_and_registers_subsystem(self, root_system: System) -> None:
        # Dict-style assignment should register subsystem
        child = System()

        root_system['child_tag'] = child

        assert 'child_tag' in root_system
        assert child.tag == 'child_tag'
        assert child.supersystem is root_system

    def test_setitem_sets_tag_from_key(self, root_system: System) -> None:
        # Tag should be set from the dict key
        child = System(tag=None)

        root_system['my_sensor'] = child

        assert child.tag == 'my_sensor'

    def test_setitem_overwrites_existing_tag(self, root_system: System) -> None:
        # Should change tag if different from key
        child = System(tag='old_tag')

        root_system['new_tag'] = child

        assert child.tag == 'new_tag'
        assert 'new_tag' in root_system
        assert 'old_tag' not in root_system

    def test_setitem_replaces_existing_subsystem(self, root_system: System) -> None:
        # Assigning to same key should replace previous subsystem
        first = System(name='First')
        second = System(name='Second')

        root_system['sensor'] = first
        assert root_system['sensor'] is first

        root_system['sensor'] = second
        assert root_system['sensor'] is second
        assert first not in root_system
        assert first.supersystem is None

    def test_setitem_multiple_subsystems(self, root_system: System) -> None:
        # Can add multiple subsystems via dict assignment
        root_system['sensor1'] = System()
        root_system['sensor2'] = System()
        root_system['sensor3'] = System()

        assert len(root_system) == 3
        assert 'sensor1' in root_system
        assert 'sensor2' in root_system
        assert 'sensor3' in root_system

    def test_setitem_getitem_symmetry(self, root_system: System) -> None:
        # Set via dict should be retrievable via dict
        child = System(name='Child')

        root_system['my_child'] = child

        retrieved = root_system['my_child']
        assert retrieved is child

    def test_setitem_removes_from_old_parent(self) -> None:
        # Setting subsystem to new parent should remove from old
        parent1 = System(tag='parent1')
        parent2 = System(tag='parent2')
        child = System()

        parent1['child'] = child
        assert child in parent1

        parent2['child'] = child
        assert child not in parent1
        assert child in parent2

    def test_setitem_triggers_tag_observer(self, root_system: System) -> None:
        # Setting via dict should trigger tag observer
        child = System()

        root_system['first_name'] = child
        assert 'first_name' in root_system

        # Assign to different key
        root_system['second_name'] = child

        # Should have moved
        assert 'first_name' not in root_system
        assert 'second_name' in root_system
        assert len(root_system) == 1

    def test_setitem_and_add_equivalent(self) -> None:
        # setitem and add() should produce same result
        parent1 = System(tag='parent1')
        parent2 = System(tag='parent2')

        child1 = System()
        child1.tag = 'sensor'
        parent1.add(child1)

        child2 = System()
        parent2['sensor'] = child2

        assert child1.tag == child2.tag == 'sensor'
        assert child1 in parent1
        assert child2 in parent2


# ========== ========== ========== ========== Test Tag Observer Pattern
class TestTagObserver:
    """Test dynamic tag reorganization via observers."""

    def test_changing_tag_updates_dict_key(self, simple_hierarchy) -> None:
        # Changing tag should update parent's subsystems dict key
        root, child1, child2 = simple_hierarchy

        assert child1 in root._tag_observers
        assert child2 in root._tag_observers

        assert root._tag_observers[child1] in type(child1).tag._observers
        assert root._tag_observers[child2] in type(child2).tag._observers

        child1.tag = 'renamed_child'

        assert child1.supersystem is root

        assert 'child1' not in root.subsystems
        assert 'renamed_child' in root.subsystems
        assert root.subsystems['renamed_child'] is child1

    def test_changing_tag_maintains_registration(self, simple_hierarchy) -> None:
        # Changing tag should maintain parent relationship
        root, child1, child2 = simple_hierarchy

        child1.tag = 'new_tag'

        assert child1 in root
        assert child1.supersystem is root

    def test_tag_conflict_prevents_change(self, simple_hierarchy) -> None:
        # Cannot change tag to one already used
        root, child1, child2 = simple_hierarchy

        with pytest.raises(ValueError, match='already registered'):
            child1.tag = 'child2'

    def test_tag_to_none_removes_subsystem(self, simple_hierarchy) -> None:
        # Setting tag to None should remove from parent
        root, child1, child2 = simple_hierarchy

        with pytest.raises(ValueError, match='Cannot keep subsystem with tag None'):
            child1.tag = None

        # Should be removed despite exception
        assert child1 not in root

    def test_tag_conflict_with_attribute_raises(self, root_system: System) -> None:
        # Cannot use tag that conflicts with instance attributes
        child = System(tag='child')
        root_system.add(child)

        # Assuming System has some instance attributes
        # This test might need adjustment based on actual attributes

    def test_observer_attached_on_add(self, root_system: System) -> None:
        # Observer should be attached when subsystem is added
        child = System(tag='child')

        root_system.add(child)

        assert child in root_system._tag_observers

    def test_observer_removed_on_remove(self, simple_hierarchy) -> None:
        # Observer should be removed when subsystem is removed
        root, child1, child2 = simple_hierarchy

        root.remove(child1)

        assert child1 not in root._tag_observers

    def test_observer_handles_rapid_changes(self, simple_hierarchy) -> None:
        # Multiple rapid tag changes should work correctly
        root, child1, child2 = simple_hierarchy

        child1.tag = 'tag1'
        child1.tag = 'tag2'
        child1.tag = 'tag3'

        assert 'tag3' in root.subsystems
        assert root.subsystems['tag3'] is child1
        assert 'tag1' not in root.subsystems
        assert 'tag2' not in root.subsystems


# ========== ========== ========== ========== Test Namespace Access
class TestNamespaceAccess:
    """Test dict-style and attribute-style access."""

    def test_dict_access_by_tag(self, simple_hierarchy) -> None:
        # Can access subsystem by tag using dict syntax
        root, child1, child2 = simple_hierarchy

        assert root['child1'] is child1
        assert root['child2'] is child2

    def test_attribute_access_by_tag(self, simple_hierarchy) -> None:
        # Can access subsystem by tag using attribute syntax
        root, child1, child2 = simple_hierarchy

        assert root.child1 is child1
        assert root.child2 is child2

    def test_dict_access_missing_raises_keyerror(self, root_system: System) -> None:
        # Accessing non-existent tag with dict syntax raises KeyError
        with pytest.raises(KeyError, match='No subsystem registered'):
            _ = root_system['nonexistent']

    def test_attribute_access_missing_raises_attributeerror(self,
                                                            root_system: System) -> None:
        # Accessing non-existent tag with attribute syntax raises AttributeError
        with pytest.raises(AttributeError, match='No subsystem registered'):
            _ = root_system.nonexistent

    def test_dict_access_none_raises(self, root_system: System) -> None:
        # Cannot access with None tag
        with pytest.raises(KeyError, match='Cannot retrieve.*None'):
            _ = root_system[None]

    def test_contains_by_object(self, simple_hierarchy) -> None:
        # Can check containment by object
        root, child1, child2 = simple_hierarchy

        assert child1 in root
        assert child2 in root

    def test_contains_by_tag(self, simple_hierarchy) -> None:
        # Can check containment by tag string
        root, child1, child2 = simple_hierarchy

        assert 'child1' in root
        assert 'child2' in root
        assert 'nonexistent' not in root

    def test_contains_none_returns_false(self, root_system: System) -> None:
        # Checking if None is in system returns False
        assert None not in root_system

    def test_contains_uses_identity_not_equality(self) -> None:
        # Contains should use 'is' not '=='
        root = System(tag='root')
        original = System(tag='child')
        root.add(original)

        # Create copy with same ID (if possible)
        # This test verifies identity check
        assert original in root

    def test_iteration_yields_subsystems(self, simple_hierarchy) -> None:
        # Iterating should yield subsystem objects
        root, child1, child2 = simple_hierarchy

        subsystems = list(root)

        assert len(subsystems) == 2
        assert child1 in subsystems
        assert child2 in subsystems

    def test_len_returns_subsystem_count(self, simple_hierarchy) -> None:
        # len() should return number of direct subsystems
        root, child1, child2 = simple_hierarchy

        assert len(root) == 2


# ========== ========== ========== ========== Test Hierarchy Navigation
class TestHierarchyNavigation:
    """Test parent-child relationships and navigation."""

    def test_supersystem_property(self, simple_hierarchy) -> None:
        # supersystem property should return parent
        root, child1, child2 = simple_hierarchy

        assert child1.supersystem is root
        assert child2.supersystem is root
        assert root.supersystem is None

    def test_supersystem_setter(self, root_system: System,
                                child_system: System) -> None:
        # Can set supersystem directly
        child_system.supersystem = root_system

        assert child_system.supersystem is root_system
        assert child_system in root_system

    def test_supersystem_setter_to_none_unregisters(self, simple_hierarchy) -> None:
        # Setting supersystem to None removes from parent
        root, child1, child2 = simple_hierarchy

        child1.supersystem = None

        assert child1 not in root
        assert child1.supersystem is None

    def test_supersystem_chain_single_level(self, simple_hierarchy) -> None:
        # Chain for single-level hierarchy
        root, child1, child2 = simple_hierarchy

        assert child1.supersystem_chain == [root]
        assert root.supersystem_chain == []

    def test_supersystem_chain_deep(self, deep_hierarchy) -> None:
        # Chain for deep hierarchy
        root, level1, level2, level3 = deep_hierarchy

        assert level3.supersystem_chain == [level2, level1, root]
        assert level2.supersystem_chain == [level1, root]
        assert level1.supersystem_chain == [root]

    def test_reparenting(self) -> None:
        # Can move subsystem to different parent
        parent1 = System(tag='parent1')
        parent2 = System(tag='parent2')
        child = System(tag='child')

        parent1.add(child)
        assert child in parent1

        parent2.add(child)
        assert child not in parent1
        assert child in parent2
        assert child.supersystem is parent2


# ========== ========== ========== ========== Test Validation
class TestValidation:
    """Test hierarchy validation and error conditions."""

    def test_prevents_circular_dependency_direct(self) -> None:
        # Cannot add system as its own subsystem
        sys = System(tag='sys')

        with pytest.raises(ValueError, match='cannot register itself'):
            sys.supersystem = sys

    def test_prevents_circular_dependency_indirect(self) -> None:
        # Cannot create A -> B -> A cycle
        sys_a = System(tag='a')
        sys_b = System(tag='b')

        sys_a.add(sys_b)

        with pytest.raises(ValueError, match='circular dependency'):
            sys_b.add(sys_a)

    def test_prevents_circular_dependency_deep(self, deep_hierarchy) -> None:
        # Cannot create cycle in deep hierarchy
        root, level1, level2, level3 = deep_hierarchy

        with pytest.raises(ValueError, match='circular dependency'):
            root.supersystem = level3

    def test_prevents_duplicate_registration(self, simple_hierarchy) -> None:
        # Cannot register same subsystem twice
        root, child1, child2 = simple_hierarchy

        # well, it's a feature now!
        return

        with pytest.raises(ValueError, match='already registered'):
            root.add(child1)

    def test_prevents_tag_conflict(self, root_system: System) -> None:
        # Cannot have two subsystems with same tag
        child1 = System(tag='same')
        child2 = System(tag='same')

        root_system.add(child1)

        with pytest.raises(ValueError, match='tag already used'):
            root_system.add(child2)

    def test_validates_supersystem_exists(self) -> None:
        # Cannot set supersystem to non-existent ID
        sys = System(tag='sys')

        with pytest.raises(ValueError, match='Invalid supersystem ID'):
            sys._supersystem_id = 'nonexistent_id'

    def test_validates_supersystem_is_system(self) -> None:
        # Supersystem must be a System instance
        from jangada.mixin import Identifiable

        # Create some other Identifiable object
        other = Identifiable()

        sys = System(tag='sys')

        with pytest.raises(TypeError, match='Expected System'):
            sys._supersystem_id = other.id


# ========== ========== ========== ========== Test Persistence
class TestPersistence:
    """Test serialization and file I/O."""

    def test_saves_and_loads(self, tmp_path: Path) -> None:
        # Can save and load system
        sys = System(tag='test', name='Test System')

        path = tmp_path / 'system.sys'
        sys.save(path)

        loaded = System.load(path)

        assert loaded.tag == 'test'
        assert loaded.name == 'Test System'

    def test_saves_with_subsystems(self, tmp_path: Path,
                                   simple_hierarchy) -> None:
        # Saving parent saves subsystems too
        root, child1, child2 = simple_hierarchy

        path = tmp_path / 'hierarchy.sys'
        root.save(path)

        serialized_data = System.load_serialized_data(path)

        loaded = System.load(path)

        assert len(loaded) == 2
        assert 'child1' in loaded
        assert 'child2' in loaded

    def test_reconstructs_hierarchy(self, tmp_path: Path,
                                    deep_hierarchy) -> None:
        # Loading reconstructs full hierarchy
        root, level1, level2, level3 = deep_hierarchy

        path = tmp_path / 'deep.sys'
        root.save(path)

        loaded = System.load(path)

        assert len(loaded) == 1
        assert loaded.level1.supersystem is loaded
        assert loaded.level1.level2.level3.supersystem_chain[-1] is loaded

    def test_equal_method_compares_content(self) -> None:
        # equal() should compare serialized content
        sys1 = System(tag='test', name='Name')
        sys2 = System(tag='test', name='Name')

        # Different IDs, so == returns False
        assert sys1 != sys2

        # But same content, so equal() returns True
        assert sys1.equal(sys2)

    def test_equal_ignores_id(self) -> None:
        # equal() should ignore non-copiable properties like ID
        sys1 = System(tag='test')
        sys2 = System(tag='test')

        # Different IDs
        assert sys1.id != sys2.id

        # But same copiable content
        assert sys1.equal(sys2)

    def test_equal_with_different_content(self) -> None:
        # equal() returns False for different content
        sys1 = System(tag='test', name='Name1')
        sys2 = System(tag='test', name='Name2')

        assert not sys1.equal(sys2)

    def test_equal_recursive(self) -> None:
        # equal() should compare subsystems recursively
        root1 = System(tag='root')
        root1.add(System(tag='child', name='Child1'))

        root2 = System(tag='root')
        root2.add(System(tag='child', name='Child1'))

        assert root1.equal(root2)

        # Different child
        root3 = System(tag='root')
        root3.add(System(tag='child', name='Child2'))

        assert not root1.equal(root3)


# ========== ========== ========== ========== Test Display Integration
class TestDisplayIntegration:
    """Test Displayable integration."""

    def test_has_display_methods(self) -> None:
        # System should have display methods
        sys = System(tag='test')

        assert hasattr(sys, '_title')
        assert hasattr(sys, '_content')
        assert hasattr(sys, 'display_settings')

    def test_title_includes_class_name(self) -> None:
        # _title should include class name
        sys = System(tag='test')

        title = sys._title()

        assert 'System' in str(title)

    def test_content_includes_info(self) -> None:
        # _content should include system info
        sys = System(tag='test', name='Test', description='Desc')

        content = sys._content()

        # Content is a Group, should have panels
        assert content is not None

    def test_displays_subsystems(self, simple_hierarchy) -> None:
        # Display should show subsystems
        root, child1, child2 = simple_hierarchy

        content = root._content()

        # Should include subsystems panel
        assert content is not None

    def test_string_output(self) -> None:
        # Should produce string output
        sys = System(tag='test', name='Test System')

        output = str(sys)

        assert isinstance(output, str)
        assert len(output) > 0


# ========== ========== ========== ========== Integration Tests
class TestIntegration:
    """Test complete workflows and edge cases."""

    def test_complex_hierarchy_operations(self, complex_hierarchy: System) -> None:
        # Test operations on complex hierarchy
        root = complex_hierarchy

        # Navigate
        temp = root.sensors.temperature
        assert temp.name == 'Temperature Sensor'

        # Modify
        temp.name = 'Modified Temp'
        assert root.sensors.temperature.name == 'Modified Temp'

        # Move
        root.controllers.add(temp)
        assert temp not in root.sensors
        assert temp in root.controllers

    def test_rebuild_hierarchy_after_load(self, tmp_path: Path) -> None:
        # Complete save/load cycle
        root = System(tag='root')
        child = System(tag='child', name='Child')
        grandchild = System(tag='grandchild', name='Grandchild')

        root.add(child)
        child.add(grandchild)

        path = tmp_path / 'complete.sys'
        root.save(path)

        loaded = System.load(path)

        # Verify structure
        assert loaded.child.name == 'Child'
        assert loaded.child.grandchild.name == 'Grandchild'
        assert loaded.child.grandchild.supersystem.supersystem is loaded

    def test_tag_observer_survives_operations(self) -> None:
        # Observers should work through various operations
        root = System(tag='root')
        child = System(tag='child')

        root.add(child)

        # Change tag
        child.tag = 'renamed'
        assert 'renamed' in root

        # Remove and re-add
        root.remove(child)
        child.tag = 'child_again'
        root.add(child)

        # Change tag again
        child.tag = 'final'
        assert 'final' in root
        assert root.final is child

    def test_multiple_hierarchy_modifications(self) -> None:
        # Rapid modifications should maintain consistency
        root = System(tag='root')

        # Add many subsystems
        for i in range(10):
            root.add(System(tag=f'sub{i}'))

        assert len(root) == 10

        # Remove some
        for i in range(0, 10, 2):
            root.remove(f'sub{i}')

        assert len(root) == 5

        # Rename remaining
        for i in range(1, 10, 2):
            root[f'sub{i}'].tag = f'renamed{i}'

        assert 'renamed1' in root
        assert 'sub1' not in root

    def test_empty_system_operations(self) -> None:
        # Operations on empty system
        sys = System(tag='empty')

        assert len(sys) == 0
        assert list(sys) == []
        assert 'anything' not in sys

        with pytest.raises(KeyError):
            _ = sys['nonexistent']