#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

import pytest
import time

from pathlib import Path

from jangada import System, load


here = Path(__file__).parent


class TestSystem:

    def test_init(self) -> None:

        print()

        kwargs = {
            'name': "SGDC - 1",
            'tag': 'sgdc',
            'description': 'Satélite Geoestacionário de Defesa e Comunicação',
            'active': True
        }

        s = System(**kwargs)


        assert s.name == kwargs['name']
        assert s.tag == kwargs['tag']
        assert s.description == kwargs['description']
        assert s.active == kwargs['active']

        print(s.subsystems)

    def test_integration(self) -> None:
        print()

        class Battery(System):
            ...

        class Cell(System):
            ...

        batt = Battery(name='Battery', tag='bat')
        batt.add(Cell(name='Cell', tag='cell1'))
        batt.add(Cell(name='Cell', tag='cell2'))

        for subsys in batt.subsystems:
            print(subsys.tag, subsys.name)

        batt.cell1.name = 'New Name'
        assert batt.cell1.name == 'New Name'

        batt.cell1.tag = 'CELL1'
        try:
            batt.cell1
        except AttributeError:
            pass
        else:
            assert False, "Should not be able to access cell1 since it was renamed"

        assert batt.CELL1.name == 'New Name'

        filepath = here / 'battery.sys'

        batt.save(filepath, overwrite=True)

        time.sleep(2)

        batt2 = load(filepath)

        assert batt2.CELL1.name == 'New Name'

        assert batt2 == batt

        filepath.unlink()


class TestSystem:
    # ----------------------------- fixtures/helpers -----------------------------

    @pytest.fixture
    def make_system(self):
        """
        Factory to build a System with a given tag.

        This is defensive against different Taggable APIs:
        - preferred: obj.tag = "x"
        - fallback: obj.set_tag("x") if exists
        """
        def _factory(tag: str | None = None) -> System:
            s = System()
            if tag is None:
                return s

            # Common API: Taggable.tag property
            try:
                s.tag = tag
                return s
            except Exception:
                pass

            # Fallback API
            if hasattr(s, "set_tag"):
                s.set_tag(tag)  # type: ignore[attr-defined]
                return s

            raise RuntimeError(
                "Unable to set System.tag in tests. "
                "Update the test factory to match Taggable API."
            )

        return _factory

    @pytest.fixture
    def parent(self, make_system) -> System:
        return make_system("parent")

    # ----------------------------- basics -----------------------------

    def test_default_subsystems_is_empty_set(self, parent: System) -> None:
        assert isinstance(parent.subsystems, set)
        assert len(parent.subsystems) == 0
        assert len(parent) == 0

    def test_iterates_over_subsystems(self, parent: System, make_system) -> None:
        a = make_system("a")
        b = make_system("b")
        parent.add(a, b)

        assert set(iter(parent)) == {a, b}

    def test_system_is_hashable(self, parent: System) -> None:
        # __hash__ is pinned to Identifiable.__hash__ in your class. :contentReference[oaicite:1]{index=1}
        assert isinstance(hash(parent), int)

    # ----------------------------- __contains__ -----------------------------

    def test_contains_system_instance(self, parent: System, make_system) -> None:
        a = make_system("a")
        assert a not in parent
        parent.add(a)
        assert a in parent

    def test_contains_by_tag_string(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)

        assert "a" in parent
        assert "missing" not in parent

    def test_contains_none_is_false(self, parent: System) -> None:
        # You explicitly return False when obj is None. :contentReference[oaicite:2]{index=2}
        assert (None in parent) is False  # type: ignore[operator]

    # ----------------------------- __getitem__ / __getattr__ -----------------------------

    def test_getitem_returns_subsystem_by_tag(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)

        assert parent["a"] is a

    def test_getitem_raises_keyerror_for_missing_tag(self, parent: System) -> None:
        with pytest.raises(KeyError):
            _ = parent["missing"]

    def test_getitem_raises_keyerror_for_none_tag(self, parent: System) -> None:
        with pytest.raises(KeyError):
            _ = parent[None]  # type: ignore[index]

    def test_getattr_returns_subsystem_by_tag(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)

        assert getattr(parent, "a") is a
        assert parent.a is a

    def test_getattr_raises_attributeerror_for_missing_tag(self, parent: System) -> None:
        with pytest.raises(AttributeError):
            _ = parent.missing

    # ----------------------------- add() validation -----------------------------

    def test_add_rejects_non_system(self, parent: System) -> None:
        with pytest.raises(TypeError):
            parent.add(object())  # type: ignore[arg-type]

    def test_add_rejects_self(self, parent: System) -> None:
        with pytest.raises(ValueError):
            parent.add(parent)

    def test_add_rejects_subsystem_with_none_tag(self, parent: System, make_system) -> None:
        a = make_system(None)
        with pytest.raises(ValueError):
            parent.add(a)

    def test_add_rejects_duplicate_system_instance(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)
        with pytest.raises(ValueError):
            parent.add(a)

    def test_add_rejects_duplicate_tag_different_instances(self, parent: System, make_system) -> None:
        a1 = make_system("dup")
        a2 = make_system("dup")
        parent.add(a1)
        with pytest.raises(ValueError):
            parent.add(a2)

    def test_add_multiple_subsystems_in_one_call(self, parent: System, make_system) -> None:
        a = make_system("a")
        b = make_system("b")
        parent.add(a, b)

        assert len(parent) == 2
        assert "a" in parent
        assert "b" in parent
        assert parent["a"] is a
        assert parent["b"] is b

    # ----------------------------- remove() behavior -----------------------------

    def test_remove_by_system_instance(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)
        assert a in parent

        parent.remove(a)

        assert a not in parent
        assert "a" not in parent
        assert len(parent) == 0

    def test_remove_by_tag_string(self, parent: System, make_system) -> None:
        a = make_system("a")
        parent.add(a)

        parent.remove("a")

        assert a not in parent
        assert len(parent) == 0

    def test_remove_raises_valueerror_when_missing_by_instance(self, parent: System, make_system) -> None:
        a = make_system("a")
        with pytest.raises(ValueError):
            parent.remove(a)

    def test_remove_raises_valueerror_when_missing_by_tag(self, parent: System) -> None:
        with pytest.raises(ValueError):
            parent.remove("missing")

    # ----------------------------- subsystems property: default/parser -----------------------------

    def test_subsystems_parser_accepts_none_as_empty_set(self, parent: System) -> None:
        parent.subsystems = None
        assert parent.subsystems == set()

    def test_subsystems_parser_rejects_non_system_elements(self, parent: System) -> None:
        with pytest.raises(TypeError):
            parent.subsystems = {object()}  # type: ignore[assignment]

    def test_subsystems_parser_coerces_iterable_to_set(self, parent: System, make_system) -> None:
        a = make_system("a")
        b = make_system("b")

        parent.subsystems = [a, b]
        assert isinstance(parent.subsystems, set)
        assert parent.subsystems == {a, b}