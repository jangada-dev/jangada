#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

import pytest

from jangada.serialization.properties import serializable_property, SerializableProperty

from typing import Any


class TestSerializableProperty:

    def test_readonly(self) -> None:
        print()

        class ReadOnlyExample:

            @serializable_property(readonly=True, default="default")
            def attr(self) -> str:
                return "attr"

        obj = ReadOnlyExample()

        print(obj.attr)

        try:
            obj.attr = 10

        except AttributeError:
            pass

        else:
            assert False

    def test_writeonce(self) -> None:
        print()

        class WriteOnceExample:

            @serializable_property(writeonce=True)
            def attr(self) -> str | None:
                try:
                    return self._attr
                except AttributeError:
                    ...

            @attr.setter
            def attr(self, value: Any) -> None:
                self._attr = value

        obj = WriteOnceExample()

        obj.attr = 10

        try:
            obj.attr = 20

        except AttributeError:
            pass

        else:
            raise AttributeError()

    def test_integration_1(self) -> None:
        print()

        import uuid

        class Identifiable:

            id = SerializableProperty(copiable=False, writeonce=True)

            @id.default
            def id(self) -> str:
                return str(uuid.uuid4().hex)

            @id.parser
            def id(self, value: Any) -> str:
                value = uuid.UUID(str(value), version=4)
                return str(value.hex)

        # check if default works
        obj = Identifiable()
        uuid.UUID(obj.id, version=4)

        # check if setter works
        obj = Identifiable()
        value = uuid.uuid4()
        obj.id = value

        # check if parser works
        assert obj.id == str(value.hex)

        # check if id property is read-only
        try:
            obj.id = "My custom ID 2"
        except AttributeError:
            pass
        else:
            raise AttributeError()

    def test_integration_2(self) -> None:
        print()

        class Nameable:
            name = SerializableProperty(default="Unnamed")

        # test default
        obj = Nameable()
        assert obj.name == "Unnamed"

        # test setter
        obj.name = "My custom name"
        assert obj.name == "My custom name"

        # test resetting
        obj.name = None
        assert obj.name == "Unnamed"

    def test_programatically_defined_properties(self) -> None:
        print()

        class Point:

            x = SerializableProperty()
            y = SerializableProperty(default=0)
            z = SerializableProperty(default=0)

            @x.parser
            def x(self, value: Any) -> float:
                return float(value) if value is not None else None

            @x.observer
            def x(self, old_value: float, new_value: float) -> None:
                print(f"x changed from {old_value} to {new_value}")

        p = Point()

        assert p.x is None
        assert p.y == 0.0
        assert p.z == 0.0

        p.x = '10.0'
        p.y = 20.0
        p.z = 30.0

        assert p.x == 10.0
        assert p.y == 20.0
        assert p.z == 30.0

        p.x = None
        assert p.x is None

        p.y = None
        assert p.y == 0.0

        p.z = None
        assert p.z == 0.0