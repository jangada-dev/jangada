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

            @serializable_property()
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

    def test_writeonce(self):
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

            @id.getter
            def id(self) -> str:
                return self._id

            @id.setter
            def id(self, value: str) -> None:
                self._id = value

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

            @name.getter
            def name(self) -> str:
                return self._name

            @name.setter
            def name(self, value: str) -> None:
                self._name = value

        # test default
        obj = Nameable()
        assert obj.name == "Unnamed"

        # test setter
        obj.name = "My custom name"
        assert obj.name == "My custom name"

        # test resetting
        obj.name = None
        assert obj.name == "Unnamed"
