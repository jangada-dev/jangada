#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

import pytest

import numpy, pandas
from pathlib import Path


from jangada.serialization import Serializable, serializable_property, SerializableProperty

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

    def test_integration_3(self) -> None:
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

    def test_programatically_defined_properties(self) -> None:
        print()

        class Orbit(Serializable):
            pass

        # After class definition, add properties programmatically
        for prop_name in ['semi_major_axis', 'eccentricity', 'inclination']:
            prop = SerializableProperty(default=0.0)
            prop.__set_name__(Orbit, prop_name)
            setattr(Orbit, prop_name, prop)

        obj = Orbit()

        print(obj.inclination)


class TesteSerializable:

    def test_Serializable(self) -> None:

        class Nameable:
            name = SerializableProperty(default="Unnamed")


        class Person(Serializable, Nameable):
            age = SerializableProperty(default=0, copiable=False)

        class Car(Serializable):
            pass


        assert 'name' in Person.serializable_properties
        assert 'age' in Person.serializable_properties

        print(Person.copiable_properties.keys())

        print(Car.serializable_properties)


    @pytest.fixture
    def example_primitive_type(self) -> list:
        return [
            # str
            'some text',

            #  numbers.Number
            True,
            False,
            1,
            1.0,
            1.0j,

            # numpy.ndarray
            numpy.array(1.0),
            numpy.array([1.0]),
            numpy.array([[1.0]]),

            Path(__file__).parent
        ]

    @pytest.fixture
    def custom_serialisable(self) -> tuple[Serializable, ...]:

        class Point(Serializable):
            x = SerializableProperty()
            y = SerializableProperty()

            def translate(self, dx: float, dy: float) -> 'Point':
                return Point(x=self.x + dx, y=self.y + dy)


        class Triangle(Serializable):

            p1 = SerializableProperty()
            p2 = SerializableProperty()
            p3 = SerializableProperty()

            def translate(self, dx: float, dy: float) -> 'Triangle':
                p1 = self.p1.translate(dx, dy)
                p2 = self.p2.translate(dx, dy)
                p3 = self.p3.translate(dx, dy)

                return Triangle(p1=p1, p2=p2, p3=p3)

        class Mesh(Serializable):

            triangles = SerializableProperty()
            name = SerializableProperty(default="Unnamed")

        return Point, Triangle, Mesh

    # ========== ========== ========== ========== primitive types
    def test_serialise_primitive_type(self, example_primitive_type):

        for obj in example_primitive_type:
            assert numpy.all(Serializable.serialize(obj) == obj), obj

    def test_deserialize_primitive_type(self, example_primitive_type):

        for obj in example_primitive_type:
            assert numpy.all(Serializable.deserialize(obj) == obj), obj

    # ========== ========== ========== ========== containers
    def test_serialise_containers(self, example_primitive_type):

        data = [
            {
                'given': example_primitive_type,
                'expected': example_primitive_type
            },
            {
                'given': tuple(example_primitive_type),
                'expected': example_primitive_type
            },
            {
                'given': {f'{k}': v for k, v in enumerate(example_primitive_type)},
                'expected': {f'{k}': v for k, v in enumerate(example_primitive_type)}
            }
        ]

        for item in data:
            given = item['given']
            expected = item['expected']
            assert numpy.all(Serializable.serialize(given) == expected), given

    def test_deserialize_containers(self, example_primitive_type):

        data = [
            {
                'given': example_primitive_type,
                'expected': example_primitive_type
            },
            {
                'given': {f'{k}': v for k, v in enumerate(example_primitive_type)},
                'expected': {f'{k}': v for k, v in enumerate(example_primitive_type)}
            }
        ]

        for item in data:
            given = item['given']
            expected = item['expected']
            assert numpy.all(Serializable.deserialize(given) == expected), given

    # ========== ========== ========== ========== subclassing
    def test_subclassing(self, custom_serialisable):

        Point, Triangle, Mesh = custom_serialisable

        assert Point in Serializable
        assert Triangle in Serializable
        assert Mesh in Serializable

    def test_subclass_instances(self, custom_serialisable):

        Point, Triangle, Mesh = custom_serialisable

        # ---------- ---------- ---------- ---------- empty instance
        point = Point()
        assert point.x is None
        assert point.y is None

        # ---------- ---------- ---------- ---------- lack data
        point = Point(x=1.0)
        assert point.x == 1.0
        assert point.y is None

        # ---------- ---------- ---------- ---------- full data
        point = Point(x=1.0, y=-1.0)
        assert point.x == 1.0
        assert point.y == -1.0

        # ---------- ---------- ---------- ---------- comparison
        p1 = Point(x=1.0, y=-1.0)
        p2 = Point(x=1, y=-1)
        p3 = Point(x=0, y=0)

        assert p1 == p2
        assert p1 != p3

        # ---------- ---------- ---------- ---------- copying
        point_original = Point(x=1.0, y=-1.0)
        point_copied1 = Point(point_original)
        point_copied2 = point_original.copy()

        assert point_original == point_copied1
        assert not (point_original is point_copied1)

        assert point_original == point_copied2
        assert not (point_original is point_copied2)

    def test_serialise_subclass_instances(self, custom_serialisable):

        print()
        import beeprint

        Point, Triangle, Mesh = custom_serialisable

        km = 1000.0

        p1 = Point(x=0.0 * km, y=4.0 * km)
        p2 = Point(x=1.0 * km, y=5.0 * km)
        p3 = Point(x=2.0 * km, y=3.0 * km)

        triangle = Triangle(p1=p1, p2=p2, p3=p3)

        triangles = [
            triangle.translate(+5 * km, +5 * km),
            triangle.translate(-5 * km, -5 * km),
            triangle.translate(-5 * km, +5 * km),
            triangle.translate(+5 * km, -5 * km)
        ]

        mesh = Mesh(triangles=triangles, name='mymesh')

        assert triangles == mesh.triangles
        assert not (triangles is mesh.triangles)  # since a copy has been made
        assert Mesh(mesh) == mesh
        assert mesh.copy() == mesh
        assert Mesh(**Serializable.serialize(mesh)) == mesh

        # for curiosity
        print()
        beeprint.pp(Serializable.serialize(mesh), indent=4)