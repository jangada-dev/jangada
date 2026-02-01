#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

import uuid

from jangada.mixin import Identifiable, Taggable, TagNamespace


class TestIdentifiable:

    def test_id(self) -> None:
        print()

        # check if it can be getter
        obj1 = Identifiable()
        assert obj1.id

        assert Identifiable.get_instance(obj1.id) is obj1

        # check if it can be setter
        obj2 = Identifiable()
        value = uuid.uuid4()
        obj2.id = value
        assert obj2.id == str(value.hex)

        # form now on, it should be read-only
        try:
            obj2.id = uuid.uuid4()
        except AttributeError:
            pass
        else:
            assert False, "Should not be able to set id"

        # retrieve object
        assert Identifiable.get_instance(str(value.hex)) is obj2


class TestTaggable:

    def test_tag(self) -> None:

        obj = Taggable()
        assert obj.tag is None

        obj.tag = "test"
        assert obj.tag == "test"

        not_allowed_examples = [10, '12_xu', 'def', 'None']

        for tag in not_allowed_examples:

            try:
                obj.tag = tag
            except ValueError:
                pass
            else:
                assert False, "Should not be able to set tag"

    def test_tagnamespace(self) -> None:
        print()

        namespace = TagNamespace()

        obj1 = Taggable()
        obj1.tag = "obj1"
        namespace.register(obj1)

        assert namespace.obj1 is obj1

        obj2 = Taggable()
        namespace.register(obj2)
        obj2.tag = "obj2"

        assert namespace.obj2 is obj2

        obj3 = Taggable()
        namespace.register(obj3)

        assert obj3 in namespace

        obj2.tag = "new_obj2_tag"
        assert namespace.new_obj2_tag is obj2



