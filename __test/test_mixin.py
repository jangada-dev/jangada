#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

import uuid

from jangada.mixin import Identifiable


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

