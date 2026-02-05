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
            # 'active': True
        }

        s = System(**kwargs)


        assert s.name == kwargs['name']
        assert s.tag == kwargs['tag']
        assert s.description == kwargs['description']
        # assert s.active == kwargs['active']

        # print(s.subsystems)

        print(s)

    def test_integration(self) -> None:
        print()

        class Battery(System):
            ...

        class Cell(System):
            ...

        batt = Battery(name='Battery', tag='bat')

        cell1 = Cell(name='Cell', tag='cell1')
        cell2 = Cell(name='Cell', tag='cell2')

        assert cell1 != cell2
        assert cell1 is not cell2  # sanity check

        batt.add(cell1, cell2)

        batt.cell1.name = 'New Name'
        assert batt.cell1.name == 'New Name'

        print(batt)

        return

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

        assert batt2.equal(batt)

        assert batt2.id == batt.id

        batt3 = Battery(batt2)

        assert batt3.CELL1.name == batt2.CELL1.name == 'New Name'
        assert batt3.cell2.tag == batt2.cell2.tag == 'cell2'

        assert batt3.id != batt.id
        assert batt3.equal(batt2)

        filepath.unlink()