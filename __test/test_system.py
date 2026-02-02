#  -*- coding: utf-8 -*-
"""
Author: Rafael R. L. Benevides
"""

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

