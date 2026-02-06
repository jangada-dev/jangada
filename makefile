test_serialization:
	PYTHONPATH=. pytest -v test/test_SerializableProperty.py
	PYTHONPATH=. pytest -v test/test_Serializable.py
	PYTHONPATH=. pytest -v test/test_Persistable.py


test_mixin:
	PYTHONPATH=. pytest -v test/test_mixin.py

test_display:
	PYTHONPATH=. pytest -v test/test_display.py

test_system:
	PYTHONPATH=. pytest -v test/test_System.py

test_selected: test_serialization test_mixin test_display test_system

test_current:
	#PYTHONPATH=. pytest -v test/test_System.py::TestSubsystemRegistration::test_add_single_subsystem
	PYTHONPATH=. pytest -s -vv test/test_System.py::TestPersistence::test_equal_recursive

test_all:
	PYTHONPATH=. pytest -s -vv test/test_*.py