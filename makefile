test_serialization:
	PYTHONPATH=. pytest -v __test/test_SerializableProperty.py
	PYTHONPATH=. pytest -v __test/test_Serializable.py
	PYTHONPATH=. pytest -v __test/test_Persistable.py


test_mixin:
	PYTHONPATH=. pytest -v __test/test_mixin.py

test_display:
	PYTHONPATH=. pytest -v __test/test_display.py

test_system:
	PYTHONPATH=. pytest -v __test/test_System.py

test_selected: test_serialization test_mixin test_display test_system

test_current:
	#PYTHONPATH=. pytest -v __test/test_System.py::TestSubsystemRegistration::test_add_single_subsystem
	PYTHONPATH=. pytest -s -vv __test/test_System.py::TestPersistence::test_equal_recursive

test_all:
	PYTHONPATH=. pytest -s -vv __test/test_*.py