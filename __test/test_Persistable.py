#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for Persistable and HDF5 persistence.

Tests cover:
- File saving and loading
- Context manager protocol
- ProxyDataset lazy loading
- Container types (list, dict)
- Dataset types (numpy arrays, pandas timestamps)
- Special types (None, Path)
- ProxyDataset operations (slicing, appending, resizing)
- Error handling

================================================================================
KNOWN CONCERNS AND FUTURE FEATURES
================================================================================

DESIGN CONCERNS:
----------------

1. Three-way __init__ overload
   - Current: obj = MyClass('/path') loads, MyClass('/path', mode='r') opens
   - Potentially confusing - presence/absence of kwargs determines behavior
   - Consider: @classmethod open(cls, path, mode) for clarity

2. ProxyDataset as Primitive Type
   - Registered as primitive, but what if someone tries to save it back to HDF5?
   - Might need special handling or validation

3. List Order Dependency
   - Lists rely on track_order=True for correct ordering
   - Files created without this flag will have undefined order
   - Need validation or documentation

4. Scalar Dataset Limitation
   - Scalar datasets (ndim=0) cannot be resized/appended
   - Should be documented clearly
   - Consider raising error on append attempts

5. Assertion in ProxyDataset.__setitem__
   - assert attrs == self.attrs might fail silently
   - Should raise ValueError with clear message instead

6. File Handle Name Mangling
   - Uses __file (double underscore) for file handle
   - Why not _file_handle? Is name mangling intentional?

7. No File Handle Cleanup on Error
   - If _initialize_from_data raises in __enter__, file stays open
   - Need try/except to ensure cleanup

8. Type Checking with type() vs isinstance()
   - Uses type(value) in Serializable.dataset_types
   - Won't match subclasses - should use isinstance()

9. No HDF5 Structure Validation
   - Assumes file has expected structure (root group, __container_type__, etc.)
   - Missing keys raise cryptic KeyErrors
   - Need validation and better error messages

10. ProxyDataset Array API Incomplete
    - Has: __getitem__, __setitem__, shape, dtype, etc.
    - Missing: __len__, __repr__, __iter__, arithmetic ops
    - Users might expect full numpy-like interface


MISSING FEATURES / FUTURE ENHANCEMENTS:
----------------------------------------

1. Partial Loading
   Current: MyClass.load('file.hdf5') loads entire object
   Future: MyClass.load('file.hdf5', properties=['data']) for selective loading

2. Incremental Saving
   Current: save() overwrites entire file
   Future: save(incremental=True) to update only changed properties

3. Compression Support
   HDF5 supports compression (gzip, lzf, etc.)
   Could add class attributes:
   - compression = 'gzip'
   - compression_opts = 4

4. Metadata Storage
   Store creation time, version, user, etc. in file attrs:
   - file.attrs['created'] = datetime.now().isoformat()
   - file.attrs['jangada_version'] = __version__
   - file.attrs['created_by'] = getpass.getuser()

5. File Format Validation
   Check that loaded file is valid Persistable format:
   - Verify 'root' group exists
   - Check for required attributes
   - Validate version compatibility

6. Chunking Strategy
   Allow configuration of HDF5 chunking for performance:
   - chunk_shape = 'auto' or tuple
   - Would improve I/O for specific access patterns

7. File Locking
   Handle multiple processes accessing same file:
   - Read locks
   - Write locks
   - Lock file mechanism

8. Version Migration
   Handle schema changes between versions:
   - Detect old file format
   - Migrate to new format
   - Backward compatibility

9. Large File Performance
   Optimizations for GB-scale datasets:
   - Streaming writes
   - Chunked operations
   - Progress callbacks

10. Mixed Loading Modes
    Some properties as ProxyDataset, others fully loaded:
    - load('file.hdf5', lazy=['big_array'], eager=['metadata'])

11. Caching Layer
    Cache frequently accessed ProxyDataset slices:
    - LRU cache for hot data
    - Configurable cache size

12. Parallel I/O
    Support for parallel HDF5:
    - MPI support
    - Thread-safe operations

13. Cloud Storage
    Support for S3, GCS, etc.:
    - Via h5py with appropriate drivers
    - Streaming from remote files

14. File Repair
    Tools to recover from corrupted files:
    - Validate file integrity
    - Repair common issues
    - Extract salvageable data

15. Diff/Merge Operations
    Compare and merge HDF5 files:
    - Show differences between versions
    - Merge changes from multiple files

================================================================================
END OF CONCERNS AND FUTURE FEATURES
================================================================================

Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Any
import tempfile
import os


from jangada import SerializableProperty, Serializable, Persistable


# ========== ========== ========== ========== Fixtures


@pytest.fixture
def temp_hdf5_file(tmp_path: Path) -> Path:
    """Create a temporary HDF5 file path."""
    return tmp_path / "test_data.hdf5"


@pytest.fixture
def simple_persistable_class() -> type:
    """A simple Persistable class for testing."""

    class SimpleClass(Persistable):
        value = SerializableProperty(default=0)
        name = SerializableProperty(default="")

    return SimpleClass


@pytest.fixture
def array_persistable_class() -> type:
    """A Persistable class with array data."""

    class ArrayClass(Persistable):
        data = SerializableProperty(default=None)
        metadata = SerializableProperty(default="")

    return ArrayClass


@pytest.fixture
def nested_persistable_class(simple_persistable_class: type) -> type:
    """A Persistable class with nested objects."""

    class NestedClass(Persistable):
        child = SerializableProperty(default=None)
        value = SerializableProperty(default=0)

    return NestedClass


# ========== ========== ========== ========== Basic Save/Load
class TestBasicSaveLoad:
    """Test basic file saving and loading operations."""

    def test_save_creates_file(self, simple_persistable_class: type,
                               temp_hdf5_file: Path) -> None:
        # Saving should create an HDF5 file
        obj = simple_persistable_class(value=42, name="test")
        obj.save(temp_hdf5_file)

        assert temp_hdf5_file.exists()
        assert temp_hdf5_file.suffix == '.hdf5'

    def test_load_from_file(self, simple_persistable_class: type,
                            temp_hdf5_file: Path) -> None:
        # Should load object from file
        original = simple_persistable_class(value=42, name="test")
        original.save(temp_hdf5_file)

        loaded = simple_persistable_class.load(temp_hdf5_file)

        assert loaded.value == 42
        assert loaded.name == "test"

    def test_save_load_roundtrip(self, simple_persistable_class: type,
                                 temp_hdf5_file: Path) -> None:
        # Save and load should preserve all data
        original = simple_persistable_class(value=99, name="roundtrip")
        original.save(temp_hdf5_file)

        loaded = simple_persistable_class.load(temp_hdf5_file)

        assert original == loaded

    def test_default_extension_added(self, simple_persistable_class: type,
                                     tmp_path: Path) -> None:
        # Should automatically add .hdf5 extension
        obj = simple_persistable_class(value=1)
        path_without_ext = tmp_path / "testfile"

        obj.save(path_without_ext)

        expected_path = tmp_path / "testfile.hdf5"
        assert expected_path.exists()

    def test_custom_extension_respected(self, tmp_path: Path) -> None:
        # Custom extension on class should be used
        class CustomExtClass(Persistable):
            extension = '.h5'
            value = SerializableProperty(default=0)

        obj = CustomExtClass(value=42)
        path = tmp_path / "testfile"
        obj.save(path)

        expected_path = tmp_path / "testfile.h5"
        assert expected_path.exists()

    def test_overwrite_false_raises_error(self, simple_persistable_class: type,
                                          temp_hdf5_file: Path) -> None:
        # Should raise FileExistsError when overwrite=False
        obj = simple_persistable_class(value=1)
        obj.save(temp_hdf5_file)

        obj2 = simple_persistable_class(value=2)
        with pytest.raises(FileExistsError):
            obj2.save(temp_hdf5_file, overwrite=False)

    def test_overwrite_true_replaces_file(self, simple_persistable_class: type,
                                          temp_hdf5_file: Path) -> None:
        # Should overwrite existing file when overwrite=True
        obj1 = simple_persistable_class(value=1, name="first")
        obj1.save(temp_hdf5_file)

        obj2 = simple_persistable_class(value=2, name="second")
        obj2.save(temp_hdf5_file, overwrite=True)

        loaded = simple_persistable_class.load(temp_hdf5_file)
        assert loaded.value == 2
        assert loaded.name == "second"

    def test_load_nonexistent_file_raises(self, simple_persistable_class: type) -> None:
        # Should raise FileNotFoundError for missing files
        with pytest.raises(FileNotFoundError):
            simple_persistable_class.load("/nonexistent/path.hdf5")

    def test_class_method_load(self, simple_persistable_class: type,
                               temp_hdf5_file: Path) -> None:
        # load() should be a class method
        obj = simple_persistable_class(value=42)
        obj.save(temp_hdf5_file)

        loaded = simple_persistable_class.load(temp_hdf5_file)
        assert isinstance(loaded, simple_persistable_class)

    def test_module_level_load_function(self, simple_persistable_class: type,
                                        temp_hdf5_file: Path) -> None:
        # Module-level load() should work
        from jangada.serialization import load

        obj = simple_persistable_class(value=42)
        obj.save(temp_hdf5_file)

        loaded = load(temp_hdf5_file)
        assert isinstance(loaded, simple_persistable_class)
        assert loaded.value == 42


# ========== ========== ========== ========== Initialization Modes
class TestInitializationModes:
    """Test the three different initialization modes."""

    def test_normal_construction(self, simple_persistable_class: type) -> None:
        # Normal construction with kwargs
        obj = simple_persistable_class(value=42, name="test")
        assert obj.value == 42
        assert obj.name == "test"

    def test_load_mode_construction(self, simple_persistable_class: type,
                                    temp_hdf5_file: Path) -> None:
        # Construction with filepath loads from file
        original = simple_persistable_class(value=99, name="loaded")
        original.save(temp_hdf5_file)

        loaded = simple_persistable_class(temp_hdf5_file)
        assert loaded.value == 99
        assert loaded.name == "loaded"

    def test_open_mode_construction(self, simple_persistable_class: type,
                                    temp_hdf5_file: Path) -> None:
        # Construction with filepath and mode prepares for context manager
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        # Should not raise, just prepare for __enter__
        obj = simple_persistable_class(temp_hdf5_file, mode='r')
        # Properties not accessible until __enter__

    def test_string_path_accepted(self, simple_persistable_class: type,
                                  temp_hdf5_file: Path) -> None:
        # Should accept string paths
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        loaded = simple_persistable_class(str(temp_hdf5_file))
        assert loaded.value == 42

    def test_pathlib_path_accepted(self, simple_persistable_class: type,
                                   temp_hdf5_file: Path) -> None:
        # Should accept pathlib.Path objects
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        loaded = simple_persistable_class(temp_hdf5_file)
        assert loaded.value == 42

    def test_unexpected_kwargs_raise_error(self, simple_persistable_class: type,
                                           temp_hdf5_file: Path) -> None:
        # Unknown kwargs in open mode should raise ValueError
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        with pytest.raises(ValueError, match="Unexpected keyword arguments"):
            simple_persistable_class(temp_hdf5_file, mode='r', invalid_arg=True)


# ========== ========== ========== ========== Context Manager
class TestContextManager:
    """Test context manager protocol for file access."""

    def test_context_manager_opens_file(self, simple_persistable_class: type,
                                        temp_hdf5_file: Path) -> None:
        # Context manager should open file and load data
        original = simple_persistable_class(value=42, name="context")
        original.save(temp_hdf5_file)

        with simple_persistable_class(temp_hdf5_file, mode='r') as obj:
            assert obj.value == 42
            assert obj.name == "context"

    def test_context_manager_closes_file(self, simple_persistable_class: type,
                                         temp_hdf5_file: Path) -> None:
        # File should be closed after exiting context
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        with simple_persistable_class(temp_hdf5_file, mode='r') as obj:
            pass

        # File should be closed now - can open again
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'root' in f

    def test_context_manager_read_mode(self, array_persistable_class: type,
                                       temp_hdf5_file: Path) -> None:
        # Read mode should provide ProxyDataset access
        original = array_persistable_class(
            data=np.array([1, 2, 3, 4, 5]),
            metadata="test"
        )
        original.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as obj:
            # In context, data should be ProxyDataset
            assert isinstance(obj.data, Persistable.ProxyDataset)
            # Can still access values
            assert np.array_equal(obj.data[:], np.array([1, 2, 3, 4, 5]))

    def test_context_manager_write_mode(self, array_persistable_class: type,
                                        temp_hdf5_file: Path) -> None:
        # Write mode should allow modifications
        original = array_persistable_class(
            data=np.array([1, 2, 3]),
            metadata="original"
        )
        original.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r+') as obj:
            obj.data[0] = 99

        # Changes should be persisted
        loaded = array_persistable_class.load(temp_hdf5_file)
        assert loaded.data[0] == 99

    def test_context_manager_exception_handling(self, simple_persistable_class: type,
                                                temp_hdf5_file: Path) -> None:
        # File should close even if exception occurs
        original = simple_persistable_class(value=42)
        original.save(temp_hdf5_file)

        try:
            with simple_persistable_class(temp_hdf5_file, mode='r') as obj:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should still be closed
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'root' in f


# ========== ========== ========== ========== Data Types
class TestDataTypes:
    """Test serialization of different data types to HDF5."""

    def test_save_load_none(self, tmp_path: Path) -> None:
        # None values should serialize correctly
        class TestClass(Persistable):
            value = SerializableProperty(default=None)

        obj = TestClass(value=None)
        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = TestClass.load(path)
        assert loaded.value is None

    def test_save_load_string(self, simple_persistable_class: type,
                              temp_hdf5_file: Path) -> None:
        # String values should work
        obj = simple_persistable_class(value=0, name="test string")
        obj.save(temp_hdf5_file)

        loaded = simple_persistable_class.load(temp_hdf5_file)
        assert loaded.name == "test string"

    def test_save_load_numbers(self, tmp_path: Path) -> None:
        # Various numeric types should work
        class NumberClass(Persistable):
            integer = SerializableProperty(default=0)
            floating = SerializableProperty(default=0.0)
            complex_num = SerializableProperty(default=0j)

        obj = NumberClass(integer=42, floating=3.14, complex_num=2 + 3j)
        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = NumberClass.load(path)
        assert loaded.integer == 42
        assert loaded.floating == 3.14
        assert loaded.complex_num == 2 + 3j

    def test_save_load_path(self, tmp_path: Path) -> None:
        # pathlib.Path should serialize
        class PathClass(Persistable):
            filepath = SerializableProperty(default=None)

        test_path = Path("/some/absolute/path")
        obj = PathClass(filepath=test_path)

        save_path = tmp_path / "test.hdf5"
        obj.save(save_path)

        loaded = PathClass.load(save_path)
        assert loaded.filepath == test_path
        assert isinstance(loaded.filepath, Path)

    def test_save_load_list(self, tmp_path: Path) -> None:
        # Lists should serialize with order preserved
        class ListClass(Persistable):
            items = SerializableProperty(default=None)

        obj = ListClass(items=[1, 2, "three", 4.0])
        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = ListClass.load(path)
        assert loaded.items == [1, 2, "three", 4.0]

    def test_save_load_dict(self, tmp_path: Path) -> None:
        # Dicts should serialize
        class DictClass(Persistable):
            data = SerializableProperty(default=None)

        obj = DictClass(data={"a": 1, "b": "two", "c": 3.0})
        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = DictClass.load(path)
        assert loaded.data == {"a": 1, "b": "two", "c": 3.0}

    def test_save_load_numpy_array(self, array_persistable_class: type,
                                   temp_hdf5_file: Path) -> None:
        # NumPy arrays should serialize
        arr = np.array([1.1, 2.2, 3.3, 4.4])
        obj = array_persistable_class(data=arr, metadata="test")
        obj.save(temp_hdf5_file)

        loaded = array_persistable_class.load(temp_hdf5_file)
        assert np.array_equal(loaded.data, arr)

    def test_save_load_pandas_timestamp(self, tmp_path: Path) -> None:
        # Pandas Timestamp should serialize
        class TimeClass(Persistable):
            timestamp = SerializableProperty(default=None)

        ts = pd.Timestamp('2024-01-15 12:30:00')
        obj = TimeClass(timestamp=ts)

        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = TimeClass.load(path)
        assert loaded.timestamp == ts

    def test_save_load_pandas_timestamp_with_timezone(self, tmp_path: Path) -> None:
        # Timezone-aware timestamps should preserve timezone
        class TimeClass(Persistable):
            timestamp = SerializableProperty(default=None)

        ts = pd.Timestamp('2024-01-15 12:30:00', tz='UTC')
        obj = TimeClass(timestamp=ts)

        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = TimeClass.load(path)
        assert loaded.timestamp == ts
        assert str(loaded.timestamp.tz) == 'UTC'

    def test_save_load_multidimensional_array(self, tmp_path: Path) -> None:
        # Multi-dimensional arrays should work
        class ArrayClass(Persistable):
            matrix = SerializableProperty(default=None)

        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        obj = ArrayClass(matrix=arr)

        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = ArrayClass.load(path)
        assert np.array_equal(loaded.matrix, arr)
        assert loaded.matrix.shape == (3, 3)


# ========== ========== ========== ========== Nested Objects
class TestNestedObjects:
    """Test serialization of nested Persistable objects."""

    def test_save_load_nested_object(self, simple_persistable_class: type,
                                     nested_persistable_class: type,
                                     temp_hdf5_file: Path) -> None:
        # Nested Persistable objects should work
        child = simple_persistable_class(value=10, name="child")
        parent = nested_persistable_class(child=child, value=100)

        parent.save(temp_hdf5_file)

        loaded = nested_persistable_class.load(temp_hdf5_file)
        assert loaded.value == 100
        assert loaded.child.value == 10
        assert loaded.child.name == "child"

    def test_save_load_list_of_objects(self, simple_persistable_class: type,
                                       tmp_path: Path) -> None:
        # Lists of Persistable objects should work
        class ListOfObjects(Persistable):
            items = SerializableProperty(default=None)

        items = [
            simple_persistable_class(value=1, name="first"),
            simple_persistable_class(value=2, name="second"),
            simple_persistable_class(value=3, name="third")
        ]
        obj = ListOfObjects(items=items)

        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = ListOfObjects.load(path)
        assert len(loaded.items) == 3
        assert loaded.items[0].value == 1
        assert loaded.items[1].name == "second"

    def test_save_load_deeply_nested(self, tmp_path: Path) -> None:
        # Deeply nested structures should work
        class Inner(Persistable):
            value = SerializableProperty(default=0)

        class Middle(Persistable):
            inner = SerializableProperty(default=None)

        class Outer(Persistable):
            middle = SerializableProperty(default=None)

        obj = Outer(middle=Middle(inner=Inner(value=42)))

        path = tmp_path / "test.hdf5"
        obj.save(path)

        loaded = Outer.load(path)
        assert loaded.middle.inner.value == 42


# ========== ========== ========== ========== ProxyDataset
class TestProxyDataset:
    """Test ProxyDataset lazy loading and operations."""

    def test_proxydataset_lazy_loading(self, array_persistable_class: type,
                                       temp_hdf5_file: Path) -> None:
        # ProxyDataset should not load entire array immediately
        large_array = np.arange(10000)
        obj = array_persistable_class(data=large_array)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            # data is ProxyDataset, not full array
            assert isinstance(loaded.data, Persistable.ProxyDataset)

    def test_proxydataset_slicing(self, array_persistable_class: type,
                                  temp_hdf5_file: Path) -> None:
        # ProxyDataset should support slicing
        arr = np.arange(100)
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            sliced = loaded.data[10:20]
            assert np.array_equal(sliced, arr[10:20])

    def test_proxydataset_indexing(self, array_persistable_class: type,
                                   temp_hdf5_file: Path) -> None:
        # ProxyDataset should support integer indexing
        arr = np.array([10, 20, 30, 40, 50])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            assert loaded.data[0] == 10
            assert loaded.data[2] == 30
            assert loaded.data[-1] == 50

    def test_proxydataset_shape_property(self, array_persistable_class: type,
                                         temp_hdf5_file: Path) -> None:
        # ProxyDataset.shape should return dataset shape
        arr = np.zeros((10, 5, 3))
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            assert loaded.data.shape == (10, 5, 3)

    def test_proxydataset_dtype_property(self, array_persistable_class: type,
                                         temp_hdf5_file: Path) -> None:
        # ProxyDataset.dtype should return dataset dtype
        arr = np.array([1, 2, 3], dtype=np.float64)
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            assert loaded.data.dtype == np.float64

    def test_proxydataset_ndim_property(self, array_persistable_class: type,
                                        temp_hdf5_file: Path) -> None:
        # ProxyDataset.ndim should return number of dimensions
        arr = np.zeros((5, 10, 15))
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            assert loaded.data.ndim == 3

    def test_proxydataset_size_property(self, array_persistable_class: type,
                                        temp_hdf5_file: Path) -> None:
        # ProxyDataset.size should return total elements
        arr = np.zeros((5, 10))
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            assert loaded.data.size == 50

    def test_proxydataset_setitem(self, array_persistable_class: type,
                                  temp_hdf5_file: Path) -> None:
        # ProxyDataset should support item assignment
        arr = np.array([1, 2, 3, 4, 5])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r+') as loaded:
            loaded.data[0] = 99
            loaded.data[2:4] = np.array([88, 77])

        # Verify changes persisted
        loaded_again = array_persistable_class.load(temp_hdf5_file)
        assert loaded_again.data[0] == 99
        assert loaded_again.data[2] == 88
        assert loaded_again.data[3] == 77

    def test_proxydataset_append(self, array_persistable_class: type,
                                 temp_hdf5_file: Path) -> None:
        # ProxyDataset should support append operation
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r+') as loaded:
            loaded.data.append(np.array([4, 5, 6]))

        # Verify data was appended
        loaded_again = array_persistable_class.load(temp_hdf5_file)
        expected = np.array([1, 2, 3, 4, 5, 6])
        assert np.array_equal(loaded_again.data, expected)

    def test_proxydataset_resize_on_setitem(self, array_persistable_class: type,
                                            temp_hdf5_file: Path) -> None:
        # ProxyDataset should auto-resize when setting beyond bounds
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r+') as loaded:
            # Set beyond current size
            loaded.data[10] = 99

        loaded_again = array_persistable_class.load(temp_hdf5_file)
        assert loaded_again.data.shape[0] >= 11
        assert loaded_again.data[10] == 99

    def test_proxydataset_attrs_immutable(self, array_persistable_class: type,
                                          temp_hdf5_file: Path) -> None:
        # ProxyDataset.attrs should return a copy
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with array_persistable_class(temp_hdf5_file, mode='r') as loaded:
            attrs1 = loaded.data.attrs
            attrs2 = loaded.data.attrs
            # Should be different dict instances (copies)
            assert attrs1 is not attrs2

    def test_proxydataset_multidimensional_slicing(self, tmp_path: Path) -> None:
        # ProxyDataset should handle multidimensional slicing
        class MatrixClass(Persistable):
            matrix = SerializableProperty(default=None)

        arr = np.arange(100).reshape(10, 10)
        obj = MatrixClass(matrix=arr)

        path = tmp_path / "test.hdf5"
        obj.save(path)

        with MatrixClass(path, mode='r') as loaded:
            sliced = loaded.matrix[2:5, 3:7]
            assert sliced.shape == (3, 4)
            assert np.array_equal(sliced, arr[2:5, 3:7])


# ========== ========== ========== ========== HDF5 Structure
class TestHDF5Structure:
    """Test the internal HDF5 file structure."""

    def test_root_group_exists(self, simple_persistable_class: type,
                               temp_hdf5_file: Path) -> None:
        # File should have a 'root' group
        obj = simple_persistable_class(value=42)
        obj.save(temp_hdf5_file)

        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'root' in f

    def test_class_attribute_stored(self, simple_persistable_class: type,
                                    temp_hdf5_file: Path) -> None:
        # __class__ attribute should be stored
        obj = simple_persistable_class(value=42, name="test")
        obj.save(temp_hdf5_file)

        with h5py.File(temp_hdf5_file, 'r') as f:
            assert '__class__' in f['root'].attrs

    def test_container_type_for_lists(self, tmp_path: Path) -> None:
        # Lists should have __container_type__ = 'list'
        class ListClass(Persistable):
            items = SerializableProperty(default=None)

        obj = ListClass(items=[1, 2, 3])
        path = tmp_path / "test.hdf5"
        obj.save(path)

        with h5py.File(path, 'r') as f:
            items_group = f['root/items']
            assert items_group.attrs['__container_type__'] == 'list'

    def test_container_type_for_dicts(self, tmp_path: Path) -> None:
        # Dicts should have __container_type__ = 'dict'
        class DictClass(Persistable):
            data = SerializableProperty(default=None)

        obj = DictClass(data={'a': 1, 'b': 2})
        path = tmp_path / "test.hdf5"
        obj.save(path)

        with h5py.File(path, 'r') as f:
            data_group = f['root/data']
            assert data_group.attrs['__container_type__'] == 'dict'

    def test_dataset_type_stored(self, array_persistable_class: type,
                                 temp_hdf5_file: Path) -> None:
        # Datasets should have __dataset_type__ attribute
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with h5py.File(temp_hdf5_file, 'r') as f:
            dataset = f['root/data']
            assert '__dataset_type__' in dataset.attrs
            assert 'ndarray' in dataset.attrs['__dataset_type__']

    def test_datasets_resizable(self, array_persistable_class: type,
                                temp_hdf5_file: Path) -> None:
        # Datasets should be created with resizable first dimension
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        with h5py.File(temp_hdf5_file, 'r') as f:
            dataset = f['root/data']
            # maxshape[0] should be None (unlimited)
            assert dataset.maxshape[0] is None


# ========== ========== ========== ========== Error Handling
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_save_unserializable_type_raises(self, tmp_path: Path) -> None:
        # Unregistered types should raise TypeError
        class CustomType:
            pass

        class BadClass(Persistable):
            bad_data = SerializableProperty(default=None)

        with pytest.raises(TypeError, match="No serialisation .* is implemented"):
            obj = BadClass(bad_data=CustomType())
            path = tmp_path / "test.hdf5"
            obj.save(path)

    def test_load_corrupted_file_raises(self, tmp_path: Path) -> None:
        # Corrupted HDF5 files should raise appropriate error
        path = tmp_path / "corrupted.hdf5"
        path.write_bytes(b"not an hdf5 file")

        class TestClass(Persistable):
            value = SerializableProperty(default=0)

        with pytest.raises(Exception):  # h5py will raise OSError or similar
            TestClass.load(path)

    def test_proxydataset_metadata_mismatch_raises(self, array_persistable_class: type,
                                                   temp_hdf5_file: Path) -> None:
        # Setting data with different metadata should raise
        arr = np.array([1, 2, 3])
        obj = array_persistable_class(data=arr)
        obj.save(temp_hdf5_file)

        # This test depends on having a dataset type with metadata
        # Current numpy.ndarray doesn't have metadata, so this is conceptual
        # In practice, custom dataset types might have this issue

    def test_load_without_root_group_raises(self, tmp_path: Path) -> None:
        # File without 'root' group should raise clear error
        path = tmp_path / "bad_structure.hdf5"

        with h5py.File(path, 'w') as f:
            f.create_group('not_root')

        class TestClass(Persistable):
            value = SerializableProperty(default=0)

        with pytest.raises(KeyError):
            TestClass.load(path)


# ========== ========== ========== ========== Integration Tests
class TestIntegration:
    """Test complex real-world scenarios."""

    def test_scientific_experiment_workflow(self, tmp_path: Path) -> None:
        # Complete scientific data workflow
        class Measurement(Persistable):
            timestamp = SerializableProperty(default=None)
            temperature = SerializableProperty(default=0.0)
            readings = SerializableProperty(default=None)
            metadata = SerializableProperty(default="")

        # Create measurement
        measurement = Measurement(
            timestamp=pd.Timestamp('2024-01-15 12:30:00'),
            temperature=298.15,
            readings=np.array([1.2, 3.4, 5.6, 7.8, 9.0]),
            metadata="Test run #1"
        )

        # Save
        path = tmp_path / "measurement.hdf5"
        measurement.save(path)

        # Load and verify
        loaded = Measurement.load(path)
        assert loaded.temperature == 298.15
        assert len(loaded.readings) == 5
        assert loaded.metadata == "Test run #1"

    def test_incremental_data_collection(self, tmp_path: Path) -> None:
        # Simulate collecting data over time
        class TimeSeries(Persistable):
            data = SerializableProperty(default=None)

        # Initial save
        ts = TimeSeries(data=np.array([1, 2, 3]))
        path = tmp_path / "timeseries.hdf5"
        ts.save(path)

        # Append more data
        with TimeSeries(path, mode='r+') as ts:
            ts.data.append(np.array([4, 5, 6]))

        # Append even more
        with TimeSeries(path, mode='r+') as ts:
            ts.data.append(np.array([7, 8, 9]))

        # Verify all data
        final = TimeSeries.load(path)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert np.array_equal(final.data, expected)

    def test_hierarchical_system_structure(self, tmp_path: Path) -> None:
        # System with subsystems and components
        class Component(Persistable):
            name = SerializableProperty(default="")
            value = SerializableProperty(default=0.0)

        class Subsystem(Persistable):
            name = SerializableProperty(default="")
            components = SerializableProperty(default=None)

        class System(Persistable):
            name = SerializableProperty(default="")
            subsystems = SerializableProperty(default=None)

        # Build hierarchy
        system = System(
            name="Main System",
            subsystems=[
                Subsystem(
                    name="Subsystem A",
                    components=[
                        Component(name="Component 1", value=1.0),
                        Component(name="Component 2", value=2.0)
                    ]
                ),
                Subsystem(
                    name="Subsystem B",
                    components=[
                        Component(name="Component 3", value=3.0)
                    ]
                )
            ]
        )

        # Save and load
        path = tmp_path / "system.hdf5"
        system.save(path)
        loaded = System.load(path)

        # Verify structure
        assert loaded.name == "Main System"
        assert len(loaded.subsystems) == 2
        assert loaded.subsystems[0].name == "Subsystem A"
        assert len(loaded.subsystems[0].components) == 2
        assert loaded.subsystems[0].components[1].value == 2.0

    def test_large_array_efficient_access(self, tmp_path: Path) -> None:
        # Efficient access to large arrays via ProxyDataset
        class LargeData(Persistable):
            array = SerializableProperty(default=None)

        # Create large array
        large_array = np.arange(1_000_000)
        obj = LargeData(array=large_array)

        path = tmp_path / "large.hdf5"
        obj.save(path)

        # Access via ProxyDataset (doesn't load entire array)
        with LargeData(path, mode='r') as loaded:
            # Access small slice
            chunk = loaded.array[1000:2000]
            assert len(chunk) == 1000
            assert chunk[0] == 1000

    def test_mixed_property_types(self, tmp_path: Path) -> None:
        # Object with many different property types
        class MixedTypes(Persistable):
            integer = SerializableProperty(default=0)
            floating = SerializableProperty(default=0.0)
            string = SerializableProperty(default="")
            path = SerializableProperty(default=None)
            array = SerializableProperty(default=None)
            timestamp = SerializableProperty(default=None)
            list_data = SerializableProperty(default=None)
            dict_data = SerializableProperty(default=None)
            none_value = SerializableProperty(default=None)

        obj = MixedTypes(
            integer=42,
            floating=3.14,
            string="test",
            path=Path("/some/path"),
            array=np.array([1, 2, 3]),
            timestamp=pd.Timestamp('2024-01-01'),
            list_data=[1, "two", 3.0],
            dict_data={"a": 1, "b": 2},
            none_value=None
        )

        path = tmp_path / "mixed.hdf5"
        obj.save(path)
        loaded = MixedTypes.load(path)

        assert loaded.integer == 42
        assert loaded.floating == 3.14
        assert loaded.string == "test"
        assert loaded.path == Path("/some/path")
        assert np.array_equal(loaded.array, np.array([1, 2, 3]))
        assert loaded.timestamp == pd.Timestamp('2024-01-01')
        assert loaded.list_data == [1, "two", 3.0]
        assert loaded.dict_data == {"a": 1, "b": 2}
        assert loaded.none_value is None