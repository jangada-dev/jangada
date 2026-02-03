**************
``Persistable``
**************

.. currentmodule:: jangada.serialization

.. autoclass:: Persistable
    :show-inheritance:


Initialization
==============

.. autosummary::
   :toctree: generated/

   Persistable.__init__


Context Manager Protocol
========================

Methods for interactive file access with lazy loading.

.. autosummary::
   :toctree: generated/

   Persistable.__enter__
   Persistable.__exit__


File I/O Methods
================

High-level methods for saving and loading objects.

.. autosummary::
   :toctree: generated/

   Persistable.save
   Persistable.load


Low-Level Serialization
=======================

Low-level methods for HDF5 data conversion.

.. autosummary::
   :toctree: generated/

   Persistable.save_serialized_data
   Persistable.load_serialized_data


Nested Classes
==============

.. autosummary::
   :toctree: generated/

   Persistable.ProxyDataset


Module-Level Functions
======================

.. autosummary::
   :toctree: generated/

   load


Overview
========

``Persistable`` extends :class:`Serializable` with HDF5 file I/O capabilities,
enabling efficient storage and retrieval of scientific data. It provides both
full object loading and lazy access via context managers.

Key Features
------------

**Three Access Modes**
   Load entire objects, access data lazily, or use context managers for
   interactive file manipulation.

**ProxyDataset**
   Lazy-loading wrapper for large arrays - access slices without loading
   entire datasets into memory.

**Automatic Type Mapping**
   Handles primitives, arrays, collections, and nested objects automatically.

**Resizable Datasets**
   Append to arrays without rewriting entire files.

**Metadata Preservation**
   Stores class information and dataset metadata for accurate reconstruction.


Usage Patterns
==============

Basic Save and Load
-------------------

Define a Persistable class::

    class Experiment(Persistable):
        name = SerializableProperty(default="")
        temperature = SerializableProperty(default=293.15)
        data = SerializableProperty(default=None)

Save to file::

    exp = Experiment(
        name="Test1",
        temperature=373.15,
        data=np.array([1.2, 3.4, 5.6])
    )
    exp.save('experiment.hdf5')

Load from file::

    # Method 1: Class method
    exp = Experiment.load('experiment.hdf5')

    # Method 2: Constructor
    exp = Experiment('experiment.hdf5')

    # Method 3: Module-level function
    from jangada.serialization import load
    exp = load('experiment.hdf5')


Context Manager for Lazy Loading
---------------------------------

For large files, use context manager mode to avoid loading everything::

    with Experiment('large_data.hdf5', mode='r') as exp:
        # exp.data is a ProxyDataset - not loaded yet
        print(exp.data.shape)  # (1000000,)

        # Load only what you need
        chunk = exp.data[100:200]  # Loads only 100 elements

        # Access metadata without loading data
        print(exp.name)
        print(exp.temperature)

Read modes:
   - ``'r'``: Read-only
   - ``'r+'``: Read and write
   - ``'w'``: Write (create new, truncate if exists)
   - ``'a'``: Read/write, create if doesn't exist


Modifying Data Efficiently
---------------------------

Update existing files without full reload::

    with Experiment('data.hdf5', mode='r+') as exp:
        # Modify single values
        exp.data[50] = 99.9

        # Modify slices
        exp.data[10:20] = np.zeros(10)

        # Append new data (efficient - no full rewrite)
        exp.data.append(np.array([7, 8, 9]))


Incremental Data Collection
----------------------------

Build datasets over time::

    # Initial creation
    exp = Experiment(
        name="Time Series",
        data=np.array([1.0, 2.0, 3.0])
    )
    exp.save('timeseries.hdf5')

    # Append data later
    for i in range(10):
        with Experiment('timeseries.hdf5', mode='r+') as exp:
            new_data = collect_measurements()
            exp.data.append(new_data)


Nested Objects
--------------

Hierarchical structures work automatically::

    class Measurement(Persistable):
        timestamp = SerializableProperty(default=None)
        value = SerializableProperty(default=0.0)

    class Experiment(Persistable):
        name = SerializableProperty(default="")
        measurements = SerializableProperty(default=None)

    exp = Experiment(
        name="Multi-point",
        measurements=[
            Measurement(timestamp=pd.Timestamp('2024-01-01'), value=1.2),
            Measurement(timestamp=pd.Timestamp('2024-01-02'), value=3.4),
            Measurement(timestamp=pd.Timestamp('2024-01-03'), value=5.6)
        ]
    )

    exp.save('experiment.hdf5')

    # Load preserves structure
    loaded = Experiment.load('experiment.hdf5')
    assert len(loaded.measurements) == 3
    assert loaded.measurements[0].value == 1.2


ProxyDataset
============

Overview
--------

``ProxyDataset`` is a lazy-loading wrapper for HDF5 datasets that provides
array-like access without loading data into memory until accessed.

When to Use
-----------

Use ProxyDataset (context manager mode) when:
   - Working with large arrays (GB-scale)
   - Only need to access parts of the data
   - Want to append data incrementally
   - Memory is limited

Use full loading (``load()`` method) when:
   - Arrays are small (MB-scale)
   - Need full numpy array operations
   - Will access most/all of the data
   - Memory is not a concern

Supported Operations
--------------------

ProxyDataset supports::

    # Indexing and slicing
    value = proxy[10]
    chunk = proxy[100:200]
    section = proxy[10:20, 5:15]  # Multidimensional

    # Assignment
    proxy[10] = 99
    proxy[10:20] = new_values

    # Appending
    proxy.append(new_data)

    # Properties
    proxy.shape
    proxy.dtype
    proxy.ndim
    proxy.size
    proxy.nbytes
    proxy.attrs  # Metadata

Not supported (load data first for these)::

    # Arithmetic operations
    result = proxy * 2  # ✗ Not supported

    # Instead:
    data = proxy[:]  # Load all
    result = data * 2  # ✓ Now works

    # Iteration
    for item in proxy:  # ✗ Not supported
        pass

    # Universal functions
    np.mean(proxy)  # ✗ Not supported
    np.mean(proxy[:])  # ✓ Load first

Examples
--------

Efficient slicing of large data::

    with Experiment('big_data.hdf5', mode='r') as exp:
        # Dataset is 10GB, only load what you need
        morning_data = exp.measurements[0:1000]
        afternoon_data = exp.measurements[5000:6000]

Appending to time series::

    # Start with initial data
    ts = TimeSeries(data=np.array([1, 2, 3]))
    ts.save('timeseries.hdf5')

    # Append over time
    for day in range(30):
        with TimeSeries('timeseries.hdf5', mode='r+') as ts:
            daily_data = collect_daily_measurements()
            ts.data.append(daily_data)

Auto-resizing datasets::

    with Experiment('data.hdf5', mode='r+') as exp:
        # Dataset currently has 100 elements
        print(exp.data.shape)  # (100,)

        # Setting beyond bounds auto-resizes
        exp.data[200] = 99.9
        print(exp.data.shape)  # (201,)


HDF5 File Structure
===================

File Organization
-----------------

Files are organized hierarchically::

    experiment.hdf5
    └── root (group)
        ├── __class__ = "mymodule.Experiment" (attribute)
        ├── name = "Test1" (attribute)
        ├── temperature = 373.15 (attribute)
        └── data (dataset)
            ├── __dataset_type__ = "numpy.ndarray" (attribute)
            └── [array data]

Type Mapping
------------

Python types map to HDF5 structures:

**Stored as Attributes**
   - ``None`` → ``'NoneType:None'``
   - ``str`` → Direct storage
   - ``int``, ``float``, ``complex`` → Direct storage
   - ``Path`` → ``'Path:/absolute/path'``

**Stored as Groups**
   - ``list`` → Group with ``__container_type__ = 'list'``
   - ``dict`` → Group with ``__container_type__ = 'dict'``
   - Nested ``Serializable`` → Group with ``__class__`` attribute

**Stored as Datasets**
   - ``numpy.ndarray`` → HDF5 dataset
   - ``pandas.Timestamp`` → Int64 dataset + timezone attribute
   - ``pandas.DatetimeIndex`` → Int64 dataset + timezone attribute
   - Custom dataset types → Via register_dataset_type

Example Structure
-----------------

For a complex object::

    class Experiment(Persistable):
        name = SerializableProperty(default="")
        metadata = SerializableProperty(default=None)
        measurements = SerializableProperty(default=None)

    exp = Experiment(
        name="Test",
        metadata={"pi": 3.14, "items": [1, 2, 3]},
        measurements=np.array([1.1, 2.2, 3.3])
    )

Produces::

    file.hdf5
    └── root/
        ├── @__class__ = "mymodule.Experiment"
        ├── @name = "Test"
        ├── metadata/  (group)
        │   ├── @__container_type__ = "dict"
        │   ├── @pi = 3.14
        │   └── items/  (group)
        │       ├── @__container_type__ = "list"
        │       ├── @0 = 1
        │       ├── @1 = 2
        │       └── @2 = 3
        └── measurements  (dataset)
            ├── @__dataset_type__ = "numpy.ndarray"
            └── [1.1, 2.2, 3.3]

(@ denotes attributes, / denotes groups/datasets)


Advanced Usage
==============

Custom File Extensions
----------------------

Override the default extension::

    class MyData(Persistable):
        extension = '.h5'
        data = SerializableProperty(default=None)

    obj = MyData(data=np.array([1, 2, 3]))
    obj.save('output')  # Saves as 'output.h5'

Multiple Files
--------------

Save different objects to different files::

    exp1 = Experiment(name="Morning")
    exp1.save('morning.hdf5')

    exp2 = Experiment(name="Afternoon")
    exp2.save('afternoon.hdf5')

    # Load them back
    experiments = [
        Experiment.load('morning.hdf5'),
        Experiment.load('afternoon.hdf5')
    ]

Batch Processing
----------------

Process large datasets in chunks::

    with LargeDataset('data.hdf5', mode='r') as dataset:
        chunk_size = 1000
        num_chunks = dataset.data.shape[0] // chunk_size

        results = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = dataset.data[start:end]
            results.append(process(chunk))

Compression
-----------

HDF5 supports compression, but this requires modifying the internal
``_save_data_in_group`` method. Future versions may support::

    class CompressedData(Persistable):
        compression = 'gzip'
        compression_opts = 4

(Not currently implemented - see Future Features in test file)


Performance Tips
================

Memory Efficiency
-----------------

**Use context manager for large files**::

    # Bad: Loads entire 10GB array into memory
    exp = Experiment.load('huge_data.hdf5')
    chunk = exp.data[100:200]

    # Good: Loads only requested chunk
    with Experiment('huge_data.hdf5', mode='r') as exp:
        chunk = exp.data[100:200]

**Append instead of rewriting**::

    # Bad: Loads, modifies, saves entire file
    exp = Experiment.load('data.hdf5')
    exp.data = np.concatenate([exp.data, new_values])
    exp.save('data.hdf5')

    # Good: Appends only new data
    with Experiment('data.hdf5', mode='r+') as exp:
        exp.data.append(new_values)

I/O Performance
---------------

**Batch small operations**::

    # Bad: Many small appends
    for value in values:
        with Experiment('data.hdf5', mode='r+') as exp:
            exp.data.append(np.array([value]))

    # Good: Single append with all data
    with Experiment('data.hdf5', mode='r+') as exp:
        exp.data.append(np.array(values))

**Access contiguous slices**::

    # Good: Contiguous access
    chunk = proxy[100:200]

    # Slower: Non-contiguous access
    elements = [proxy[i] for i in [10, 50, 100, 500]]

Storage Efficiency
------------------

**Use appropriate dtypes**::

    # Wasteful: float64 for integer data
    data = np.array([1, 2, 3], dtype=np.float64)  # 24 bytes

    # Efficient: int8 for small integers
    data = np.array([1, 2, 3], dtype=np.int8)  # 3 bytes

**Mark non-essential data as non-copiable**::

    class Analysis(Persistable):
        raw_data = SerializableProperty(default=None, copiable=True)
        # Cached result - don't save
        _cached = SerializableProperty(default=None, copiable=False)


Limitations
===========

Current Limitations
-------------------

1. **Circular References**
   Circular references will cause infinite recursion during save.

2. **Scalar Datasets**
   0-dimensional datasets cannot be resized or appended to.

3. **First Dimension Only**
   Datasets can only be resized along axis 0.

4. **No Compression Control**
   Cannot currently configure HDF5 compression from Python.

5. **No Partial Updates**
   Must save entire object or use context manager.

6. **ProxyDataset API Limited**
   Not a full numpy array replacement - supports slicing only.

7. **No File Locking**
   Multiple processes accessing same file may cause issues.

8. **No Version Migration**
   Changing class structure between saves requires manual handling.

See test_persistable.py header for complete list of limitations and
future enhancement plans.


Troubleshooting
===============

File Not Found
--------------

**Problem**: ``FileNotFoundError`` when loading.

**Solution**: Check path is correct::

    from pathlib import Path
    path = Path('data.hdf5')
    assert path.exists()

Permission Errors
-----------------

**Problem**: Cannot write to file.

**Solution**: Check file permissions and that file isn't open elsewhere::

    # Close any open context managers first
    with exp:
        pass  # File closes here

    # Now can reopen
    exp.save('data.hdf5')

Type Errors
-----------

**Problem**: ``TypeError: instances of X cannot be saved``

**Solution**: Register the type or make it Serializable::

    # Option 1: Register as primitive
    Serializable.register_primitive_type(MyType)

    # Option 2: Make it Serializable
    class MyType(Serializable):
        ...

Corrupted Files
---------------

**Problem**: File won't load after crash.

**Solution**: HDF5 files can be checked and repaired::

    # Check file integrity (external tool)
    $ h5debug data.hdf5

    # Or in Python
    import h5py
    try:
        with h5py.File('data.hdf5', 'r') as f:
            pass  # If this works, file is OK
    except Exception as e:
        print(f"File corrupted: {e}")


See Also
========

* :doc:`serializable` - Parent class
* :doc:`serializable_property` - Property descriptor
* :doc:`type_aliases` - Type definitions
* `HDF5 Documentation <https://docs.h5py.org/>`_ - h5py reference