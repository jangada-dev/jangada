#  -*- coding: utf-8 -*-
"""
Jangada: A Python framework for modeling scientific systems with persistence.

Jangada provides robust infrastructure components for data-intensive scientific
applications, emphasizing code quality through comprehensive documentation,
testing, and clean architectural patterns.

Key Features
------------
- **Explicit serialization**: Descriptor-driven schemas with HDF5 persistence
- **Composable capabilities**: Orthogonal mixin classes for identity, metadata, and state
- **Rich terminal output**: Template-based formatting with the Rich library
- **System modeling**: Hierarchical system/subsystem paradigms with lazy loading

Modules
-------
serialization
    Core serialization framework with SerializableProperty, Serializable, and Persistable
mixin
    Orthogonal capability mixins: Identifiable, Taggable, Nameable, Describable, Colorable, Activatable
display
    Base classes for Rich library integration: Displayable and Representable
system
    System/subsystem modeling with hierarchical organization

Examples
--------
Create a scientific model with persistence and metadata:

>>> from jangada import Persistable, SerializableProperty
>>> from jangada.mixin import Identifiable, Nameable, Taggable
>>>
>>> class Model(Persistable, Identifiable, Nameable, Taggable):
...     data = SerializableProperty(default=None)
...
...     def __init__(self, name: str, tag: str):
...         self.name = name
...         self.tag = tag
>>>
>>> model = Model(name="Experiment-1", tag="ml_model")
>>> model.save("/path/to/storage.h5")
>>> loaded = Model.load("/path/to/storage.h5")
>>> print(f"Loaded {loaded.name} with ID {loaded.id}")

Build hierarchical systems:

>>> from jangada import System
>>>
>>> parent = System(name="MainSystem", tag="root")
>>> child = System(name="Subsystem-A", tag="child_a")
>>> parent.add(child)
>>> print(parent.format())
"""


from .serialization import *
from .mixin import *
from .display import *
from .system import System


__all__ = [
    "Serializable",
    "Persistable",
    "SerializableProperty",
    "Identifiable",
    "Taggable",
    "Nameable",
    "Describable",
    "Colorable",
    "Activatable",
    "Displayable",
    "System",
]


try:
    # this will run if the jangada is installed
    from importlib.metadata import metadata, PackageNotFoundError

    meta = metadata('jangada')

    # Access specific fields
    __author__ = meta['Author-email']
    __license__ = meta['License']
    __version__ = meta['Version']

except PackageNotFoundError:
    # this will run during development
    import toml
    from pathlib import Path

    pyproject_filepath = Path(__file__).parent.parent / "pyproject.toml"

    with pyproject_filepath.open() as file:
        pyproject = toml.load(file)

    __version__ = pyproject["project"]["version"]
    __author__ = pyproject["project"]["authors"][0]["name"]
    __license__ = pyproject["project"]["license"]["text"]