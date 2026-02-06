# Jangada

A Python framework for modeling scientific systems with persistence and composable object capabilities.

## Overview

Jangada provides robust infrastructure components for data-intensive scientific applications, emphasizing:

- **Explicit serialization**: Descriptor-driven schemas with HDF5 persistence
- **Composable capabilities**: Orthogonal mixin classes for identity, metadata, and state
- **Rich terminal output**: Template-based formatting with the Rich library
- **System modeling**: Hierarchical system/subsystem paradigms with lazy loading

## Key Features

### Serialization & Persistence
- **SerializableProperty**: Descriptors with validation, change tracking, and lazy initialization
- **Serializable**: Registry-backed in-memory serialization protocol
- **Persistable**: HDF5 persistence layer with lazy dataset loading via ProxyDataset
- Support for NumPy arrays, Pandas DataFrames, and custom types

### Composable Mixins
Six orthogonal capability classes:
- **Identifiable**: Globally unique IDs with weak reference registry
- **Taggable**: Validated symbolic tags for organization
- **Nameable**: Human-readable names with validation
- **Describable**: Long-form descriptions
- **Colorable**: Canonical color representation (hex format)
- **Activatable**: Boolean activation state

### Display & Representation
- **Displayable**: Abstract base for Rich library integration
- **Representable**: Template method pattern for consistent formatting
- Customizable terminal output with panels, tables, and color

### System Modeling
- **System**: Container for subsystems with hierarchical organization
- Namespace management for system/subsystem paradigms
- Integration with all framework capabilities

## Installation

```bash
pip install jangada
```

For development:
```bash
pip install jangada[dev]
```

For documentation building:
```bash
pip install jangada[docs]
```

## Quick Start

```python
from jangada import Persistable, SerializableProperty
from jangada.mixin import Identifiable, Nameable, Taggable

class ScientificModel(Persistable, Identifiable, Nameable, Taggable):
    """A simple scientific model with persistence and metadata."""
    
    data = SerializableProperty(default=None)
    parameters = SerializableProperty(default=dict)
    
    def __init__(self, name: str, tag: str):
        self.name = name
        self.tag = tag
        self.parameters = {"learning_rate": 0.01}

# Create and persist
model = ScientificModel(name="Experiment-1", tag="ml_model")
model.save("/path/to/storage.h5")

# Load later
loaded_model = ScientificModel.load("/path/to/storage.h5")
print(f"Loaded {loaded_model.name} with ID {loaded_model.id}")
```

## Design Philosophy

Jangada follows these principles:

- **Modularity**: Orthogonal components that work independently
- **Composability**: Mix-and-match capabilities via multiple inheritance
- **Explicitness**: Clear schemas and configuration over magic
- **Efficiency**: Lazy loading for large datasets
- **Quality**: Comprehensive documentation, testing, and type hints

## Documentation

Full documentation is available at [readthedocs.io](https://jangada.readthedocs.io).

## Requirements

- Python >= 3.10
- numpy >= 1.20.0
- pandas >= 1.3.0
- h5py >= 3.0.0
- matplotlib >= 3.4.0
- rich >= 10.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Author

Rafael - [GitHub Profile](https://github.com/yourusername)
