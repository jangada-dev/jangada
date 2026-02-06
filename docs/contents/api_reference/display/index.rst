#####################
Terminal Display
#####################

.. currentmodule:: jangada.display

The display module provides a framework for creating beautiful, styled terminal
output using the Rich library. Objects can define their visual representation
through customizable panels, tables, and forms with persistent theme support.


Overview
========

The display module provides two main classes:

- **DisplaySettings** - Persistent configuration for all display styling
- **Displayable** - Abstract base class for objects with Rich terminal display

Key features:

- Rich library integration for beautiful terminal output
- Customizable themes (colors, styles, layouts)
- Automatic table formatting with smart alignment
- Form-style key-value displays
- HTML/SVG export capabilities
- Persistent settings (save/load themes)

.. autosummary::
   :toctree: generated/

   DisplaySettings
   Displayable


Quick Start
===========

Basic displayable object::

    from jangada.display import Displayable
    from rich.text import Text

    class Sensor(Displayable):
        def __init__(self, name, temperature):
            self.name = name
            self.temperature = temperature

        def _title(self):
            return Text(f"Sensor: {self.name}", style="bold")

        def _content(self):
            return f"Temperature: {self.temperature}°C"

    sensor = Sensor("Temp-01", 23.5)
    print(sensor)  # Displays formatted panel

Table display::

    import pandas as pd
    from jangada.display import Displayable
    from rich.text import Text

    class DataView(Displayable):
        def __init__(self, df):
            self.df = df

        def _title(self):
            return Text("Data Table")

        def _content(self):
            return self.format_as_table(self.df, max_rows=20)

    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    view = DataView(df)
    print(view)

Custom styling::

    sensor = Sensor("Temp-01", 23.5)
    sensor.display_settings.panel_border_style = 'green'
    sensor.display_settings.console_width = 120
    print(sensor)

Save theme::

    settings = DisplaySettings()
    settings.panel_border_style = 'magenta'
    settings.table_header_style = 'bold white'
    settings.save('my_theme.disp')

    # Load later
    sensor.display_settings = DisplaySettings.load('my_theme.disp')


Class Reference
===============

DisplaySettings
---------------

.. autoclass:: DisplaySettings
    :members:
    :show-inheritance:

Displayable
-----------

.. autoclass:: Displayable
    :members:
    :inherited-members:
    :show-inheritance:


Core Concepts
=============

Template Method Pattern
-----------------------

Displayable uses the **template method pattern**: subclasses implement content,
framework handles formatting.

**Subclass implements:**

- ``_title()`` - Returns the panel title
- ``_content()`` - Returns the panel body

**Framework provides:**

- Panel creation and styling
- Border and title formatting
- Console rendering
- String capture
- Export functions

Example::

    class MyObject(Displayable):
        # You implement these:
        def _title(self):
            return Text("My Object")

        def _content(self):
            return "Object contents here"

    # Framework does the rest:
    obj = MyObject()
    print(obj)  # Formatted panel automatically

Rich Integration
----------------

Displayable integrates deeply with Rich:

**Magic methods:**

- ``__rich__()`` - Direct Rich rendering
- ``__str__()`` - Captured Rich output with ANSI codes

**Usage contexts:**

- Rich Console: ``console.print(obj)``
- IPython/Jupyter: Display ``obj`` directly
- Standard print: ``print(obj)``
- Logging: ``logger.info(str(obj))``

Example::

    from rich.console import Console

    console = Console()

    # All of these work:
    console.print(obj)  # Uses __rich__()
    print(obj)          # Uses __str__()
    obj                 # IPython display via __rich__()

Display Settings
----------------

Each Displayable instance has its own **DisplaySettings** object:

**Factory pattern:**

- Each instance gets independent settings
- No shared state by default
- Can share settings explicitly if desired

Example::

    obj1 = MyObject()
    obj2 = MyObject()

    # Independent settings
    obj1.display_settings.panel_border_style = 'red'
    obj2.display_settings.panel_border_style = 'blue'

    # Or share settings
    shared = DisplaySettings()
    obj1.display_settings = shared
    obj2.display_settings = shared


Helper Methods
==============

format_as_form
--------------

Creates key-value form displays:

**Basic usage**::

    def _content(self):
        return self.format_as_form({
            'Name': 'Alice',
            'Age': '25',
            'City': 'NYC'
        })

**With pandas Series**::

    import pandas as pd

    def _content(self):
        info = pd.Series({
            'Mean': self.data.mean(),
            'Std': self.data.std()
        })
        return self.format_as_form(info)

**Features:**

- Keys automatically get ``:`` appended
- Keys styled with ``property_style`` setting
- Values left-aligned
- Compact layout (no borders)

format_as_table
---------------

Creates formatted tables from DataFrames:

**Basic usage**::

    def _content(self):
        return self.format_as_table(self.df)

**With options**::

    def _content(self):
        return self.format_as_table(
            self.df,
            show_index=True,
            format_header_as_property=True,
            max_rows=20,
            round_floats=2
        )

**Features:**

- Auto-alignment by dtype (numbers right, text left)
- Optional index column display
- Truncation for large datasets (first n/2, ..., last n/2)
- Float rounding (global or per-column)
- Flexible column/header alignment
- Rich styling integration

**Alignment options**::

    # All columns same alignment
    table = self.format_as_table(df, align_column='right')

    # Per-column alignment
    table = self.format_as_table(df, align_column={
        'name': 'left',
        'value': 'right'
    })

    # Auto-detect (default)
    table = self.format_as_table(df, align_column=None)

**Float rounding**::

    # Round all floats to 2 decimals
    table = self.format_as_table(df, round_floats=2)

    # Per-column rounding
    table = self.format_as_table(df, round_floats={
        'price': 2,
        'percentage': 1
    })

**Truncation**::

    # Show first 10, ..., last 10 rows
    table = self.format_as_table(df, max_rows=21)


DisplaySettings Configuration
==============================

Console Settings
----------------

**console_width** : int
    Maximum console width in characters. Default: 150

    Example::

        settings.console_width = 120  # Narrower display
        settings.console_width = 200  # Wider display

Property Styling
----------------

**property_style** : str
    Rich style for property labels in forms. Default: 'bold bright_yellow'

    Example::

        settings.property_style = 'bold cyan'
        settings.property_style = 'italic bright_white'

Panel Styling
-------------

**panel_border_style** : str
    Border color/style. Default: 'bright_cyan'

    Example::

        settings.panel_border_style = 'green'
        settings.panel_border_style = 'bold red'
        settings.panel_border_style = '#FF00FF'

**panel_box** : str
    Box style name from rich.box. Default: 'ROUNDED'

    Valid values:

    - 'ROUNDED': ╭─╮ (friendly, default)
    - 'SQUARE': ┌─┐ (classic)
    - 'DOUBLE': ╔═╗ (formal)
    - 'HEAVY': ┏━┓ (bold)
    - 'MINIMAL': ╶─╴ (subtle)
    - 'ASCII': +--+ (compatibility)

    Example::

        settings.panel_box = 'DOUBLE'
        settings.panel_box = 'MINIMAL'

**panel_title_align** : str
    Title alignment. Default: 'center'

    Valid values: 'left', 'center', 'right'

    Example::

        settings.panel_title_align = 'left'

Table Styling
-------------

**table_index_style** : str or None
    Index column style. Default: None

    Example::

        settings.table_index_style = 'dim'
        settings.table_index_style = 'bold yellow'

**table_header_style** : str or None
    Header row style. Default: 'bold bright_yellow'

    Example::

        settings.table_header_style = 'bold white'
        settings.table_header_style = None  # No styling

**table_round_floats** : int or None
    Global float rounding. Default: None

    Example::

        settings.table_round_floats = 2  # Two decimals
        settings.table_round_floats = None  # Full precision

**table_spacing** : int
    Column spacing in characters. Default: 4

    Example::

        settings.table_spacing = 2  # Compact
        settings.table_spacing = 8  # Spacious


Usage Patterns
==============

Simple Text Display
-------------------

Display simple text with styling::

    class Message(Displayable):
        def __init__(self, title, text):
            self.title_text = title
            self.text = text

        def _title(self):
            return Text(self.title_text, style="bold cyan")

        def _content(self):
            return self.text

    msg = Message("Alert", "System update available")
    print(msg)

Styled Title
------------

Use Rich Text for advanced styling::

    from rich.text import Text

    def _title(self):
        return Text.assemble(
            ("Sensor ", "bold"),
            (self.name, "cyan"),
            (" [", "dim"),
            (self.id[:8], "yellow"),
            ("]", "dim")
        )

Dynamic Styling
---------------

Change style based on state::

    def _title(self):
        style = "green" if self.active else "red"
        return Text(self.name, style=f"bold {style}")

    def _content(self):
        status = "[green]Active[/green]" if self.active else "[red]Inactive[/red]"
        return f"Status: {status}"

Combined Layouts
----------------

Combine forms and tables::

    from rich.console import Group

    def _content(self):
        # Summary info
        info = self.format_as_form({
            'Total Samples': str(len(self.data)),
            'Mean': f'{self.data.mean():.2f}',
            'Std Dev': f'{self.data.std():.2f}'
        })

        # Detailed table
        table = self.format_as_table(
            self.data.head(10),
            max_rows=10
        )

        # Combine with blank line separator
        return Group(info, "", table)

Nested Panels
-------------

Create nested panel structures::

    from rich.panel import Panel

    def _content(self):
        # Inner panels
        info_panel = Panel(
            self.format_as_form(self.info),
            title="Info"
        )

        data_panel = Panel(
            self.format_as_table(self.data),
            title="Data"
        )

        # Group them
        return Group(info_panel, data_panel)

Conditional Content
-------------------

Show different content based on state::

    def _content(self):
        if not self.has_data:
            return "[dim italic]No data available[/dim italic]"

        if self.summary_mode:
            return self.format_as_form(self.get_summary())
        else:
            return self.format_as_table(self.data)


Advanced Topics
===============

Theme Management
----------------

**Create themes**::

    # Dark theme
    dark = DisplaySettings()
    dark.panel_border_style = 'bright_cyan'
    dark.property_style = 'bold bright_yellow'
    dark.save('dark.disp')

    # Light theme
    light = DisplaySettings()
    light.panel_border_style = 'blue'
    light.property_style = 'bold black'
    light.save('light.disp')

**Apply themes**::

    # Load and apply
    theme = DisplaySettings.load('dark.disp')
    obj.display_settings = theme

    # Or set directly
    obj.display_settings = DisplaySettings.load('corporate.disp')

**Share themes**::

    theme = DisplaySettings.load('theme.disp')

    for obj in objects:
        obj.display_settings = theme

Export Functions
----------------

**HTML export**::

    html = obj.to_html()

    with open('report.html', 'w') as f:
        f.write(html)

    # Embed in larger document
    full_html = f'''
    <html>
    <head><title>Report</title></head>
    <body>
        <h1>System Report</h1>
        {obj.to_html()}
    </body>
    </html>
    '''

**SVG export**::

    svg = obj.to_svg()

    with open('diagram.svg', 'w') as f:
        f.write(svg)

**Features:**

- Preserves colors and formatting
- Self-contained (inline styles)
- HTML includes CSS
- SVG is vector graphics (scalable)

Custom Panel Configuration
--------------------------

Override ``_display_panel()`` for complete control::

    def _display_panel(self):
        return Panel(
            self._content(),
            title=self._title(),
            subtitle="Footer text",  # Add footer
            border_style=self.display_settings.panel_border_style,
            padding=(1, 2),  # Custom padding
            expand=True,  # Fill width
        )

Rich Renderables
----------------

Return any Rich renderable from ``_content()``::

    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.table import Table

    def _content(self):
        # Tree structure
        tree = Tree("Components")
        for comp in self.components:
            tree.add(comp.name)
        return tree

    # Or syntax-highlighted code
    def _content(self):
        return Syntax(self.code, "python")

    # Or styled table
    def _content(self):
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name")
        table.add_column("Value")
        # ... add rows
        return table


Examples Gallery
================

Data Report
-----------

Scientific data report with statistics::

    class DataReport(Displayable):
        def __init__(self, name, data):
            self.name = name
            self.data = data

        def _title(self):
            return Text(f"Data Report: {self.name}", style="bold cyan")

        def _content(self):
            # Statistics
            stats = self.format_as_form({
                'Samples': str(len(self.data)),
                'Mean': f'{self.data.mean():.3f}',
                'Std Dev': f'{self.data.std():.3f}',
                'Min': f'{self.data.min():.3f}',
                'Max': f'{self.data.max():.3f}'
            })

            # Data preview
            preview = self.format_as_table(
                self.data.head(20),
                show_index=True,
                max_rows=20,
                round_floats=3
            )

            return Group(
                Panel(stats, title="Statistics"),
                "",
                Panel(preview, title="Data Preview")
            )

System Status
-------------

System status dashboard::

    class SystemStatus(Displayable):
        def __init__(self, system_name, metrics):
            self.system_name = system_name
            self.metrics = metrics

        def _title(self):
            return Text(f"⚙ {self.system_name}", style="bold")

        def _content(self):
            status_style = "green" if self.metrics['status'] == 'OK' else "red"

            info = {
                'Status': f"[{status_style}]{self.metrics['status']}[/{status_style}]",
                'Uptime': self.metrics['uptime'],
                'CPU': f"{self.metrics['cpu']}%",
                'Memory': f"{self.metrics['memory']}%",
                'Disk': f"{self.metrics['disk']}%"
            }

            return self.format_as_form(info)

Experiment Results
------------------

Experimental results with data table::

    class ExperimentResults(Displayable):
        def __init__(self, experiment_name, results_df):
            self.experiment_name = experiment_name
            self.results = results_df

        def _title(self):
            return Text.assemble(
                ("Experiment: ", "bold"),
                (self.experiment_name, "cyan")
            )

        def _content(self):
            return self.format_as_table(
                self.results,
                show_index=True,
                format_index_as_property=True,
                format_header_as_property=True,
                align_column='right',
                round_floats=4,
                max_rows=30
            )

Configuration Display
---------------------

Display configuration settings::

    class ConfigView(Displayable):
        def __init__(self, config_dict):
            self.config = config_dict

        def _title(self):
            return Text("Configuration", style="bold yellow")

        def _content(self):
            # Flatten nested config
            flat_config = {}
            for section, settings in self.config.items():
                for key, value in settings.items():
                    flat_config[f"{section}.{key}"] = str(value)

            return self.format_as_form(flat_config)


Best Practices
==============

Styling Guidelines
------------------

**Use semantic colors:**

- Green: Success, active, OK
- Red: Error, inactive, warning
- Yellow: Attention, highlight
- Cyan/Blue: Info, neutral
- Dim: Secondary info

**Example**::

    status = "[green]Active[/green]" if self.active else "[red]Inactive[/red]"
    priority = "[yellow]High[/yellow]" if self.priority > 5 else "Normal"

**Keep styles consistent:**

- Use display_settings for global styles
- Override only when needed
- Document color meanings

Content Organization
--------------------

**Progressive disclosure:**

- Summary info first
- Detailed data after
- Most important at top

**Example**::

    def _content(self):
        return Group(
            self._summary(),      # Key info
            "",
            self._details(),      # Full data
            "",
            self._footer()        # Meta info
        )

**Appropriate detail level:**

- Forms: 5-15 items
- Tables: 10-30 rows
- Text: 5-10 lines

**Use truncation:**

- max_rows for large datasets
- Summaries for long text
- Pagination if needed

Performance
-----------

**Avoid expensive operations in _content():**

- Cache computed values
- Pre-process data
- Don't query databases directly

**Example**::

    class DataView(Displayable):
        def __init__(self, data):
            self.data = data
            self._summary = self._compute_summary()  # Cache

        def _compute_summary(self):
            # Expensive computation done once
            return {...}

        def _content(self):
            # Use cached summary
            return self.format_as_form(self._summary)

Error Handling
--------------

**Handle missing data gracefully:**

::

    def _content(self):
        if self.data is None:
            return "[dim italic]No data available[/dim italic]"

        if len(self.data) == 0:
            return "[dim italic]No records found[/dim italic]"

        return self.format_as_table(self.data)

**Validate in __init__, not _content:**

::

    def __init__(self, data):
        if data is None:
            raise ValueError("Data cannot be None")
        self.data = data


Troubleshooting
===============

Common Issues
-------------

**Panel not displaying:**

- Check that _title() and _content() return valid types
- Ensure Rich is installed
- Verify console width setting

**Colors not showing:**

- Terminal may not support colors
- Check ``force_terminal`` setting
- Try different terminal emulator

**Table formatting issues:**

- DataFrame has invalid dtypes
- Missing required columns
- Index issues (try ``show_index=False``)

**Float rounding not working:**

- Check dtype is actually float
- Verify round_floats format (int or dict)
- Column names must match exactly in dict

Theme Issues
------------

**Theme not loading:**

::

    # Check file exists
    from pathlib import Path
    assert Path('theme.disp').exists()

    # Load with error handling
    try:
        settings = DisplaySettings.load('theme.disp')
    except Exception as e:
        print(f"Failed to load theme: {e}")

**Settings not applying:**

::

    # Verify settings object is assigned
    assert obj.display_settings is not None

    # Check specific setting
    print(obj.display_settings.panel_border_style)

Export Issues
-------------

**HTML/SVG missing content:**

- Content may not render in export context
- Some Rich features don't export well
- Try simpler renderables

**File encoding issues:**

::

    # Use explicit UTF-8
    html = obj.to_html()
    with open('output.html', 'w', encoding='utf-8') as f:
        f.write(html)


Performance Considerations
==========================

String Rendering
----------------

``__str__()`` captures Rich output to string:

- Creates StringIO buffer
- Renders to buffer
- Returns buffered string

For many objects, consider:

- Rendering once, caching result
- Direct ``__rich__()`` in Rich contexts
- Lazy rendering

Table Formatting
----------------

``format_as_table()`` creates DataFrame copy:

- Memory allocation for copy
- Type conversions (astype)
- String formatting for display

For large DataFrames:

- Use max_rows to limit size
- Pre-filter data
- Consider pagination

Display Settings
----------------

Each instance has independent settings:

- Small memory overhead
- Factory creates new instance each time
- Share settings to reduce overhead::

    shared = DisplaySettings()
    for obj in objects:
        obj.display_settings = shared


See Also
========

Related Modules
---------------

* :doc:`../serialization/persistable` - DisplaySettings uses Persistable
* :doc:`../mixin/nameable` - Often combined with Displayable
* :doc:`../system/system` - Uses Displayable for hierarchy display

External Resources
------------------

* `Rich Documentation <https://rich.readthedocs.io/>`_ - Rich library docs
* `Rich Renderables <https://rich.readthedocs.io/en/stable/protocol.html>`_ - Rich protocol
* `Rich Tables <https://rich.readthedocs.io/en/stable/tables.html>`_ - Table reference
* `Rich Text <https://rich.readthedocs.io/en/stable/text.html>`_ - Text styling