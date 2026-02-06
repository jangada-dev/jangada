#  -*- coding: utf-8 -*-
"""
Rich terminal display formatting for domain objects.

This module provides a framework for creating beautiful, styled terminal output
using the Rich library. Objects define their visual representation through
customizable panels, tables, and forms with persistent theme support.
"""

from __future__ import annotations

import pandas

from abc import ABC, abstractmethod

from io import StringIO

from rich.markup import escape
from rich.text import Text
from rich.panel import Panel
from rich.console import Console, RenderableType
from rich.table import Table
from rich import box
from rich.align import Align

from typing import Any

from jangada import Persistable, SerializableProperty


# ========== ========== ========== ========== ========== ==========
class DisplaySettings(Persistable):
    """
    Persistent configuration for terminal display formatting.

    DisplaySettings provides centralized control over all visual aspects of
    terminal output, including console width, panel styling, and table
    formatting. Settings can be customized per-instance and saved as reusable
    themes.

    All styling properties use Rich's style syntax, supporting colors,
    attributes (bold, italic), and combinations.

    Class Attributes
    ----------------
    extension : str
        File extension for saved settings files ('.disp').

    Attributes
    ----------
    console_width : int
        Maximum console output width in characters. Default 150.
    property_style : str
        Style for property labels in forms. Default 'bold bright_yellow'.
    panel_border_style : str
        Style for panel borders. Default 'bright_cyan'.
    panel_box : str
        Box style name from rich.box. Default 'ROUNDED'.
    panel_title_align : str
        Panel title alignment. Default 'center'.
    table_index_style : str or None
        Style for table index column. Default None.
    table_header_style : str or None
        Style for table headers. Default 'bold bright_yellow'.
    table_round_floats : int or None
        Decimal places for float rounding. Default None.
    table_spacing : int
        Column spacing in characters. Default 4.

    Examples
    --------
    Create and customize settings::

        settings = DisplaySettings()
        settings.panel_border_style = 'green'
        settings.console_width = 120
        settings.table_spacing = 8

    Save as theme::

        settings.save('my_theme.disp')

    Load theme::

        settings = DisplaySettings.load('my_theme.disp')
        obj.display_settings = settings

    Notes
    -----
    As a Persistable subclass, DisplaySettings can be saved to HDF5 files
    with the .disp extension, making themes portable and shareable.

    Rich style strings support:
    - Colors: 'red', 'blue', 'bright_yellow', '#FF0000'
    - Modifiers: 'bold', 'italic', 'underline', 'dim'
    - Combinations: 'bold red', 'italic bright_cyan'

    See Also
    --------
    Displayable : Uses DisplaySettings for formatting
    """

    extension = '.disp'

    # ---------- ---------- ---------- ---------- console
    console_width: int = SerializableProperty(default=150, doc="""
        Maximum width for console output in characters.
        
        Controls how wide the output panel can be. Adjust based on your terminal
        size or display preferences. Wider displays may benefit from larger values,
        while smaller terminals should use lower values.
        
        Type
        ----
        int
        
        Default
        -------
        150
        
        Examples
        --------
        Standard terminal::
        
            settings.console_width = 120
        
        Wide display::
        
            settings.console_width = 200
        
        Narrow display::
        
            settings.console_width = 80
        """)

    # ---------- ---------- ---------- ---------- property
    property_style: str = SerializableProperty(default='bold bright_yellow', doc="""
        Rich style string for property labels in forms.
        
        Applied to keys in key-value forms created by format_as_form(). Makes
        property names visually distinct from their values.
        
        Type
        ----
        str
        
        Default
        -------
        'bold bright_yellow'
        
        Format
        ------
        Rich style syntax: "[modifiers] [color]"
        - Modifiers: bold, italic, underline, dim
        - Colors: color names, bright_*, or #RRGGBB
        
        Examples
        --------
        >>> settings.property_style = 'bold cyan'
        >>> settings.property_style = 'italic bright_white'
        >>> settings.property_style = 'underline #FF00FF'
        """)

    # ---------- ---------- ---------- ---------- panel
    panel_border_style: str = SerializableProperty(default='bright_cyan', doc = """
Rich color/style for panel borders.

Defines the color and style of the rectangular border around panels.
Can be any valid Rich color or style string.

Type
----
str

Default
-------
'bright_cyan'

Format
------
Rich style syntax: "[modifiers] [color]"

Examples
--------
>>> settings.panel_border_style = 'green'
>>> settings.panel_border_style = 'bold red'
>>> settings.panel_border_style = '#FF00FF'
>>> settings.panel_border_style = 'dim white'
""")
    panel_box: str = SerializableProperty(default='ROUNDED', doc = """
Box style name for panel borders.

Determines which characters are used for panel borders. Must match an
attribute name from the rich.box module.

Type
----
str

Default
-------
'ROUNDED'

Valid Values
------------
- 'ROUNDED': ╭─╮ (default, friendly appearance)
- 'SQUARE': ┌─┐ (classic box drawing)
- 'DOUBLE': ╔═╗ (formal double lines)
- 'HEAVY': ┏━┓ (bold/thick lines)
- 'MINIMAL': ╶─╴ (subtle thin lines)
- 'SIMPLE': ──  (plain horizontal lines)
- 'ASCII': +--+ (ASCII-only, high compatibility)

Examples
--------
>>> settings.panel_box = 'DOUBLE'   # Formal look
>>> settings.panel_box = 'HEAVY'    # Bold look
>>> settings.panel_box = 'MINIMAL'  # Subtle look
""")
    panel_title_align: str = SerializableProperty(default='center', doc = """
Panel title alignment within the top border.

Controls where the title text appears horizontally in the panel's
top border.

Type
----
str

Default
-------
'center'

Valid Values
------------
- 'left': Title at left side
- 'center': Title centered (default)
- 'right': Title at right side

Examples
--------
>>> settings.panel_title_align = 'left'
>>> settings.panel_title_align = 'center'
>>> settings.panel_title_align = 'right'
""")

    # ---------- ---------- ---------- ---------- table
    table_index_style: str|None = SerializableProperty(default=None, doc = """
Rich style for DataFrame index column.

Applied to the index column when format_as_table() is called with
show_index=True. Use None for no special styling.

Type
----
str or None

Default
-------
None

Examples
--------
>>> settings.table_index_style = 'dim'
>>> settings.table_index_style = 'italic bright_black'
>>> settings.table_index_style = 'bold yellow'
>>> settings.table_index_style = None  # No styling
""")
    table_header_style: str = SerializableProperty(default='bold bright_yellow', doc = """
Rich style for table column headers.

Applied to the first row (header row) in tables created by format_as_table().
Makes column names stand out from data rows.

Type
----
str or None

Default
-------
'bold bright_yellow'

Examples
--------
>>> settings.table_header_style = 'bold white'
>>> settings.table_header_style = 'underline cyan'
>>> settings.table_header_style = 'bold bright_magenta'
>>> settings.table_header_style = None  # No styling
""")
    table_round_floats: int|None = SerializableProperty(default=None, doc = """
Number of decimal places for float columns.

When set, all float columns in tables are rounded to this many decimal
places before display. Use None to show full precision.

Type
----
int or None

Default
-------
None (full precision)

Examples
--------
Two decimal places::

    settings.table_round_floats = 2
    # 3.14159 displays as "3.14"

Four decimal places::

    settings.table_round_floats = 4
    # 3.14159 displays as "3.1416"

Full precision::

    settings.table_round_floats = None
    # 3.14159 displays as "3.14159"

Notes
-----
This is a global setting. For per-column rounding, pass round_floats
argument to format_as_table() instead.
""")
    table_spacing: int = SerializableProperty(default=4, doc = """
Horizontal spacing between table columns in characters.

Controls the padding (number of spaces) between columns in tables.
Larger values make tables more spacious but wider.

Type
----
int

Default
-------
4

Examples
--------
Compact tables::

    settings.table_spacing = 2

Standard spacing::

    settings.table_spacing = 4

Spacious tables::

    settings.table_spacing = 8

Notes
-----
Spacing affects both column data and headers. Very small values (0-1)
may make columns hard to distinguish. Very large values (>10) may make
tables too wide for standard terminals.
""")


class Displayable(ABC):
    """
    Abstract base for objects with Rich terminal display.

    Displayable provides a framework for beautiful terminal output via Rich
    panels. Subclasses define content through abstract methods _title() and
    _content(), while the framework handles formatting, styling, and rendering.

    Integrates with Rich's protocol (__rich__) and provides string output
    (__str__), making objects displayable in both Rich-aware and standard
    contexts.

    Attributes
    ----------
    display_settings : DisplaySettings
        Configuration for display formatting. Auto-created per instance via
        factory, allowing independent customization.

    Abstract Methods
    ----------------
    _title() -> Text
        Generate panel title text.
    _content() -> RenderableType
        Generate panel body content.

    Examples
    --------
    Simple implementation::

        class Sensor(Displayable):
            def __init__(self, name, temp):
                self.name = name
                self.temp = temp

            def _title(self):
                return Text(f"Sensor: {self.name}")

            def _content(self):
                return f"Temperature: {self.temp}°C"

    With table::

        class Report(Displayable):
            def __init__(self, df):
                self.df = df

            def _title(self):
                return Text("Data Report")

            def _content(self):
                return self.format_as_table(self.df)

    Custom styling::

        obj = Sensor("Temp-01", 23.5)
        obj.display_settings.panel_border_style = 'green'
        print(obj)

    Notes
    -----
    Uses template method pattern: framework provides structure, subclasses
    provide content. This ensures consistent styling across all displays.

    See Also
    --------
    DisplaySettings : Configuration class
    format_as_form : Helper for key-value displays
    format_as_table : Helper for DataFrame displays
    """

    # ========== ========== ========== ========== ========== class attributes
    display_settings: DisplaySettings = SerializableProperty(copiable=False, doc = """
        Configuration for display formatting and styling.
        
        Provides access to all visual customization options for this object's
        terminal display. Each instance receives its own DisplaySettings object
        by default (via factory), allowing per-object customization without
        affecting other instances.
        
        Type
        ----
        DisplaySettings
        
        Default
        -------
        Auto-created DisplaySettings instance with default values (factory function)
        
        Constraints
        -----------
        None - can be freely assigned any DisplaySettings instance
        
        Examples
        --------
        Customize styling::
        
            obj.display_settings.panel_border_style = 'green'
            obj.display_settings.table_header_style = 'bold magenta'
            obj.display_settings.console_width = 120
        
        Access specific settings::
        
            width = obj.display_settings.console_width
            border = obj.display_settings.panel_border_style
            spacing = obj.display_settings.table_spacing
        
        Share settings across objects::
        
            shared = DisplaySettings()
            shared.panel_border_style = 'yellow'
            shared.table_spacing = 8
            
            obj1.display_settings = shared
            obj2.display_settings = shared
            # Both objects now share the same styling
        
        Load saved theme::
        
            theme = DisplaySettings.load('dark_theme.disp')
            obj.display_settings = theme
        
        Save current settings::
        
            obj.display_settings.save('my_custom_theme.disp')
        
        Temporary style changes::
        
            # Modify for one display
            obj.display_settings.console_width = 100
            print(obj)
            
            # Reset to defaults
            obj.display_settings = DisplaySettings()
        
        Notes
        -----
        **Factory Behavior**
            The display_settings property uses a factory function to create a new
            DisplaySettings instance for each Displayable object. This ensures:
            - Each instance has independent settings by default
            - No unexpected sharing between objects
            - Settings can be customized per-object without side effects
        
        **Sharing Settings**
            To share settings across multiple objects, explicitly assign the same
            DisplaySettings instance:
            
                shared_settings = DisplaySettings()
                obj1.display_settings = shared_settings
                obj2.display_settings = shared_settings
            
            Now changes to shared_settings affect both objects.
        
        **Persistence**
            DisplaySettings is Persistable, so settings can be:
            - Saved to .disp files (themes)
            - Loaded from saved themes
            - Versioned and shared across projects
            - Used to maintain consistent styling
        
        **Immediate Effect**
            Changes to display_settings take effect immediately on the next render:
            - print(obj) uses current settings
            - obj.__rich__() uses current settings
            - obj.to_html() uses current settings
            
            No need to "apply" or "refresh" - just change and use.
        
        **Available Settings**
            See DisplaySettings documentation for all available options:
            - Console: console_width
            - Panel: panel_border_style, panel_box, panel_title_align
            - Property labels: property_style
            - Tables: table_index_style, table_header_style, table_round_floats, table_spacing
        
        See Also
        --------
        DisplaySettings : The settings class with all options
        format_as_form : Uses property_style setting
        format_as_table : Uses table_* settings
        """)

    @display_settings.default
    def display_settings(self):
        """
        Factory for display settings.

        Returns
        -------
        DisplaySettings
            New DisplaySettings with defaults.
        """
        return DisplaySettings()

    # ========== ========== ========== ========== ========== special methods
    def __str__(self) -> str:
        """
        String representation with Rich formatting.

        Returns
        -------
        str
            Formatted panel with ANSI color codes.

        Notes
        -----
        Output includes ANSI codes (force_terminal=True) for colored terminal
        display. Width is controlled by display_settings.console_width.
        """
        string_io = StringIO()
        console = Console(file=string_io,
                          force_terminal=True,
                          width=self.display_settings.console_width)

        console.print(self._display_panel())

        return string_io.getvalue()

    def __rich__(self) -> RenderableType:
        """
        Rich protocol for direct rendering.

        Returns
        -------
        RenderableType
            Panel renderable for Rich console.

        Notes
        -----
        Enables direct printing via Rich Console, IPython/Jupyter display,
        and composition with other Rich renderables.
        """
        return self._display_panel()

    # ========== ========== ========== ========== ========== private methods
    ...

    # ========== ========== ========== ========== ========== protected methods
    @abstractmethod
    def _title(self) -> Text:
        """
        Generate panel title.

        Must be implemented by subclasses.

        Returns
        -------
        Text
            Title text (string or Rich Text object).

        Examples
        --------
        >>> def _title(self):
        ...     return Text("My Object", style="bold")
        """
        ...

    @abstractmethod
    def _content(self) -> RenderableType:
        """
        Generate panel body content.

        Must be implemented by subclasses.

        Returns
        -------
        RenderableType
            Body content (str, Text, Table, or any Rich renderable).

        Examples
        --------
        >>> def _content(self):
        ...     return self.format_as_table(self.data)
        """
        ...

    def _display_panel(self) -> Panel:
        """
        Create formatted panel with current settings.

        Returns
        -------
        Panel
            Configured Rich Panel.

        Notes
        -----
        Combines _title() and _content() with display_settings styling.
        Can be overridden for complete panel customization.
        """
        return Panel(
            self._content(),
            title=self._title(),
            border_style=self.display_settings.panel_border_style,
            title_align=self.display_settings.panel_title_align,
            expand=False,
            box=getattr(box, self.display_settings.panel_box)
        )

    # ========== ========== ========== ========== ========== public methods
    def format_as_form(self, data: dict[str, str]|pandas.Series) -> Table:
        """
        Format data as key-value form.

        Creates two-column table with keys (left) and values (right).

        Parameters
        ----------
        data : dict[str, str] or pandas.Series
            Key-value pairs to display.

        Returns
        -------
        Table
            Rich Table in form layout.

        Examples
        --------
        >>> form = self.format_as_form({
        ...     'Name': 'Alice',
        ...     'Age': '25',
        ...     'City': 'NYC'
        ... })

        Notes
        -----
        Keys automatically get ':' appended and use property_style.
        Values are left-aligned with no special styling.
        """
        form = Table.grid(padding=(0, 4), expand=False)
        form.add_column(justify='left', style=self.display_settings.property_style)
        form.add_column(justify='left', style=None)

        for prop, value in data.items():
            form.add_row(f'{prop}:', value)

        return form

    def format_as_table(self,
                        frame: pandas.DataFrame,
                        show_index: bool = True,
                        format_index_as_property: bool = False,
                        format_header_as_property: bool = False,
                        align_header: str|dict[str, str] = 'center',
                        align_column: str|dict[str, str]|None = None,
                        max_rows: int = 31,
                        **kwargs) -> Table:
        """
        Format DataFrame as Rich table.

        Creates formatted table with automatic alignment, optional truncation,
        and extensive customization options.

        Parameters
        ----------
        frame : pandas.DataFrame
            Data to display.
        show_index : bool, optional
            Include index column. Default True.
        format_index_as_property : bool, optional
            Use property_style for index. Default False.
        format_header_as_property : bool, optional
            Use property_style for headers. Default False.
        align_header : str or dict[str, str], optional
            Header alignment. Default 'center'.
        align_column : str, dict[str, str], or None, optional
            Column alignment. None = auto-detect. Default None.
        max_rows : int, optional
            Max rows before truncation. Default 31.
        **kwargs
            Additional options:
            - header_style : str or None
            - index_style : str or None
            - round_floats : int, dict, or None

        Returns
        -------
        Table
            Formatted Rich Table.

        Raises
        ------
        TypeError
            If alignment or rounding parameters have invalid types.

        Examples
        --------
        Basic usage::

            >>> table = self.format_as_table(df)

        Custom alignment::

            >>> table = self.format_as_table(df, align_column='right')

        Float rounding::

            >>> table = self.format_as_table(df, round_floats=2)

        Large dataset::

            >>> table = self.format_as_table(df, max_rows=20)

        Notes
        -----
        Auto-alignment (align_column=None): numeric right, datetime/text left.

        Truncation: Shows first n/2 rows, '...', last n/2 rows when > max_rows.

        Creates DataFrame copy for processing.
        """
        # ---------- ---------- resolve header style
        if format_header_as_property:
            header_style = self.display_settings.property_style
        else:
            header_style: str|None = kwargs.pop('header_style', self.display_settings.table_header_style)

        # ---------- ---------- resolve header length
        if show_index:

            # ---------- ---------- resolve index style
            if format_index_as_property:
                index_style = self.display_settings.property_style
            else:
                index_style: str|None = kwargs.pop('index_style', self.display_settings.table_index_style)

            _frame = frame.reset_index()

        else:
            _frame = frame.copy()

        # ---------- ---------- ---------- ---------- column names
        columns = _frame.columns

        # ---------- ---------- ---------- ---------- create table
        table = Table.grid(padding=(0, self.display_settings.table_spacing), expand=False)

        # ---------- ---------- resolve column alignments
        if isinstance(align_column, str):

            for _ in columns:
                table.add_column(justify=align_column)

        elif isinstance(align_column, dict):

            for column in columns:

                if column in align_column:
                    table.add_column(justify=align_column[column])

                else:
                    table.add_column(justify='left')

        elif align_column is None:

            for column in columns:

                if pandas.api.types.is_datetime64_any_dtype(_frame[column]):
                    table.add_column(justify='left')

                elif pandas.api.types.is_numeric_dtype(_frame[column]):
                    table.add_column(justify='right')

                else:
                    table.add_column(justify='left')

        else:
            raise TypeError(f'Invalid type for align_column argument: {type(align_column)}')

        # ---------- ---------- resolve header alignment
        if isinstance(align_header, str):
            table.add_row(*(Align(escape(col), align_header) for col in columns), style=header_style)

        elif isinstance(align_header, dict):
            table.add_row(*(Align(escape(col), align_header[col]) for col in columns), style=header_style)

        else:
            raise TypeError(f'Invalid type for align_header argument: {type(align_header)}')

        # ---------- ---------- ---------- ---------- rounding floats
        round_floats: int|dict[str, int]|None = kwargs.pop('round_floats', self.display_settings.table_round_floats)

        if round_floats is not None:

            if isinstance(round_floats, int):

                for col in _frame.select_dtypes(include='float').columns:
                    _frame[col] = _frame[col].apply(lambda val: f'{val:.{round_floats}f}')

            elif isinstance(round_floats, dict):

                for col, round_value in round_floats.items():
                    _frame[col] = _frame[col].apply(lambda val: f'{val:.{round_value}f}')

            else:
                raise TypeError(f'Invalid type for round_floats argument: {type(round_floats)}')

        # ---------- ---------- ---------- ---------- populate table
        __frame = _frame.astype(str)

        if len(__frame) <= max_rows:
            for count, row in __frame.iterrows():

                if show_index:
                    _index = Text(row.values[0], style=index_style)
                    table.add_row(_index, *row.values[1:])
                else:
                    table.add_row(*row.values)

        else:
            n_rows: int = (max_rows - 1) // 2

            for count, row in __frame.head(n_rows).iterrows():

                if show_index:
                    _index = Text(row.values[0], style=index_style)
                    table.add_row(_index, *row.values[1:])
                else:
                    table.add_row(*row.values)

            table.add_row(*(Align.center('...') for _ in columns))

            for count, row in __frame.tail(n_rows).iterrows():

                if show_index:
                    _index = Text(row.values[0], style=index_style)
                    table.add_row(_index, *row.values[1:])
                else:
                    table.add_row(*row.values)

        # ---------- ---------- ---------- ---------- ---------- ----------
        return table

    def to_html(self) -> str:
        """
        Export display as HTML.

        Returns
        -------
        str
            HTML with inline styles.

        Examples
        --------
        >>> html = obj.to_html()
        >>> with open('output.html', 'w') as f:
        ...     f.write(html)
        """
        console = Console(record=True)
        console.print(self)
        return console.export_html()

    def to_svg(self) -> str:
        """
        Export display as SVG.

        Returns
        -------
        str
            SVG vector graphics.

        Examples
        --------
        >>> svg = obj.to_svg()
        >>> with open('output.svg', 'w') as f:
        ...     f.write(svg)
        """
        console = Console(record=True)
        console.print(self)
        return console.export_svg()

    # ---------- ---------- ---------- ---------- ---------- properties
    ...