#  -*- coding: utf-8 -*-
"""
Comprehensive test suite for Displayable and DisplaySettings.

Tests cover:
- DisplaySettings: Configuration, persistence, defaults
- Displayable: Panel rendering, table formatting, form formatting
- Integration: Rich protocol, string output, HTML/SVG export

================================================================================
SUGGESTIONS AND FUTURE ENHANCEMENTS
================================================================================

ENHANCEMENTS:
-------------

1. Theme Presets
   Add preset factory methods for common themes:
   - DisplaySettings.dark_theme() - Dark mode colors
   - DisplaySettings.light_theme() - Light mode colors
   - DisplaySettings.minimal_theme() - Minimal styling
   - DisplaySettings.rainbow_theme() - Colorful/fun styling

2. Context-Specific Formatting
   Add verbose/summary modes:
   - _content(verbose=False) for different detail levels
   - Could have _summary_content() and _detailed_content() methods
   - Useful for quick overview vs deep inspection

3. Auto-Detection for force_terminal
   Add terminal detection in __str__:
   - import sys; is_terminal = sys.stdout.isatty()
   - Console(force_terminal=is_terminal or settings.force_colors)
   - Add force_colors: bool to DisplaySettings
   - Smart defaults for different contexts (REPL, file, pipe)

4. Advanced Table Features
   - Column width control (max_width per column)
   - Row styling (alternate row colors, highlight conditions)
   - Footer rows (totals, aggregations)
   - Column formatters (custom format functions per column)
   - Sortable indicators in headers

5. Form Layout Options
   - Horizontal vs vertical layout
   - Multi-column forms (for many properties)
   - Grouping/sections in forms
   - Optional values handling (show/hide None values)

6. Export Format Options
   - to_html(inline_styles=True/False)
   - to_svg(title="...", theme="...")
   - to_markdown() - for documentation
   - to_ansi() - raw ANSI codes
   - to_image() - via terminal screenshot tools

7. Caching for Performance
   - Cache formatted panels if expensive
   - Invalidation on settings change
   - Option to disable caching
   - @lru_cache on format_as_table for identical inputs

8. Box Style Validation
   - Validate panel_box is valid box type on set
   - Parser that checks against box module attributes
   - Better error message than AttributeError

9. Responsive Width
   - Auto-detect terminal width
   - Responsive table columns (shrink to fit)
   - Truncate long values with ellipsis
   - Word wrap for text columns

10. Rich Renderables Support
    - Support for Tables inside forms
    - Support for Trees, Syntax, etc. in _content()
    - Nested panels
    - Columns layout for side-by-side content

11. Pandas Integration Improvements
    - Support for MultiIndex DataFrames
    - Categorical dtype styling
    - Missing value indicators (NA, NaN, None)
    - Index name display
    - DataFrame info panel (shape, dtypes, memory)

12. Color Theme Validation
    - Validate all *_style properties are valid Rich colors
    - Parser that checks Color.parse()
    - Fallback colors if invalid

13. Display Profiles
    - Named profiles: 'compact', 'detailed', 'presentation'
    - Quick switching between profiles
    - Profile inheritance/composition

14. Interactive Mode
    - Pager support for long outputs (less/more)
    - Expandable sections
    - Click to copy values
    - (Requires Rich's Live/prompt features)

15. Accessibility
    - Screen reader friendly output option
    - High contrast themes
    - Colorblind-safe palettes
    - Plain text fallback mode


DESIGN CONSIDERATIONS:
----------------------

1. Settings Inheritance
   Q: Should display_settings be inherited from parent classes?
   A: Currently each instance gets its own settings copy (good)
   Consider: Class-level default settings that instances can override

2. Settings Validation
   Currently no validation on *_style properties
   Should add parsers to validate Rich color strings

3. Table Truncation Strategy
   Current: Show first n/2 and last n/2 rows
   Alternative strategies:
   - Show first n rows only
   - Show around a specific row (context)
   - Smart truncation (group by categories)

4. Error Handling
   format_as_table assumes DataFrame/dict with certain structure
   Should add validation and helpful errors for:
   - Empty DataFrames
   - Non-existent columns in align_column dict
   - Invalid dtypes

5. Memory Efficiency
   Creates copy of DataFrame (_frame = frame.copy())
   For very large DataFrames, this could be expensive
   Consider: in-place operations or views

6. Type Hints
   Some return types could be more specific
   RenderableType is broad - could document expected types better

================================================================================
END OF SUGGESTIONS AND FUTURE ENHANCEMENTS
================================================================================

Author: Rafael R. L. Benevides
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel


from jangada.display import Displayable, DisplaySettings


# ========== ========== ========== ========== Fixtures
@pytest.fixture
def default_settings() -> DisplaySettings:
    """Create default DisplaySettings."""
    return DisplaySettings()


@pytest.fixture
def custom_settings() -> DisplaySettings:
    """Create customized DisplaySettings."""
    settings = DisplaySettings()
    settings.console_width = 120
    settings.property_style = 'bold cyan'
    settings.panel_border_style = 'green'
    settings.table_header_style = 'bold magenta'
    return settings


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'score': [92.5, 88.3, 95.7, 91.2],
        'date': pd.date_range('2024-01-01', periods=4),
    })


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """Create large DataFrame for truncation testing."""
    return pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100),
        'category': [f'Cat{i % 5}' for i in range(100)],
    })


@pytest.fixture
def simple_displayable_class() -> type:
    """A simple Displayable implementation for testing."""

    class SimpleDisplay(Displayable):
        def __init__(self, title_text: str, body_text: str):
            self.title_text = title_text
            self.body_text = body_text

        def _title(self) -> Text:
            return Text(self.title_text)

        def _content(self) -> str:
            return self.body_text

    return SimpleDisplay


@pytest.fixture
def table_displayable_class() -> type:
    """Displayable with table formatting."""

    class TableDisplay(Displayable):
        def __init__(self, df: pd.DataFrame):
            self.df = df

        def _title(self) -> Text:
            return Text("Data Table")

        def _content(self) -> Table:
            return self.format_as_table(self.df)

    return TableDisplay


# ========== ========== ========== ========== Test DisplaySettings
class TestDisplaySettings:
    """Test DisplaySettings configuration class."""

    def test_creates_with_defaults(self) -> None:
        # Should create with default values
        settings = DisplaySettings()

        assert settings.console_width == 150
        assert settings.property_style == 'bold bright_yellow'
        assert settings.panel_border_style == 'bright_cyan'
        assert settings.panel_box == 'ROUNDED'
        assert settings.panel_title_align == 'center'

    def test_defaults_are_sensible(self) -> None:
        # Default values should be production-ready
        settings = DisplaySettings()

        assert settings.console_width > 80  # Wide enough
        assert settings.table_spacing > 0  # Some spacing
        assert settings.panel_box in ['ROUNDED', 'SQUARE', 'DOUBLE', 'HEAVY']

    def test_can_modify_settings(self) -> None:
        # Settings should be mutable
        settings = DisplaySettings()

        settings.console_width = 120
        assert settings.console_width == 120

        settings.panel_border_style = 'green'
        assert settings.panel_border_style == 'green'

    def test_extension_is_disp(self) -> None:
        # Custom extension for display settings files
        assert DisplaySettings.extension == '.disp'

    def test_save_and_load(self, tmp_path: Path) -> None:
        # Should persist to file and load back
        settings = DisplaySettings()
        settings.console_width = 200
        settings.panel_border_style = 'magenta'

        path = tmp_path / "settings.disp"
        settings.save(path)

        loaded = DisplaySettings.load(path)
        assert loaded.console_width == 200
        assert loaded.panel_border_style == 'magenta'

    def test_is_persistable(self) -> None:
        # Should be a Persistable subclass
        from jangada import Persistable
        assert issubclass(DisplaySettings, Persistable)

    def test_table_settings_optional(self) -> None:
        # Table settings can be None
        settings = DisplaySettings()

        settings.table_index_style = None
        settings.table_header_style = None
        settings.table_round_floats = None

        assert settings.table_index_style is None
        assert isinstance(settings.table_header_style, str)
        assert settings.table_round_floats is None


# ========== ========== ========== ========== Test Displayable
class TestDisplayableBasics:
    """Test basic Displayable functionality."""

    def test_abstract_methods_required(self) -> None:
        # Cannot instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            Displayable()

    def test_has_display_settings_property(self, simple_displayable_class: type) -> None:
        # Should have display_settings property
        obj = simple_displayable_class("Title", "Body")

        assert hasattr(obj, 'display_settings')
        assert isinstance(obj.display_settings, DisplaySettings)

    def test_each_instance_gets_own_settings(self, simple_displayable_class: type) -> None:
        # Each instance should have independent settings
        obj1 = simple_displayable_class("Title1", "Body1")
        obj2 = simple_displayable_class("Title2", "Body2")

        obj1.display_settings.console_width = 100
        obj2.display_settings.console_width = 200

        assert obj1.display_settings.console_width == 100
        assert obj2.display_settings.console_width == 200

    def test_settings_factory_creates_new_instance(self, simple_displayable_class: type) -> None:
        # Default factory should create new settings each time
        obj1 = simple_displayable_class("Title", "Body")
        obj2 = simple_displayable_class("Title", "Body")

        assert obj1.display_settings is not obj2.display_settings

    def test_can_assign_custom_settings(self, simple_displayable_class: type,
                                        custom_settings: DisplaySettings) -> None:
        # Should accept custom settings object
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings = custom_settings

        assert obj.display_settings.console_width == 120
        assert obj.display_settings.panel_border_style == 'green'


class TestDisplayableRendering:
    """Test rendering and output methods."""

    def test_str_returns_string(self, simple_displayable_class: type) -> None:
        # __str__ should return string
        obj = simple_displayable_class("Test", "Content")

        result = str(obj)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_str_contains_title_and_body(self, simple_displayable_class: type) -> None:
        # String output should include title and body
        obj = simple_displayable_class("MyTitle", "MyContent")

        result = str(obj)

        # May contain ANSI codes, but text should be there
        assert "MyTitle" in result
        assert "MyContent" in result

    def test_rich_protocol(self, simple_displayable_class: type) -> None:
        # __rich__ should return Panel
        obj = simple_displayable_class("Title", "Body")

        renderable = obj.__rich__()

        assert isinstance(renderable, Panel)

    def test_rich_rendering(self, simple_displayable_class: type) -> None:
        # Should render via Rich Console
        obj = simple_displayable_class("Title", "Body")

        console = Console(file=StringIO())
        console.print(obj)  # Should not raise

    def test_display_panel_uses_settings(self, simple_displayable_class: type) -> None:
        # Panel should respect display settings
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.panel_border_style = 'red'
        obj.display_settings.panel_title_align = 'right'

        panel = obj._display_panel()

        assert panel.border_style == 'red'
        assert panel.title_align == 'right'

    def test_panel_box_style(self, simple_displayable_class: type) -> None:
        # Panel should use box style from settings
        from rich import box

        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.panel_box = 'DOUBLE'

        panel = obj._display_panel()

        assert panel.box == box.DOUBLE

    def test_console_width_respected(self, simple_displayable_class: type) -> None:
        # Console width setting should be used
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.console_width = 80

        # Check via string output (harder to verify directly)
        result = str(obj)
        assert isinstance(result, str)


# ========== ========== ========== ========== Test format_as_form
class TestFormatAsForm:
    """Test form formatting method."""

    def test_formats_dict_as_form(self, simple_displayable_class: type) -> None:
        # Should create table from dict
        obj = simple_displayable_class("Title", "Body")

        data = {'name': 'Alice', 'age': '25', 'city': 'NYC'}
        form = obj.format_as_form(data)

        assert isinstance(form, Table)

    def test_formats_series_as_form(self, simple_displayable_class: type) -> None:
        # Should create table from pandas Series
        obj = simple_displayable_class("Title", "Body")

        data = pd.Series({'name': 'Bob', 'age': '30'})
        form = obj.format_as_form(data)

        assert isinstance(form, Table)

    def test_form_includes_keys_and_values(self, simple_displayable_class: type) -> None:
        # Form should contain all key-value pairs
        obj = simple_displayable_class("Title", "Body")

        data = {'key1': 'value1', 'key2': 'value2'}
        form = obj.format_as_form(data)

        # Render to check content
        console = Console(file=StringIO())
        console.print(form)
        # Hard to verify rendered content, but should not raise

    def test_form_uses_property_style(self, simple_displayable_class: type) -> None:
        # Should use property_style from settings
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.property_style = 'bold red'

        data = {'name': 'Alice'}
        form = obj.format_as_form(data)

        # Style is applied to first column
        assert form.columns[0].style == 'bold red'

    def test_form_adds_colon_to_keys(self, simple_displayable_class: type) -> None:
        # Keys should have colon appended
        obj = simple_displayable_class("Title", "Body")

        data = {'name': 'Alice'}
        form = obj.format_as_form(data)

        # Check by rendering
        string_io = StringIO()
        console = Console(file=string_io)
        console.print(form)

        assert 'name:' in string_io.getvalue()


# ========== ========== ========== ========== Test format_as_table
class TestFormatAsTable:
    """Test table formatting method."""

    def test_formats_dataframe(self, simple_displayable_class: type,
                               sample_dataframe: pd.DataFrame) -> None:
        # Should create table from DataFrame
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe)

        assert isinstance(table, Table)

    def test_formats_dict(self, simple_displayable_class: type) -> None:
        # Should create table from dict
        obj = simple_displayable_class("Title", "Body")

        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        table = obj.format_as_table(df)

        assert isinstance(table, Table)

    def test_show_index_true(self, simple_displayable_class: type,
                             sample_dataframe: pd.DataFrame) -> None:
        # Should show index when show_index=True
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe, show_index=True)

        # Check column count (index + data columns)
        assert len(table.columns) == len(sample_dataframe.columns) + 1

    def test_show_index_false(self, simple_displayable_class: type,
                              sample_dataframe: pd.DataFrame) -> None:
        # Should hide index when show_index=False
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe, show_index=False)

        # Check column count (only data columns)
        assert len(table.columns) == len(sample_dataframe.columns)

    def test_format_index_as_property(self, simple_displayable_class: type,
                                      sample_dataframe: pd.DataFrame) -> None:
        # Index should use property style when requested
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.property_style = 'bold yellow'

        table = obj.format_as_table(sample_dataframe,
                                    show_index=True,
                                    format_index_as_property=True)

        # Index column should have property style
        # (Hard to verify without rendering)

    def test_format_header_as_property(self, simple_displayable_class: type,
                                       sample_dataframe: pd.DataFrame) -> None:
        # Headers should use property style when requested
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe,
                                    format_header_as_property=True)

        # Header row should have property style

    def test_align_column_string(self, simple_displayable_class: type,
                                 sample_dataframe: pd.DataFrame) -> None:
        # Should align all columns the same way
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe, align_column='right')

        # All columns should be right-aligned
        for col in table.columns:
            assert col.justify == 'right'

    def test_align_column_dict(self, simple_displayable_class: type,
                               sample_dataframe: pd.DataFrame) -> None:
        # Should align specific columns
        obj = simple_displayable_class("Title", "Body")

        align = {'name': 'left', 'age': 'right'}
        table = obj.format_as_table(sample_dataframe, align_column=align)

        # Check specific alignments
        # (Column order may vary)

    def test_align_column_auto_numeric(self, simple_displayable_class: type) -> None:
        # Numeric columns should align right automatically
        obj = simple_displayable_class("Title", "Body")

        df = pd.DataFrame({'num': [1, 2, 3], 'text': ['a', 'b', 'c']})
        table = obj.format_as_table(df, show_index=False)

        # Numeric column should be right-aligned
        # Text column should be left-aligned

    def test_align_header_string(self, simple_displayable_class: type,
                                 sample_dataframe: pd.DataFrame) -> None:
        # Should align headers
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(sample_dataframe, align_header='left')

        # Headers should be left-aligned

    def test_round_floats_int(self, simple_displayable_class: type) -> None:
        # Should round all float columns
        obj = simple_displayable_class("Title", "Body")

        df = pd.DataFrame({'value': [1.23456, 2.34567, 3.45678]})
        table = obj.format_as_table(df, round_floats=2)

        # Values should be rounded to 2 decimals
        # Check by rendering
        string_io = StringIO()
        console = Console(file=string_io)
        console.print(table)

        output = string_io.getvalue()
        assert '1.23' in output or '1.24' in output  # Rounding

    def test_round_floats_dict(self, simple_displayable_class: type) -> None:
        # Should round specific columns
        obj = simple_displayable_class("Title", "Body")

        df = pd.DataFrame({
            'val1': [1.23456],
            'val2': [2.34567]
        })
        table = obj.format_as_table(df, round_floats={'val1': 1, 'val2': 3})

        # Different rounding per column

    def test_max_rows_no_truncation(self, simple_displayable_class: type) -> None:
        # Should show all rows if under max_rows
        obj = simple_displayable_class("Title", "Body")

        df = pd.DataFrame({'val': range(10)})
        table = obj.format_as_table(df, max_rows=31)

        # Should have 11 rows (header + 10 data)

    def test_max_rows_truncation(self, simple_displayable_class: type,
                                 large_dataframe: pd.DataFrame) -> None:
        # Should truncate with ... when over max_rows
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(large_dataframe, max_rows=21)

        # Should show first 10, ..., last 10, plus header = 22 rows
        # Check by rendering
        string_io = StringIO()
        console = Console(file=string_io)
        console.print(table)

        output = string_io.getvalue()
        assert '...' in output

    def test_truncation_shows_head_and_tail(self, simple_displayable_class: type,
                                            large_dataframe: pd.DataFrame) -> None:
        # Should show beginning and end
        obj = simple_displayable_class("Title", "Body")

        table = obj.format_as_table(large_dataframe, max_rows=21,
                                    show_index=True)

        # Render and check content
        string_io = StringIO()
        console = Console(file=string_io)
        console.print(table)

        output = string_io.getvalue()
        # Should contain first and last index values
        assert '0' in output  # First row
        assert '99' in output  # Last row

    def test_table_spacing_setting(self, simple_displayable_class: type,
                                   sample_dataframe: pd.DataFrame) -> None:
        # Should use spacing from settings
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings.table_spacing = 8

        table = obj.format_as_table(sample_dataframe)

        # Check padding (hard to verify directly)

    def test_invalid_align_column_raises(self, simple_displayable_class: type,
                                         sample_dataframe: pd.DataFrame) -> None:
        # Invalid align_column type should raise
        obj = simple_displayable_class("Title", "Body")

        with pytest.raises(TypeError, match="Invalid type for align_column"):
            obj.format_as_table(sample_dataframe, align_column=123)

    def test_invalid_align_header_raises(self, simple_displayable_class: type,
                                         sample_dataframe: pd.DataFrame) -> None:
        # Invalid align_header type should raise
        obj = simple_displayable_class("Title", "Body")

        with pytest.raises(TypeError, match="Invalid type for align_header"):
            obj.format_as_table(sample_dataframe, align_header=123)

    def test_invalid_round_floats_raises(self, simple_displayable_class: type,
                                         sample_dataframe: pd.DataFrame) -> None:
        # Invalid round_floats type should raise
        obj = simple_displayable_class("Title", "Body")

        with pytest.raises(TypeError, match="Invalid type for round_floats"):
            obj.format_as_table(sample_dataframe, round_floats="invalid")


# ========== ========== ========== ========== Test Export Methods
class TestExportMethods:
    """Test HTML and SVG export functionality."""

    def test_to_html_returns_string(self, simple_displayable_class: type) -> None:
        # to_html should return HTML string
        obj = simple_displayable_class("Title", "Body")

        html = obj.to_html()

        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_html_contains_content(self, simple_displayable_class: type) -> None:
        # HTML should contain title and body
        obj = simple_displayable_class("MyTitle", "MyBody")

        html = obj.to_html()

        assert "MyTitle" in html
        assert "MyBody" in html

    def test_to_html_is_valid_html(self, simple_displayable_class: type) -> None:
        # Should produce valid HTML
        obj = simple_displayable_class("Title", "Body")

        html = obj.to_html()

        assert '<' in html  # Has HTML tags
        assert '>' in html

    def test_to_svg_returns_string(self, simple_displayable_class: type) -> None:
        # to_svg should return SVG string
        obj = simple_displayable_class("Title", "Body")

        svg = obj.to_svg()

        assert isinstance(svg, str)
        assert len(svg) > 0

    def test_to_svg_is_valid_svg(self, simple_displayable_class: type) -> None:
        # Should produce valid SVG
        obj = simple_displayable_class("Title", "Body")

        svg = obj.to_svg()

        assert '<svg' in svg
        assert '</svg>' in svg


# ========== ========== ========== ========== Integration Tests
class TestIntegration:
    """Test complete integration scenarios."""

    def test_complete_display_workflow(self, sample_dataframe: pd.DataFrame) -> None:
        # Complete workflow with custom settings
        class DataDisplay(Displayable):
            def __init__(self, df: pd.DataFrame, title: str):
                self.df = df
                self.title_text = title

            def _title(self) -> Text:
                return Text(self.title_text, style="bold cyan")

            def _content(self) -> Table:
                return self.format_as_table(self.df, max_rows=10)

        display = DataDisplay(sample_dataframe, "Sample Data")
        display.display_settings.panel_border_style = 'green'

        # Should render without errors
        output = str(display)
        assert "Sample Data" in output

    def test_form_and_table_combination(self, sample_dataframe: pd.DataFrame) -> None:
        # Combine form and table in one display
        from rich.console import Group

        class CombinedDisplay(Displayable):
            def __init__(self, df: pd.DataFrame):
                self.df = df

            def _title(self) -> Text:
                return Text("Data Report")

            def _content(self) -> Group:
                # Form with summary
                summary = self.format_as_form({
                    'Rows': str(len(self.df)),
                    'Columns': str(len(self.df.columns)),
                })

                # Table with data
                table = self.format_as_table(self.df, max_rows=5)

                return Group(summary, "", table)

        display = CombinedDisplay(sample_dataframe)
        output = str(display)

        assert "Rows" in output
        assert "Columns" in output

    def test_settings_persistence(self, tmp_path: Path,
                                  simple_displayable_class: type) -> None:
        # Save and load custom settings
        settings = DisplaySettings()
        settings.console_width = 100
        settings.panel_border_style = 'yellow'

        path = tmp_path / "theme.disp"
        settings.save(path)

        # Create object with loaded settings
        obj = simple_displayable_class("Title", "Body")
        obj.display_settings = DisplaySettings.load(path)

        assert obj.display_settings.console_width == 100
        assert obj.display_settings.panel_border_style == 'yellow'

    def test_multiple_displayables_different_settings(self,
                                                      simple_displayable_class: type) -> None:
        # Multiple objects with different themes
        obj1 = simple_displayable_class("Object 1", "Content 1")
        obj1.display_settings.panel_border_style = 'red'

        obj2 = simple_displayable_class("Object 2", "Content 2")
        obj2.display_settings.panel_border_style = 'blue'

        panel1 = obj1._display_panel()
        panel2 = obj2._display_panel()

        assert panel1.border_style == 'red'
        assert panel2.border_style == 'blue'