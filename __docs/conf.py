#  -*- coding: utf-8 -*-
"""
refer to:
    https://www.sphinx-doc.org/en/master/usage/configuration.html#module-conf
"""

import os
import sys


# ========== ========== ========== ========== ========== Project Information
"""
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""
project = 'Jangada'
copyright = '2025, Rafael R. L. Benevides, J. M. da Silva'
author = 'Rafael R. L. Benevides, J. M. da Silva'
version = '0.0.0-dev'
release = version


# ========== ========== ========== ========== ========== General configuration
"""
https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
"""
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'numpydoc',
    'sphinx_design',
    'sphinx_copybutton',
    # 'sphinx_gallery.gen_gallery'
]


# ========== ========== ========== ========== ========== HTML
"""
https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html
"""
html_static_path = ['_static']
html_css_files = ['style.css']
html_favicon = '_static/logo/jangada.png'
html_title = f'{project} {release}'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'logo': {
        'text': 'Jangada',
        'image_light': '_static/logo/jangada.png',
        'image_dark': '_static/logo/jangada.png',
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jangada-dev/jangada",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/jangada",
            "icon": 'fa-solid fa-box',
            "type": "fontawesome",
        },
    ],
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'navbar_align': 'content',  # 'content'
    'primary_sidebar_end': ['indices.html', 'sidebar-ethical-ads.html'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],

    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#hide-the-previous-and-next-buttons
    # "show_prev_next": False,

    "footer_start": ["sphinx-version"],
    "footer_center": ["copyright"],
    "footer_end": ["theme-version"]
}
html_context = {
    "default_mode": "auto"
}


# ========== ========== ========== ========== ========== autodoc
"""
https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
"""
sys.path.insert(0, os.path.abspath('..'))


# ========== ========== ========== ========== ========== autodoc
"""
https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
"""
sys.path.insert(0, os.path.abspath('..'))

autoclass_content = 'class'
autodoc_class_signature = "mixed"
autodoc_default_options = {
    'show-inheritance': True,
    'members': False,
    'member_order': 'bysource',
}
autodoc_typehints = 'signature'  # 'signature', 'description', 'none', 'both'
autodoc_type_aliases = {
    'NumericType': 'NumericType',
    'ArrayLike': 'ArrayLike',
    'NDArray': 'NDArray',
    'Unit': 'Unit',
    'QuantityLike': 'QuantityLike',
    '_ArrayLikeNumber_co | Quantity': 'QuantityLike'
    # 'SubType' : 'smet.utils.Serialisable.SubType',
    # 'SerialisableType': 'smet.utils.Serialisable.SerialisableType',
    # 'SType': 'smet.utils.Serialisable.SType'
}
autodoc_typehints_format = 'short'
autodoc_typehints_description_target = 'documented_params'


def autodoc_skip_member(app, what, name, obj, skip, options):
    """https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-skip-member"""

    exclusions = [
        '__weakref__',  # special-members
        '__doc__',
        '__module__',
        '__dict__',  # undoc-members
        # '__init__',
        '__new__',
        '__setattr__',
        '__getattr__',
    ]

    exclude = name in exclusions
    return skip or exclude


def autodoc_process_bases(app, name, obj, options, bases):
    """https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-bases"""
    return bases[:-1]


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    """https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-signature"""

    replaces = [
        ('numbers.Number | numpy.ndarray[numpy.number] | astropy.units.quantity.Quantity', 'NumericType'),
        ('astropy.units.core.Unit', 'Unit'),
        ('_ArrayLikeNumber_co | Quantity', 'QuantityLike')
    ]

    if signature is not None:
        for item in replaces:
            signature = signature.replace(*item)

    return signature, return_annotation


def autodoc_process_docstring(app, what, name, obj, option, lines):
    """https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring"""

    if what == 'data' and name.endswith('Like'):
        print(name)
        # print(lines)
        lines.pop(-2)
        lines.pop(-1)

        while '' in lines:
            lines.remove('')

        print(obj.__doc__)


# ========== ========== ========== ========== ========== to do
"""
https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration
"""
todo_include_todos = True


# ========== ========== ========== ========== ========== intersphinx
"""
https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration
"""
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}


# ========== ========== ========== ========== ========== Math
"""
https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-math
"""
...


# ========== ========== ========== ========== ========== mathjax
"""
https://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-mathjax3_config
"""
...


# ========== ========== ========== ========== ========== inheritance diagram
"""
https://www.sphinx-doc.org/en/master/usage/extensions/inheritance.html#configuration
"""
...


# ========== ========== ========== ========== ========== autosummary (automatically loaded by numpydoc)
"""
https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#generating-stub-pages-automatically
"""
...


# ========== ========== ========== ========== ========== numpydoc
"""
https://numpydoc.readthedocs.io/en/latest/install.html#configuration
"""
numpydoc_show_class_members = False
numpydoc_xref_aliases = {'QuantityLike': 'QuantityLike'}
numpydoc_xref_ignore = {'QuantityLike'}


# ========== ========== ========== ========== ========== sphinx_design
"""
https://sphinx-design.readthedocs.io/en/pydata-theme/get_started.html
"""
...


# ========== ========== ========== ========== ========== sphinx_copybutton
"""
https://sphinx-copybutton.readthedocs.io/en/latest/
"""

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = True
copybutton_copy_empty_lines = True
copybutton_remove_prompts = True


# ========== ========== ========== ========== ========== Extension setup
"""
conf.py becomes itself an extension when an setup function is added to it

https://www.sphinx-doc.org/en/master/extdev/appapi.html#extension-setup
"""


def setup(app):
    # app.add_js_file('copybutton.js')
    # app.add_css_file('toggle.css')
    app.connect('autodoc-skip-member', autodoc_skip_member)
    app.connect('autodoc-process-bases', autodoc_process_bases)
    app.connect('autodoc-process-docstring', autodoc_process_docstring)
    app.connect('autodoc-process-signature', autodoc_process_signature)
    # app.connect('autodoc-before-process-signature', autodoc_before_process_signature)