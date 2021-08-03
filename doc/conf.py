# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('./mock'))

# -- Project information -----------------------------------------------------

project = 'QSynthesis'
copyright = '2021, Quarkslab'
author = 'Robin David'

# The full version, including alpha/beta/rc tags
release = '0.1'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = 'en'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'  # also monokai, friendly, colorful

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'breathe',
    'sphinx.ext.intersphinx',
    'sphinx_fontawesome'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['figs']

autodoc_default_flags = ['members', 'inherited-members']


autoclass_content = "both"  # Comment class with both class docstring and __init__ docstring

autodoc_typehints = "signature"

autodoc_type_aliases = {
    'Addr': 'qsynthesis.types.Addr',
    'BitSize': 'qsynthesis.types.BitSize',
    'ByteSize': 'qsynthesis.types.ByteSize',
    'Hash': 'qsynthesis.types.Hash',
    'Char': 'qsynthesis.types.Char',
    'Input': 'qsynthesis.types.Input',
    'Output': 'qsynthesis.types.Output',
    'IOPair': 'qsynthesis.types.IOPair',
    'IOVector': 'qsynthesis.types.IOVector'
}


intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}



# ==============================================================
#                     LATEX CONFIGURATION
# ==============================================================

report_title = "QSynthesis: Documentation"
report_reference = u'21-01-781-REP'
report_version = release
report_filename = 'qsynthesis.tex'
report_date = datetime.date.today().strftime("%d %B %Y")

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', report_filename, report_title,
     u'QB', 'manual'),
]

# Additional stuff for the LaTeX preamble.
custom_latex_preamble = r'''
    \newcommand{\myref}{%s}
    \newcommand{\myversion}{%s}
    \newcommand{\mydate}{%s}
    \usepackage{float}
    \usepackage{xcolor}
    \setlength{\parskip}{5pt}
    \usepackage{multicol}
    \usepackage{tikz}
    \usepackage{fontawesome}
    \definecolor{LightCyan}{rgb}{0.88, 1.0, 1.0}
''' % (report_reference, report_version, report_date)

latex_additional_files = ['latexstyling.sty']
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'fontpkg': '\\usepackage{lmodern}',
    'babel': '\\usepackage[english]{babel}',  # choix du langage : le langage utilise doit etre le dernier de la liste
    'fncychap': '',  # don't redefine chapter commands
    'preamble': '\\usepackage{latexstyling}' + custom_latex_preamble,
    # remove blank pages and same margin on odd and even pages
    'classoptions': ',oneside',
    'figure_align': 'H',
    'maketitle': '\\maketitle',
    'passoptionstopackages': r'\PassOptionsToPackage{svgnames}{xcolor}',
    'sphinxsetup': 'warningborder=1pt, '  # also: cautious, attention, danger, error
                   'warningBorderColor={rgb}{1.0, 0.51, 0.26}, '  # (http://latexcolor.com)
                   'warningBgColor={rgb}{1.0, 0.8, 0.64}, '  # 

                   'noteborder=2pt, '  # also: hint, important, tip
                   'noteBorderColor={rgb}{0.57, 0.63, 0.81}, '
}
