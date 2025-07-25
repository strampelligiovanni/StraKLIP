# -*- coding: utf-8 -*-
#
# spaceKLIP documentation build configuration file, created by
# sphinx-quickstart on Wed Jul 6, 2022.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# fix for `ImportError: No module named _tkinter`:
import matplotlib

matplotlib.use("agg")

#### Get version
with open(Path(__file__).parent.parent.parent / "pyproject.toml", "rb") as metadata_file:
    configuration = tomllib.load(metadata_file)
    metadata = configuration["project"]
    project = metadata["name"]

    # The short X.Y version.
    try:
        version = project.__version__.split("-", 1)[0]
        # The full version, including alpha/beta/rc tags.
        release = project.__version__
    except AttributeError:
        version = "dev"
        release = "dev"

# -- General configuration ------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    # 'sphinx_automodapi.automodapi',
    'nbsphinx'
    # 'sphinx.ext.githubpages',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []  # ['./_templates']

# # mock imports for autodoc
# autodoc_mock_imports = ["webbpsf", "webbpsf_ext"]

# nbsphinx settings
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
# nbsphinx_prolog = """
# {% set docname = env.doc2path(env.docname, base=None) %}
# .. note::  `Download the full notebook for this tutorial here <https://github.com/kammerje/spaceKLIP/tree/develop/docs/source/{{ docname }}>`_
# """

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'
automodapi_inherit_docstrings = True

# General information about the project.
project = u'StraKLIP'
copyright = u'Giovanni M. Strampelli'
author = u'Giovanni M. Strampelli'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = version
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#
# today = ''
#
# Else, today_fmt is used as the format for a strftime call.
#
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
#
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# Supress "nonlocal image URI found"
suppress_warnings = ['image.nonlocal_uri']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
# html_theme = 'sphinx_book_theme'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'bizstyle'
# html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    ### for alabaster theme
    # 'logo': 'logo.png',
    # 'logo_name': True,
    # 'font_family': 'Georgia',
    # 'github_banner': True,
    # 'github_repo': 'vortex-exoplanet/VIP',
    # 'github_count': True,
    # 'github_type': 'star',
    # 'fixed_sidebar': True,
    # 'analytics_id': 'UA-84473187-1',

    ### for classic theme
    # 'stickysidebar': True,
    # 'sidebarbgcolor': 'Gray',
    # 'footerbgcolor': 'Gray',
    # 'relbarbgcolor': 'Gray'

    ### for bizstyle theme
    # 'rightsidebar': True

    ### for sphinx_book theme
    # 'path_to_docs': 'docs/source',
    # 'repository_url': 'https://github.com/vortex-exoplanets/VIP',
    # 'repository_branch': 'main',
    # 'launch_buttons': {
    #     'binderhub_url': 'https://mybinder.org/',#'v2/gh/vortex-exoplanet/VIP_extras/master',
    #     'notebook_interface': 'jupyterlab'
    # },
    # 'use_edit_page_button': True,
    # 'use_issues_button': True,
    # 'use_repository_button': True,
    # 'use_download_button': True,
    'logo_only': True,
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#
html_title = u'StraKLIP'

# A shorter title for the navigation bar.  Default is the same as html_title.
#
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = '_static/StraKLIPlogo2.png'


# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
def setup(app):
    app.add_css_file('wider_docs.css')


html_static_path = ['./_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#
# html_extra_path = []

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
#
# html_last_updated_fmt = None

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#
# html_sidebars = {
#     'index':    ['mysidebar.html', 'localtoc.html', 'searchbox.html'],
#     '**':       ['mysidebar.html', 'localtoc.html', 'relations.html',
#                  'searchbox.html']
# }


# Additional templates that should be rendered to pages, maps page names to
# template names.
#
# html_additional_pages = {}

# If false, no module index is generated.
#
# html_domain_indices = True

# If false, no index is generated.
#
# html_use_index = True

# If true, the index is split into individual pages for each letter.
#
# html_split_index = False

# If true, links to the reST sources are added to the pages.
#
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr', 'zh'
#
html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# 'ja' uses this config value.
# 'zh' user can custom change `jieba` dictionary path.
#
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'StraKLIPdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'StraKLIP.tex', u'StraKLIP Documentation',
     u'Giovanni M. Strampelli', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#
# latex_use_parts = False

# If true, show page references after internal links.
#
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
#
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
#
# latex_appendices = []

# It false, will not define \strong, \code, 	itleref, \crossref ... but only
# \sphinxstrong, ..., \sphinxtitleref, ... To help avoid clash with user added
# packages.
#
# latex_keep_old_macro_names = True

# If false, no module index is generated.
#
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'StraKLIP', u'StraKLIP Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'StraKLIP', u'StraKLIP Documentation',
     author, 'StraKLIP', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#
# texinfo_appendices = []

# If false, no module index is generated.
#
# texinfo_domain_indices = True

# How to jy URL addresses: 'footnote', 'no', or 'inline'.
#
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#
# texinfo_no_detailmenu = False

html_context = {'display_github': True,
                'github_user': 'strampelligiovanni',
                'github_repo': 'StraKLIP',
                'github_version': 'main/docs/'}
