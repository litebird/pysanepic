import pysanepic

project = "pysanepic"
author = "Luca Pagano, Guillaume Patanchon"
copyright = "2026, " + author
release = version = pysanepic.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
# the notebooks need litebird_sim and minutes of CPU: shown as saved
nb_execution_mode = "off"
exclude_patterns = ["_build", "notebooks/lbs_example_output", "**/.ipynb_checkpoints"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_use_ivar = True

html_theme = "furo"
html_title = f"pysanepic {release}"
myst_heading_anchors = 3
