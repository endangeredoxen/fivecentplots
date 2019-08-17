import fivecentplots
import os

# Move to this directory (to check git repo)
build_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(build_dir)

meta_text = f"""
package:
  name: fivecentplots
  version: {fivecentplots.__version__}

source:
  path: {build_dir}

build:
  noarch: python
  script: python setup.py install --single-version-externally-managed --record=record.txt

requirements:
  host:
    - python
    - setuptools
  run:
    - python
    - bokeh >=1.3.1
    - matplotlib >=3.1.0
    - pandas >=0.25.0
    - pywin32 >=223
    - scipy >=1.3.0
    - xlrd >=1.2.0

test:
  imports:
    - fivecentplots

about:
  home: https://github.com/endangeredoxen/fivecentplots
  license: GPLv3
  summary: "Custom plotting wrapper for matplotlib"
  description: |
    Custom plotting wrapper for matplotlib
  dev_url: https://github.com/endangeredoxen/fivecentplots
  doc_url: https://github.com/endangeredoxen/fivecentplots
  doc_source_url: https://github.com/endangeredoxen/fivecentplots/blob/master/README.rst
"""

with open("meta.yaml", "w") as fp:
    fp.write(meta_text)

os.system(f"conda build {build_dir}")
