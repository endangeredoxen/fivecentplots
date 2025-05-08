Installation
============

Basic
-----

Released versions of **fivecentplots** are hosted on `Pypi <https://pypi.org/project/fivecentplots/>`_ and can be installed via ``pip``:

.. code-block:: python

   pip install fivecentplots

Development versions can be installed via `GitHub <https://github.com/endangeredoxen/fivecentplots>`_


Dependencies
************

* Python >=3.7
* pandas >=1.5
* matplotlib >= 3.1.3
* numpy >= 1.13.3
* scipy >= 1.4.1
* natsort

Font support
************

When running on Linux, you might consider installing additional fonts for better looking labels:

.. code-block::

   sudo apt install -y ttf-mscorefonts-installer
   rm ~/.cache/matplotlib -rf


Test Development
----------------

To run the tests locally, clone the repo, cd to the top `fivecentplots` directory, and run:

.. code-block:: python

   pip install .[test]

or for python 3.6:

.. code-block:: python

   pip install .[test36]

Doc Development
---------------

To build the tests locally, clone the repo, cd to the top `fivecentplots` directory, and run:

.. code-block:: python

   pip install .[doc]

.. note:: Building docs requires ``pandoc``` which must be installed on your machine.  For linux you can run ``sudo apt install pandoc``