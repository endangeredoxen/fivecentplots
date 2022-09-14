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

* Python >=3.6
* pandas >=1.05 (tested up to 1.4.3)
* matplotlib >= 3.1.3 (tested up to 3.5.3; some lower versions will work for some plot types but no guarantees)
* numpy >= 1.13.3 (tested up to 1.23.1)
* scipy >= 1.4.1 (tested up to 1.9.0)
* natsort

Optional
********
* fileio (0.2.2 or higher):  enables pasting of keywords from clipboard from ``ini`` file

.. note:: When running on Linux, you might consider installing additional fonts for better looking labels:

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