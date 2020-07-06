Welcome to the unit tests for fivecentplots.  These tests do a pixel-
by-pixel value comparison of fresh images against a master image file.
Because of very slight differences between platforms and different
versions of matplotlib, the tests may not function properly except for:

* windows 10
* matplotlib versions 2.2 or 3.2

This needs to be sorted out at some point

Installs required for running tests:

* pytest
* imageio