DisCoWebS
============
Diffsky joint modelling of Cosmos-Web and SDSS data

Installation
------------
To install DisCoWebS into your environment from the source code::

    $ cd /path/to/root/DisCoWebS
    $ pip install .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/DisCoWebS
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

