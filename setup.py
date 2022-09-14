"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

# Get the long description from the README file
with open(path.join('README.rst')) as f:
    long_description = f.read()

# get the version
with open(path.join('fivecentplots', 'version.txt'), 'r') as input:
    __version__ = input.readlines()[0]

setup(
    name='fivecentplots',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='Wrapper for popular python plotting packages to make visually appealing plots easier',
    long_description=long_description,
    long_description_content_type="text/markdown",

    # The project's main homepage.
    url='https://github.com/endangeredoxen/fivecentplots',
    download_url='https://github.com/endangeredoxen/fivecentplots/archive/v0.4.4.tar.gz',

    # Author details
    author='Steve Nicholes',
    author_email='endangeredoxen@users.noreply.github.com',

    # Choose your license
    license='GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',

    # What does your project relate to?
    keywords='data',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', ]),

    dependency_links=['https://github.com/endangeredoxen/fileio/zipball/master'],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas',
                      'numpy',
                      'matplotlib<=3.5.3',
                      'scipy',
                      'xlrd',
                      'openpyxl',
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
                    'test': ['pytest', 'opencv-python', 'imageio']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'fivecentplots': ['themes/gray.py', 'themes/white.py', 'tests/*', 'doc/_static/docstrings/*', 'keywords.xlsx', 'version.txt'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('C:/my_data', ['pywebify/config.ini'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
