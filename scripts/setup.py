# -*- coding: utf-8 -*-

#
# Imports
#
import os, sys, subprocess, time
from   setuptools import setup, find_packages, Extension

packageName = "nauka"
githubURL   = "https://github.com/obilaniu/Nauka"


#
# Restrict to Python 3.4+
#
if sys.version_info[:2] < (3, 4):
	sys.stdout.write(packageName+" is Python 3.4+ only!\n")
	sys.exit(1)


#
# Retrieve setup scripts
#
from . import git, versioning, utils


#
# Read long description
#
with open(os.path.join(git.getSrcRoot(),
                       "scripts",
                       "LONG_DESCRIPTION.txt"), "r") as f:
	long_description = f.read()


#
# Synthesize version.py file
#
with open(os.path.join(git.getSrcRoot(),
                       "src",
                       packageName,
                       "version.py"), "w") as f:
	f.write(versioning.synthesizeVersionPy())



#
# Perform setup.
#
setup(
    name                 = packageName,
    version              = versioning.verFull,
    author               = "Olexa Bilaniuk",
    author_email         = "anonymous@anonymous.com",
    license              = "MIT",
    url                  = githubURL,
    download_url         = githubURL+"/archive/v{}.tar.gz".format(versioning.verNormal),
    description          = "A collection of utilities for scientific experiments.",
    long_description     = long_description,
    classifiers          = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
    python_requires      = '>=3.4',
    install_requires     = [
        "numpy>=1.10",
    ],
    zip_safe             = False,
)
