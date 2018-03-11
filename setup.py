# -*- coding: utf-8 -*-

#
# Imports
#
import os, sys, subprocess, time
from   setuptools import setup, find_packages, Extension


#
# Versioning
#

def getVersionInfo():
	naukaVerMajor  = 0
	naukaVerMinor  = 0
	naukaVerPatch  = 2
	naukaVerIsRel  = True
	naukaVerPreRel = "dev0"
	
	naukaVerShort  = "{naukaVerMajor}.{naukaVerMinor}.{naukaVerPatch}".format(**locals())
	naukaVerNormal = naukaVerShort
	naukaVerGit    = getGitVer()
	naukaEpochTime = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
	naukaISO8601   = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(naukaEpochTime))
	if naukaVerIsRel or not naukaVerGit:
		naukaVerSemVer = naukaVerNormal
		naukaVerFull   = naukaVerNormal
	else:
		naukaVerSemVer = naukaVerNormal+"-"+naukaVerPreRel+"+"+naukaISO8601+".g"+naukaVerGit
		naukaVerFull   = naukaVerNormal+".dev0+"+naukaVerGit
	
	return locals()

def getRoot():
	return os.path.dirname(__file__) or "."

def getGitVer():
	cwd = getRoot()
	if not os.path.isdir(os.path.join(cwd, ".git")):
		return ""
	
	env = os.environ.copy()
	env['LANGUAGE'] = env['LANG'] = env['LC_ALL'] = 'C'
	
	try:
		naukaVerGit = subprocess.Popen(
		    ["git", "rev-parse", "HEAD"],
		    stdout = subprocess.PIPE,
		    stderr = subprocess.PIPE,
		    cwd    = cwd,
		    env    = env
		).communicate()[0].strip().decode("ascii")
	except OSError:
		naukaVerGit = ""
	
	if naukaVerGit == "HEAD":
		naukaVerGit = ""
	
	return naukaVerGit

def writeVersionFile(f, versionInfo):
	f.write("""\
#
# THIS FILE IS GENERATED FROM NAUKA SETUP.PY
#
short_version = "{naukaVerShort}"
version       = "{naukaVerNormal}"
full_version  = "{naukaVerFull}"
git_revision  = "{naukaVerGit}"
sem_revision  = "{naukaVerSemVer}"
release       = {naukaVerIsRel}
build_time    = "{naukaISO8601}"
""".format(**versionInfo))

def getDownloadURL(v):
	return "https://github.com/obilaniu/Nauka/archive/v{}.tar.gz".format(v)




if __name__ == "__main__":
	#
	# Defend against Python2
	#
	if sys.version_info[0] < 3:
		sys.stdout.write("This package is Python 3+ only!\n")
		sys.exit(1)
	
	#
	# Handle version information
	#
	versionInfo = getVersionInfo()
	naukaVerFull = versionInfo["naukaVerFull"]
	with open(os.path.join(getRoot(), "src", "nauka", "version.py"), "w") as f:
		writeVersionFile(f, versionInfo)
	
	
	#
	# Setup
	#
	setup(
	    #
	    # The basics
	    #
	    name                 = "nauka",
	    version              = naukaVerFull,
	    author               = "Olexa Bilaniuk",
	    author_email         = "anonymous@anonymous.com",
	    license              = "MIT",
	    url                  = "https://github.com/obilaniu/Nauka",
	    download_url         = getDownloadURL(naukaVerFull),
	    
	    #
	    # Descriptions
	    #
	    description          = ("A collection of utilities for scientific experiments."),
	    
	    long_description     =
	    """\
Nauka is a collection of utilities for scientific experiments.

The name "Nauka" is a rough transliteration of the Ukrainian word "Наука",
meaning "science".""",
	    
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
	    
	    # Sources
	    packages             = find_packages("src"),
	    package_dir          = {'': 'src'},
	    python_requires      = '>=3.4',
	    install_requires     = [
	        "numpy>=1.10",
	    ],
	    
	    # Misc
	    zip_safe             = False,
	)

