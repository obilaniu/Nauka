#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, subprocess, time
from   . import git

#
# Primary version information.
#

# int:  Major version.
verMajor  = 0
# int:  Minor version.
verMinor  = 0
# int:  Patch version.
verPatch  = 6
# bool: Whether this is a release or not.
verIsRel  = True
# str:  Additional suffix for pre-release versions. Empty for releases.
verPreRel = ""

#
# Computed version information.
#
# We should, but probably don't, obey PEP 440:
#     https://www.python.org/dev/peps/pep-0440/
#

# str:  Short-form release version.
verShort  = "{}.{}.{}".format(verMajor, verMinor, verPatch)
# str:  Normal-form release version.
verNormal = verShort
# str:  Full Git SHA-1 tag, or empty string if not Git-controlled.
verGit    = git.getGitVer()
# bool: False if Git-controlled and working directory unclean; True otherwise.
verClean  = (not verGit) or (git.isGitClean())
# int:  POSIX time. Nominal build time as seconds since the Epoch.
posixTime    = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
# str:  ISO8601 timestamp equivalent to POSIX time.
iso8601Time  = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(posixTime))
# str:  SemVer-compatible version number.
if verIsRel or not verGit:
	verSemVer = verNormal
else:
	_cleanString = ".DIRTY"                                if not verClean else ""
	_gitString   = ".g{0}{1}".format(verGit, _cleanString) if     verGit   else ""
	verSemVer    = "{0}-{1}+{2}{3}".format(
	    verNormal,
	    verPreRel,
	    iso8601Time,
	    _gitString,
	)
# str:  Full version number.
if verIsRel or not verGit:
	verFull    = verNormal
else:
	_gitString = "+{0}".format(verGit) if verGit else ""
	verFull    = "{0}.{1}{2}".format(
	    verNormal,
	    verPreRel,
	    _gitString,
	)

#
# Version utilities
#
def synthesizeVersionPy():
	templatePath = os.path.join(git.getSrcRoot(),
	                            "scripts",
	                            "version.py.in")
	
	with open(templatePath, "r") as f:
		return f.read().format(**globals())
