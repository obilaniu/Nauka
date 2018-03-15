# -*- coding: utf-8 -*-

#
# Random assorted stuff.
#
# Functions and utilities in this module may ONLY rely on the presence of
# standard Python library packages, plus numpy.
#

import argparse                             as Ap
import ast
import hashlib
import numpy                                as np
import os
import random
import re
import sys
import time
import warnings



def npBlasTest(k=8192, dtype=np.float32):
	x = np.zeros((k,k), dtype=dtype)
	t =- time.time()
	x.dot(x)
	t += time.time()
	return 2*k**3/t

def toBytesUTF8(x, errors="strict"):
	return x.encode("utf-8", errors=errors) if isinstance(x, str) else x

def pbkdf2                   (dkLen, password, salt="", rounds=1, hash="sha256"):
	password = toBytesUTF8(password)
	salt     = toBytesUTF8(salt)
	return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)

def getIntFromPBKDF2         (nbits, password, salt="", rounds=1, hash="sha256", signed=False):
	nbits = int(nbits)
	assert nbits%8 == 0
	dkLen = nbits // 8
	buf   = pbkdf2(dkLen, password, salt, rounds, hash)
	i     = int.from_bytes(buf, "little", signed=signed)
	return i

def getNpRandomStateFromPBKDF2      (password, salt="", rounds=1, hash="sha256"):
	uint32le = np.dtype(np.uint32).newbyteorder("<")
	buf      = pbkdf2(624*4, password, salt, rounds, hash)
	buf      = np.frombuffer(buf, dtype=uint32le).copy("C")
	return ("MT19937", buf, 624, 0, 0.0)

def seedNpRandomFromPBKDF2          (password, salt="", rounds=1, hash="sha256"):
	npRandomState = getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
	np.random.set_state(npRandomState)
	return npRandomState

def getRandomStateFromPBKDF2        (password, salt="", rounds=1, hash="sha256"):
	npRandomState = getNpRandomStateFromPBKDF2(password, salt, rounds, hash)
	twisterState  = tuple(npRandomState[1].tolist()) + (624,)
	return (3, twisterState, None)

def seedRandomFromPBKDF2            (password, salt="", rounds=1, hash="sha256"):
	randomState = getRandomStateFromPBKDF2(password, salt, rounds, hash)
	random.setstate(randomState)
	return randomState

def getTorchRandomSeedFromPBKDF2    (password, salt="", rounds=1, hash="sha256"):
	return getIntFromPBKDF2(64, password, salt, rounds, hash, signed=True)

def seedTorchRandomManualFromPBKDF2 (password, salt="", rounds=1, hash="sha256"):
	import torch
	seed = getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
	torch.random.manual_seed(seed)
	return seed

def seedTorchCudaManualFromPBKDF2   (password, salt="", rounds=1, hash="sha256"):
	import torch
	seed = getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
	torch.cuda.manual_seed(seed)
	return seed

def seedTorchCudaManualAllFromPBKDF2(password, salt="", rounds=1, hash="sha256"):
	import torch
	seed = getTorchRandomSeedFromPBKDF2(password, salt, rounds, hash)
	torch.cuda.manual_seed_all(seed)
	return seed

def getTorchOptimizerFromAction(params, spec, **kwargs):
	from torch.optim import (SGD, RMSprop, Adam,)
	
	if   spec.name in ["sgd", "nag"]:
		return SGD    (params, spec.lr, spec.mom,
		               nesterov=(spec.name=="nag"),
		               **kwargs)
	elif spec.name in ["rmsprop"]:
		return RMSprop(params, spec.lr, spec.rho, spec.eps, **kwargs)
	elif spec.name in ["adam"]:
		return Adam   (params, spec.lr, (spec.beta1, spec.beta2), spec.eps,
		               **kwargs)
	elif spec.name in ["amsgrad"]:
		return Adam   (params, spec.lr, (spec.beta1, spec.beta2), spec.eps,
		               amsgrad=True,
		               **kwargs)
	else:
		raise NotImplementedError("Optimizer "+spec.name+" not implemented!")

class PlainObject(object):
	pass

class OptimizerAction(Ap.Action):
	def __init__(self, **kwargs):
		defaultOpt = kwargs.setdefault("default", "sgd")
		defaultOpt = Ap.Namespace(**OptimizerAction.parseOptSpec(defaultOpt))
		kwargs["default"] = defaultOpt
		kwargs["metavar"] = "OPTSPEC"
		kwargs["nargs"]   = None
		kwargs["type"]    = str
		super().__init__(**kwargs)
	
	def __call__(self, parser, namespace, values, option_string):
		#
		# Create new namespace
		#
		
		setattr(namespace, self.dest, Ap.Namespace())
		ns = getattr(namespace, self.dest)
		ns.__dict__.update(**OptimizerAction.parseOptSpec(values))
	
	@staticmethod
	def parseOptSpec   (values):
		#
		# Split the argument string:
		#
		# --arg optimizername:key0,key1,key2=value0,key3=value1
		#
		split  = values.split(":", 1)
		name   = split[0].strip()
		rest   = split[1] if len(split) == 2 else ""
		args   = []
		kwargs = {}
		
		def carveRest(s, sep):
			quotepairs = {"'": "'", "\"": "\"", "{":"}", "[":"]", "(":")"}
			val   = ""
			quote = ""
			prevC = ""
			for i, c in enumerate(s):
				if quote:
					if   c == quote[-1]  and prevC != "\\":
						val    += c
						prevC   = ""
						quote   = quote[:-1]
					elif c in quotepairs and prevC != "\\":
						val    += c
						prevC   = ""
						quote  += quotepairs[c]
					elif prevC == "\\":
						val     = val[:-1]+c
						prevC   = ""
					else:
						val    += c
						prevC   = c
				else:
					if   c == sep:
						break
					elif c in quotepairs and prevC != "\\":
						val    += c
						prevC   = ""
						quote  += quotepairs[c]
					elif prevC == "\\":
						val     = val[:-1]+c
						prevC   = ""
					else:
						val    += c
						prevC   = c
				
			return val, s[i+1:]
		
		while rest:
			positionalVal, positionalRest = carveRest(rest, ",")
			keywordKey,    keywordRest    = carveRest(rest, "=")
			
			#
			# If the distance to the first "=" (or end-of-string) is STRICTLY
			# shorter than the distance to the first ",", we have found a
			# keyword argument.
			#
			
			if len(keywordKey)<len(positionalVal):
				key       = re.sub("\\s+", "", keywordKey)
				val, rest = carveRest(keywordRest, ",")
				try:    kwargs[key] = ast.literal_eval(val)
				except: kwargs[key] = val
			else:
				if len(kwargs) > 0:
					raise ValueError("Positional optimizer argument \""+r+"\" found after first keyword argument!")
				val, rest = positionalVal, positionalRest
				try:    args += [ast.literal_eval(val)]
				except: args += [val]
		
		#
		# Parse argument string according to optimizer
		#
		if   name in ["sgd"]:
			return OptimizerAction.filterSGD(*args, **kwargs)
		elif name in ["nag"]:
			return OptimizerAction.filterNAG(*args, **kwargs)
		elif name in ["adam"]:
			return OptimizerAction.filterAdam(*args, **kwargs)
		elif name in ["amsgrad"]:
			return OptimizerAction.filterAmsgrad(*args, **kwargs)
		elif name in ["rmsprop"]:
			return OptimizerAction.filterRmsprop(*args, **kwargs)
		elif name in ["yf", "yfin", "yellowfin"]:
			return OptimizerAction.filterYellowfin(*args, **kwargs)
		else:
			raise ValueError("Unknown optimizer \"{}\"!".format(name))
	
	@staticmethod
	def filterSGD      (lr=1e-3, mom=0.9, nesterov=False):
		lr       = float(lr)
		mom      = float(mom)
		nesterov = bool(nesterov)
		
		assert lr    >  0
		assert mom   >= 0 and mom   < 1
		
		d = locals()
		d["name"] = "sgd"
		return d
	@staticmethod
	def filterNAG      (lr=1e-3, mom=0.9, nesterov=True):
		d = OptimizerAction.filterSGD(lr, mom, nesterov)
		d["name"] = "nag"
		return d
	@staticmethod
	def filterAdam     (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		lr    = float(lr)
		beta1 = float(beta1)
		beta2 = float(beta2)
		eps   = float(eps)
		
		assert lr    >  0
		assert beta1 >= 0 and beta1 < 1
		assert beta2 >= 0 and beta2 < 1
		assert eps   >= 0
		
		d = locals()
		d["name"] = "adam"
		return d
	@staticmethod
	def filterAmsgrad  (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		d = OptimizerAction.filterAdam(lr, beta1, beta2, eps)
		d["name"] = "amsgrad"
		return d
	@staticmethod
	def filterRmsprop  (lr=1e-3, rho=0.9, eps=1e-8):
		lr    = float(lr)
		rho   = float(rho)
		eps   = float(eps)
		
		assert lr    >  0
		assert rho   >= 0 and rho   < 1
		assert eps   >= 0
		
		d = locals()
		d["name"] = "rmsprop"
		return d
	@staticmethod
	def filterYellowfin(lr=1.0, mom=0.0, beta=0.999, curvWW=20, nesterov=False):
		lr       = float(lr)
		mom      = float(mom)
		beta     = float(beta)
		curvWW   = int(curvWW)
		nesterov = bool(nesterov)
		
		assert lr     >  0
		assert mom    >= 0 and mom  <  1
		assert beta   >= 0 and beta <= 1
		assert curvWW >= 3
		
		d = locals()
		d["name"] = "yellowfin"
		return d


class CudaDeviceAction(Ap.Action):
	def __init__(self, **kwargs):
		#
		# If a CudaDeviceAction flag is given, but without argument device IDs,
		# the default interpretation is "want CUDA, best guess at devices". The
		# default interpretation can be changed by providing the `const`
		# argument of argparse.Action.
		#
		if "const" not in kwargs:
			#
			# The default interpretation depends on the environment variable
			# CUDA_VISIBLE_DEVICES.
			#
			if "CUDA_VISIBLE_DEVICES" in os.environ:
				#
				# The environment has CUDA_VISIBLE_DEVICES set. We assume that
				# variable was wisely set, so we craft an ordered list of
				# integers as long as the variable indicates there are devices.
				#
				CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
				CUDA_VISIBLE_DEVICES = self.parseString(CUDA_VISIBLE_DEVICES)
				CUDA_VISIBLE_DEVICES = list(range(len(CUDA_VISIBLE_DEVICES)))
			else:
				#
				# The environment does not have CUDA_VISIBLE_DEVICES set. We'll
				# make a best guess that the user has and wants the first GPU.
				#
				CUDA_VISIBLE_DEVICES = [0]
		else:
			CUDA_VISIBLE_DEVICES = kwargs["const"]
		
		kwargs.setdefault("metavar", "DEVIDS")
		kwargs.setdefault("const",   CUDA_VISIBLE_DEVICES)
		kwargs["nargs"]   = "?"
		kwargs["default"] = []
		kwargs["type"]    = str
		kwargs.setdefault("help",    "List of CUDA device ids to use.")
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		#
		# If values is a list, it's probably because it's the const= list.
		# If values is a string, we parse it.
		#
		if isinstance(values, list):
			setattr(ns, self.dest, values)
		else:
			setattr(ns, self.dest, self.parseString(values))
	
	@classmethod
	def parseString(kls, dIdCommaSepString):
		#
		# Parse the device ID string.
		# 
		# We assume a comma-separated list of integers and hyphenated integer
		# ranges. Example:
		# 
		#           0,3-4,2 ,   1,   6-7
		#
		
		deviceIds = []
		
		for dIdRange in dIdCommaSepString.split(","):
			dIdRange      = dIdRange.strip()
			dIdRangeSplit = dIdRange.split("-")
			
			if   not dIdRange:
				continue
			elif len(dIdRangeSplit) == 0:
				pass
			elif len(dIdRangeSplit) == 1 and dIdRangeSplit[0]:
					deviceIds.append(int(dIdRangeSplit[0].strip()))
			elif len(dIdRangeSplit) == 2 and dIdRangeSplit[0] and dIdRangeSplit[1]:
				start = int(dIdRangeSplit[0].strip())
				end   = int(dIdRangeSplit[1].strip())
				step  = +1 if start <= end else -1
				deviceIds.extend(range(start, end+step, step))
			else:
				raise ValueError(
				    "Broken CUDA device range specification \"{}\"".format(dIdRange)
				)
		
		return deviceIds

class PresetAction(Ap.Action):
	def __init__(self, **kwargs):
		if kwargs.get("default", Ap.SUPPRESS) != Ap.SUPPRESS:
			warnings.warn("A PresetAction ignores the default= keyword argument!")
		
		if "choices" not in kwargs:
			raise ValueError("Must provide a dictionary of presets!")
		else:
			self.presets = kwargs["choices"]
		if not isinstance(self.presets, dict):
			raise TypeError("Preset choices must be a dictionary!")
		
		for presetName, preset in self.presets.items():
			if not isinstance(presetName, str):
				raise TypeError("Preset names must be of type \"str\", but "
				                "one of them is not!")
			if not isinstance(preset,     list):
				raise TypeError("Presets must be a list of strings, but "
				                "one of them is not!")
		
		kwargs["default"] = Ap.SUPPRESS
		kwargs["choices"] = list(self.presets.keys())
		kwargs["nargs"]   = None
		kwargs["type"]    = str
		
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		parser.parse_args(self.presets[values], ns)

class DirPathAction(Ap.Action):
	ENVVAR     = None
	DEFDEFVAL  = None
	HELPSTRING = None
	
	def __init__(self, **kwargs):
		"""
		For DirPathActions, the argument-handling logic is as follows:
		
		  - If arg explicitly given, use that.
		  - If arg not given but ENVVAR set, use that.
		  - If arg not given and ENVVAR unset, but default set, use that.
		  - If arg not given,    ENVVAR unset  and default unset, use DEFDEFVAL.
		"""
		defaultNoEnv = kwargs.get("default", self.DEFDEFVAL)
		default = os.environ.get(self.ENVVAR, defaultNoEnv)
		kwargs["default"] = default
		kwargs["nargs"]   = None
		kwargs.setdefault("type", str)
		kwargs.setdefault("help", self.HELPSTRING)
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		setattr(ns, self.dest, values)

class BaseDirPathAction(DirPathAction):
	ENVVAR     = "NAUKA_BASEDIR"
	DEFDEFVAL  = "work"
	HELPSTRING = "Path to the base directory from which this experiment's true " \
	             "working directory will be derived."

class DataDirPathAction(DirPathAction):
	ENVVAR     = "NAUKA_DATADIR"
	DEFDEFVAL  = "data"
	HELPSTRING = "Path to the datasets directory."

class TmpDirPathAction(DirPathAction):
	ENVVAR     = "NAUKA_TMPDIR"
	DEFDEFVAL  = "tmp"
	HELPSTRING = "Path to a local, fast-storage, temporary directory."

class Subcommand(object):
	cmdname       = None
	parserArgs    = {}
	subparserArgs = {}
	
	@classmethod
	def name(kls):
		return kls.cmdname or kls.__name__
	
	@classmethod
	def subcommands(kls):
		for s in kls.__dict__.values():
			if isinstance(s, type) and issubclass(s, Subcommand):
				yield s
	
	@classmethod
	def addToSubparser(kls, subp):
		argp = subp.add_parser(kls.name(), **kls.parserArgs)
		return kls.addAllArgs(argp)
	
	@classmethod
	def addArgs(kls, argp):
		pass
	
	@classmethod
	def addAllArgs(kls, argp=None):
		argp        = Ap.ArgumentParser(**kls.parserArgs) if argp is None else argp
		subcommands = list(kls.subcommands())
		if subcommands:
			subp = argp.add_subparsers(**kls.subparserArgs)
			for s in subcommands:
				s.addToSubparser(subp)
		argp.set_defaults(
		    __argp__ = argp,
		    __kls__  = kls,
		)
		kls.addArgs(argp)
		return argp
	
	@classmethod
	def run(kls, a, *args, **kwargs):
		a.__argp__.print_help()
		return 0

