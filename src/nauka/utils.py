# -*- coding: utf-8 -*-

#
# Random assorted stuff.
#
# Functions and utilities in this module may ONLY rely on the presence of
# standard Python library packages, plus numpy.
#

import hashlib
import numpy                                as np
import random
import time



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

