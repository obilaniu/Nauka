# -*- coding: utf-8 -*-
import numbers, math


class LR(numbers.Real):
	def __new__      (cls, *args, **kwargs):
		lr = super().__new__(cls)
		lr._stepNum = 0
		return lr
	def __float__    (self):    return 1.
	def __bool__     (self):    return bool(float(self))
	def __abs__      (self):    return abs(float(self))
	def __add__      (self, x): return float(self) +  x
	def __ceil__     (self):    return math.ceil (float(self))
	def __divmod__   (self, x): return divmod(float(self), x)
	def __eq__       (self, x): return float(self) == x
	def __floor__    (self):    return math.floor(float(self))
	def __floordiv__ (self, x): return float(self) // x
	def __format__   (self, f): return format(float(self), f)
	def __ge__       (self, x): return float(self) >= x
	def __gt__       (self, x): return float(self) >  x
	def __le__       (self, x): return float(self) <= x
	def __lt__       (self, x): return float(self) <  x
	def __mod__      (self, x): return float(self) %  x
	def __mul__      (self, x): return float(self) *  x
	def __neg__      (self):    return -float(self)
	def __pos__      (self):    return +float(self)
	def __pow__      (self, x): return pow(float(self), x)
	def __radd__     (self, x): return x +  float(self)
	def __rdivmod__  (self, x): return divmod(x, float(self))
	def __repr__     (self):    return "LR()"
	def __rfloordiv__(self, x): return x // float(self)
	def __rmod__     (self, x): return x %  float(self)
	def __rmul__     (self, x): return x *  float(self)
	def __round__    (self, n): return round(float(self), n)
	def __rpow__     (self, x): return pow(x, float(self))
	def __rsub__     (self, x): return x -  float(self)
	def __rtruediv__ (self, x): return x /  float(self)
	def __str__      (self):    return repr(self) + " @ "+str(self.stepNum)
	def __sub__      (self, x): return float(self) -  x
	def __truediv__  (self, x): return float(self) /  x
	def __trunc__    (self):    return math.trunc(float(self))
	
	def step         (self, *, memo=None):
		if memo is None:
			memo = set() # Recursive descent begins now.
		if id(self) not in memo:
			memo.add(id(self))
			self._step()
			for child in self.children:
				if isinstance(child, LR):
					child.step(memo=memo)
	
	def _step        (self):
		self._stepNum += 1
	
	@property
	def stepNum      (self):    return self._stepNum
	@property
	def children     (self):    return []


class ConstantLR(LR):
	def __init__     (self, lr=1.0):
		self._lr = float(lr)
	def __float__    (self):
		return self._lr
	def __repr__     (self):
		return "ConstantLR(lr={})".format(repr(self._lr))


class LambdaLR(LR):
	def __init__     (self, lrLambda=lambda stepNum: 1.0):
		self._lrLambda = lrLambda
	def __float__    (self):
		return self._lrLambda(self.stepNum)
	def __repr__     (self):
		return "LambdaLR(lrLambda={})".format(repr(self._lrLambda))


class ProdLR(LR):
	def __init__     (self, *children):
		self._children = children
	def __float__    (self):
		product = 1.0
		for lr in self._children:
			product *= lr
		return product
	def __repr__     (self):
		return "ProdLR({})".format(", ".join(
			[repr(child) for child in self._children]
		))
	@property
	def children     (self): return self._children


class ClampLR(LR):
	def __new__      (cls,  lr, lrMin=None, lrMax=None):
		if lr is None:
			raise ValueError("Invalid LR: "+repr(lr))
		if lrMin is None and lrMax is None:
			# Don't bother creating a wrapper ClampLR object if it isn't going
			# to accomplish anything whatsoever.
			if   isinstance(lr, LR):
				return lr
			elif isinstance(lr, numbers.Real):
				return ConstantLR(lr)
			else:
				raise ValueError("Invalid LR: "+repr(lr))
		return super().__new__(cls)
	def __init__     (self, lr, lrMin=None, lrMax=None):
		self._lr    = lr
		self._lrMin = lrMin
		self._lrMax = lrMax
	def __float__    (self):
		lr = float(self._lr)
		lr = lr if self._lrMax is None else min(lr, float(self._lrMax))
		lr = lr if self._lrMin is None else max(lr, float(self._lrMin))
		return lr
	def __repr__     (self):
		return "ClampLR(lr={}, lrMin={}, lrMax={})".format(
		    repr(self._lr),
		    repr(self._lrMin),
		    repr(self._lrMax),
		)
	@property
	def children     (self): return [self._lr]


class StepLR(LR):
	def __init__     (self, stepSize, gamma=0.95):
		self._stepSize = stepSize
		self._gamma    = gamma
	def __float__    (self):
		return self._gamma ** (self.stepNum // self._stepSize)
	def __repr__     (self):
		return "StepLR(stepSize={}, gamma={})".format(
		    repr(self._stepSize),
		    repr(self._gamma)
		)


class ExpLR(LR):
	def __init__     (self, gamma=0.95):
		self._gamma = gamma
	def __float__    (self):
		return self._gamma ** self.stepNum
	def __repr__     (self):
		return "ExpLR(gamma={})".format(repr(self._gamma))


class CosLR(LR):
	def __init__     (self, period, lrA=1.0, lrB=0.0):
		self._period = period
		self._lrA    = lrA
		self._lrB    = lrB
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf = self._period
		modstep = self.stepNum % periodf
		cosine  = math.cos(2*math.pi*modstep/periodf)
		factor  = 0.5*(1.0-cosine)
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "CosLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class SawtoothLR(LR):
	def __init__     (self, period, lrA=1.0, lrB=0.0):
		self._period = period
		self._lrA    = lrA
		self._lrB    = lrB
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf   = self._period
		modstep   = self.stepNum % periodf
		factor    = modstep/(periodf-1.0)
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "SawtoothLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class TriangleLR(LR):
	def __init__     (self, period, lrA=1.0, lrB=4.0):
		self._period = period
		self._lrA    = lrA
		self._lrB    = lrB
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf   = self._period
		modstep   = self.stepNum % periodf
		period1   = periodf//2
		period2   = periodf-period1
		firsthalf = modstep < period1
		periodh   = period1 if firsthalf else period2
		hmodstep  = modstep if firsthalf else periodf-modstep
		factor    = hmodstep/periodh
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "TriangleLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class PlateauLR(LR):
	"""Mostly lifted from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau"""
	def __init__     (self, patience=10, cooldown=0, gamma=0.1, threshold=1e-4,
	                        lrMin=0, eps=1e-8, mode="min", thresholdMode="rel",
	                        verbose=False):
		if gamma >= 1.0:
			raise ValueError("Invalid gamma {} >= 1.0!".format(repr(gamma)))
		if mode          not in {"min", "max"}:
			raise ValueError("Unknown mode={}!".format(repr(mode)))
		if thresholdMode not in {"rel", "abs"}:
			raise ValueError("Unknown thresholdMode={}!".format(repr(thresholdMode)))
		
		self._lr              = 1.0
		self._patience        = patience
		self._cooldown        = cooldown
		self._gamma           = gamma
		self._threshold       = threshold
		self._lrMin           = lrMin
		self._eps             = eps
		self._mode            = mode
		self._thresholdMode   = thresholdMode
		self._verbose         = verbose
		
		self._best            = math.inf if mode=="min" else -math.inf
		self._numBadSteps     = 0
		self._cooldownCounter = 0
	
	def __float__    (self):
		return float(self._lr)
	
	def __repr__     (self):
		return "PlateauLR(patience={}, cooldown={}, gamma={}, threshold={}, " \
		       "lrMin={}, eps={}, mode={}, thresholdMode={}, verbose={})".format(
		    repr(self._patience),
		    repr(self._cooldown),
		    repr(self._gamma),
		    repr(self._threshold),
		    repr(self._lrMin),
		    repr(self._eps),
		    repr(self._mode),
		    repr(self._thresholdMode),
		    repr(self._verbose),
		)
	
	def _step        (self):
		if not hasattr(self, "current"):
			return
		current = float(self.__dict__.pop("current"))
		
		if self._isNewBest(current):
			self._best         = current
			self._numBadSteps  = 0
		else:
			self._numBadSteps += 1
		
		if self._coolingDown:
			self._cooldownCounter -= 1
			self._numBadSteps      = 0
		
		if self._numBadSteps > self._patience:
			self._cooldownCounter = self._cooldown
			self._numBadSteps     = 0
			lrOld = float(self._lr)
			lrNew = float(max(lrOld*self._gamma, self._lrMin))
			if lrOld-lrNew > self._eps:
				self._lr = lrNew
				if self._verbose:
					print("Step {:6d}: Reducing LR from {:.4e} to {:.4e} .".format(
					      self.stepNum, lrOld, lrNew
					))
		
		super()._step()
	
	def _isNewBest(self, current):
		if self._mode=="min":
			if self._thresholdMode=="rel":
				dynamicBest = self._best*(1.-self._threshold)
			else:
				dynamicBest = self._best-self._threshold
			return current < dynamicBest
		else:
			if self._thresholdMode=="rel":
				dynamicBest = self._best*(1.+self._threshold)
			else:
				dynamicBest = self._best+self._threshold
			return current > dynamicBest
	
	@property
	def _coolingDown (self):
		return self._cooldownCounter > 0
	
