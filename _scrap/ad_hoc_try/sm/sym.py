
import re


class Operator(object):
	pass


class MatrixDot(Operator):
	def __init__(self, args):
		self._args = args


class Symbol(object):
	RE_STR = re.compile("[a-zA-Z][a-zA-Z0-9]*")

	def __init__(self, s, shape=None):
		assert not Symbol.RE_STR.match(s) is None, \
			"String representation should match `{}`, but got {}".format(Symbol.RE_STR.pattern, s)
		self._str = s
		self._shape = shape

	def __repr__(self):
		if not self._shape is None:
			return "Symbol({}, {})".format(
				self._str, 
				" x ".join([
					s.name for s in self._shape
				])
			)

		else:
			return "Symbol({})".format(self._str)


	@property
	def name(self):
		return self._str

	@property
	def T(self):
		assert not self._shape is None
		return Symbol(self.name, reversed(self._shape))
		
	@property
	def is_matrix(self):
		return not self._shape is None and len(self._shape) == 2
	
	

	def __mul__(self, other):
		if self.is_matrix and other.is_matrix:




def symbols(s):
	for sym in s.split(" "):
		yield Symbol(sym.strip())



n, m, T = symbols("n m T")

X = Symbol("X", shape=(n, T))
Y = Symbol("Y", shape=(m, T))


X.T * X