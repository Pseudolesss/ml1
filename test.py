import numpy as np
import data

class test(object):
	"""docstring for test"""
	def m(self):
		self.a = 3
		self.b = 2

t = test()
t.m()
print(t.a)


(x,y) = data.make_dataset1(10)

print("x: ", x.shape, "  y: ", y.shape)

for rows in x:
	print()
	
	for values in rows:
		print(values)

print()

print(len(y))
for values in y:
	print(values)

print()
print(len(x))

cov = np.cov(x, None, False)
print(cov)

print(2**-1*2)

print(np.dot([2,3], [5,6]))
print(np.dot(y, y))

ba = []
ba.append(2)
ba.append(3)
print(ba[0]+ba[1])
si=[1,2]
si.append(0)
print(si[2])


arg = np.zeros(6)
print(arg)

bar = zip([1,2,3],[4,5,6])
print(bar[0])