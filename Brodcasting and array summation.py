
import numpy as np
#array summation
a = np.array([1, 2, 3])
b = 2
print(a + b)
# Broadcasting
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1], [2]])
print(a + b)

#use np.sum() to sum all the elements in an array
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))

#use keepdims=True to keep the dimensions of the array
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(a, axis=0, keepdims=True))
print(np.sum(a, axis=1, keepdims=True))
