import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1,2,3],[4,5,6]])

# Tells proper dimention like 2x2
# print(a.shape)
# print("shape:- ", b.shape)

#Tells number of dimention like 2
# print(a.ndim)
# print(b.ndim)

#Tell totals number of elements like 3 or 6
# print(a.size)
# print(b.size)

#Tells array data type like int64
# print(a.dtype)
# print(b.dtype)

#Generate array with number of 0 elements like [0, 0]
# x = np.zeros(2)
# print(x)

#Generate array with 1 like [1,1,1,1,1]
# print(np.ones(5))

#Generate array with 5 which is 3 elements [5,5,5] 
# print(np.full(3, 5))

#Genereate matrix of rxc like 3x3 of 2d
# print(np.eye(3))

#Generate array with n number with 0 like [0,0,0,0]
# print(np.empty(4))

#Generate array with start, stop, step like [1,3,5,7,9]
# print(np.arange(1, 10, 2))

#Generate array with number of gap with value like [1, 5.5, 10]
# print(np.linspace(1, 10, 3))


# print(np.random.seed(42)) Lock the algorithm that same number generate every time in random number
# print(np.random.rand(3,2)) Genereate 0 to 1 numbers in the 3x2 dimetion

# print(np.random.randint(0, 10, 5)) Generate list of 1 to 10 random numbers with 5 elements

# print(np.random.randn(4)) Generaet 0 to 1 numbers 1 d with 4 elements

# print(np.random.choice([10,20,30], 3))

# print(np.random.shuffle(a))

# print(np.random.permutation(a))

# rang = np.random.default_rng(123)
# print(rang.random((3,2))) Generae 3x2 dimention fromt 0 to 1
# print(rang.integers(0, 10, 5)) Generate list with 5 elements from 0 to 10
# print(rang.normal(size=4)) same as randn
                
# Reshaping

arr = np.arange(12)
# print(arr.reshape(3,4)) #Reshape to 2d dimetion with 3x4
# print(arr.reshape(-1, 6)) Reshpae to the 2d with 2x6 

# print(a.flatten())
# print(np.arange(12).reshape(2, 3, 2).flatten()) Flat the any dimension array to the 1d

# print(np.random.rand(3, 2).transpose()) Transpose col to the row

# print(np.squeeze(arr))


a = np.array([1,2,3])
b = np.array([4,5,6,7])

# print(np.concatenate([a,b], axis=0)) Concate same dimension arrays
# print(np.stack([a,b], axis=0))
# print(np.vstack([a,b])) convert 1d to 2 do

# print(np.split(b, 2)) split into 2 array with same number of elements
# print(np.repeat(b, 4)) Repease eacch elements 4 times in sequence

