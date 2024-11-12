import numpy as np

def prod(A,b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - np.dot(A[i][i+1:],x[i+1:]))/A[i][i]
    return x

def random_matrix(n):
    A = np.random.randn(n,n)

    for k in range(0,n-1):
        for i in range(k+1,n):
            # Calculeaza multiplicatorii Gaussieni
            A[i][k] = -A[i][k]/A[k][k]   # Se suprascriu in triunghiul inferior
            # Aplica multiplicatorii
            for j in range(k+1,n):
                A[i][j] = A[i][j] + A[k][j]*A[i][k]
    A_gauss = np.triu(A)
    b = np.random.randn(n)
    x = prod(A_gauss, b)
    return x

print("Rezultatul pentru matrice random de 6")
print(random_matrix(6))

def exemplu_lab():
    A = [[2,4,-2], [4, 9, -3],[-2, -3, 7]]
    b = [2,8,10]
    A = np.array(A)
    n = 3
    for k in range(0,n-1):
        for i in range(k+1,n):
            # Calculeaza multiplicatorii Gaussieni
            A[i][k] = -A[i][k]/A[k][k]   # Se suprascriu in triunghiul inferior
            # Aplica multiplicatorii
            for j in range(k+1,n):
                A[i][j] = A[i][j] + A[k][j]*A[i][k]

    b = np.array(b)
    for k in range(0,n-1):
        for i in range(k+1,n):
            b[i] = b[i] + A[i][k]*b[k]


    A = np.triu(A)
    # print(A)
    # print(b)
    x = prod(A,b)
    return x

print("Exemplu lab")
print(exemplu_lab())