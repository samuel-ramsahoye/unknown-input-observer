import numpy as np
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from numpy.linalg import svd
from numpy.linalg import  pinv
from scipy.signal import place_poles
from control import acker
import matplotlib.pyplot as plt
from  scipy.signal import dlsim, dlti


A  = np.array([[1,0,0],[0,1,1],[0,0,1]])

B = np.array([[0,1],[1,1],[1,0]])

C = np.array([[0,1,-1],[1,0,0]])

D = np.array([[0,1],[0,1]])

L = 4

m = B.shape[1]

p = C.shape[0]

#  define J_L

def compute_J_L(A, B, C, D, L):

    J_L = np.zeros((D.shape[0]*(L+1), D.shape[1]*(L+1)))

    for i in range(L+1):

        for j in range(i+1):

            if i == j:

                J_L[i*D.shape[0]:(i+1)*D.shape[0], 
                j*D.shape[1]:(j+1)*D.shape[1]] = D

            elif j == i - 1:

                J_L[i*D.shape[0]:(i+1)*D.shape[0], 
                j*D.shape[1]:(j+1)*D.shape[1]] = C.dot(B)

            else:

                J_L[i*D.shape[0]:(i+1)*D.shape[0], 
                j*D.shape[1]:(j+1)*D.shape[1]] = C.dot(matrix_power(A, i-1-j)).dot(B)

    return J_L

J_Lp1 = compute_J_L(A, B, C, D, L + 1)

J_L = compute_J_L(A, B, C, D, L)

J_Lm1 =compute_J_L(A, B, C, D, L - 1)

assert (matrix_rank(J_Lp1) - matrix_rank(J_L)) == m

# print(J_L)
N_ = null_space(J_Lm1.T).T

def get_matrix_1(A, B, C, D, L):

    matrix_1 = np.zeros((D.shape[0]*(L+1), D.shape[1]))

    matrix_1[:D.shape[0]] = D

    matrix_1[D.shape[0]:2*D.shape[0]] = C.dot(B)

    for i in range(2, L+1):

        matrix_1[i*D.shape[0]:(i+1)*D.shape[0]] = C.dot(matrix_power(A, i-1)).dot(B)

    return matrix_1

def get_O_L(A, C, L):

    O_L = np.zeros((C.shape[0]*(L+1), C.shape[1]))

    O_L[:C.shape[0]] = C

    for i in range(1, L+1):

        O_L[i*C.shape[0]:(i+1)*C.shape[0]] =  C.dot(matrix_power(A, i))

    return O_L

matrix_2 = np.zeros((p + N_.shape[0],p + N_.shape[1] ))

# N_  = np.array(([[-1,1,0,0],[0,0,-1,1]]))

matrix_2[:p,:p] = np.eye(p)

matrix_2[p:,p:] = N_

matrix_1 = get_matrix_1(A, B, C, D, L)

matrix_3 =  matrix_2.dot(matrix_1)

W  = np.vstack((null_space(matrix_3.T).T, pinv(matrix_3)))

N = W.dot(matrix_2)

O_L =  get_O_L(A, C, L)

# N = np.array([[-1,1,0,0,0,0],[0,0,-1,1,0,0],[0,0,0,0,1,-1],[1,0,0,0,0,0]])

S  = N.dot(O_L)

S_1 = S[:-m]

S_2 = S[-m:]

E_ = A - B.dot(S_2)


F_1 = np.zeros((A.shape[0], S_1.shape[0]))

E = E_ - F_1.dot(S_1)

F = np.zeros((F_1.shape[0], F_1.shape[1]+ B.shape[1]))

F[:,:F_1.shape[1]] = F_1

F[:,F_1.shape[1]:] = B

F = F.dot(N)

G = np.linalg.pinv(np.vstack((B, D)))

# =============================================================================
# Simulation
# =============================================================================
u = np.random.rand(100, m)

t = np.arange(100)

x_estimate = []

x_estimate.append(np.array([-3,-1,-2]))

system = dlti(A, B, C, D)

system_t, system_y, system_x = dlsim(system, u, t, x0  = [1,1,1])

u_estimate =  []

for i in range(t.size-L):
    
    x_estimate.append(E.dot(x_estimate[i]) + F.dot(system_y[i:i+L+1].ravel()))
    
    u_estimate.append(G.dot(np.append(x_estimate[i+1]- A.dot(x_estimate[i]), system_y[i] - C.dot(x_estimate[i]))))
    
x_estimate  = np.array(x_estimate)

u_estimate = np.array(u_estimate)

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].plot(system_t,  system_x[:,0], label = "truth")

ax[0].plot(system_t[:-(L-1)],  x_estimate[:,0], label = "estimate")

ax[0].legend()

ax[1].plot(system_t,  system_x[:,1], label = "truth")

ax[1].plot(system_t[:-(L-1)],  x_estimate[:,1], label = "estimate")

ax[1].legend()

ax[2].plot(system_t,  system_x[:,2], label = "truth")

ax[2].plot(system_t[:-(L-1)],  x_estimate[:,2], label = "estimate")

ax[2].legend()

plt.suptitle("state")

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(system_t,  u[:,0], label = "truth")

ax[0].plot(system_t[:-L],  u_estimate[:,0], label = "estimate")

ax[0].legend()

ax[1].plot(system_t,  u[:,1], label = "truth")

ax[1].plot(system_t[:-L],  u_estimate[:,1], label = "estimate")

ax[1].legend()


plt.suptitle("input")












    

