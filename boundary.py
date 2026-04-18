import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def boundary_problem_solver(GridX, GridY, mask, rho, psi):
    N = GridX.shape[0]
    h = GridX[0,1]-GridX[0,0]
    A = sp.sparse.lil_matrix((N**2, N**2))
    b = np.zeros(N**2)
    
    for i in range(N):
        for j in range(N):
            x = GridX[i, j]
            y = GridY[i, j]
            id = N*i + j
            
            if not mask[i,j] or i ==0 or i == N-1 or j == 0 or j == N-1:
                A[id, id] = 1
                b[id] = psi(x,y)
            else:
                A[id, id] = -4
                A[id, id-1] = 1
                A[id, id+1] = 1
                A[id, id-N] = 1
                A[id, id+N] = 1
                b[id] = h**2 * rho(x,y)
    A_csr = A.tocsr()
    phi = sp.sparse.linalg.spsolve(A_csr, b)
    phi = phi.reshape((N, N))   
    
    return phi


N=500
x = np.linspace(-2.5, 2.5, N)
y = np.linspace(-2.5, 2.5, N)

X, Y = np.meshgrid(x, y)

rho = lambda x, y: np.exp(1/max(0.0001,1-(x**2+y**2))) if (x**2+y**2) < 1 else 0
psi = lambda x, y: 0

mask = X==X

phi = boundary_problem_solver(X, Y, mask, rho, psi)

phi_plot = np.copy(phi)
phi_plot[~mask] = np.nan

plt.figure(figsize=(6, 5))
# Use contourf to plot the filled contours of the potential
contour = plt.contourf(X, Y, phi_plot, levels=50, cmap='inferno')
plt.colorbar(contour, label='Potential $\phi$')
plt.title("Electrostatic Potential in a Circular Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal') # Ensures our circle looks like a circle
plt.show()
print(phi)