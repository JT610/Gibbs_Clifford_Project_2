#########################################################################################
# IMPORTS                                                                               #
#########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#########################################################################################
# TASK A                                                                                #
#########################################################################################

"""
Visualization of the electric field around two conducting spheres
using the Method of Image Charges.
"""
# System parameters

eps0 = 8.854e-12          # vacuum permittivity
R = 0.5                   # sphere radius
d = 5.0                   # separation (center to center), must be > 2R
Q = 1e-6                  # charge magnitude

MAX_ITERS = 200
TOL = 1e-12

# Image charge generator

def generate_image_charges(R, d, Q, max_iters=MAX_ITERS, tol=TOL):
    """
    Returns:
        pos1, q1  (arrays of image charge positions and magnitudes in sphere 1)
        pos2, q2  (arrays of image charge positions and magnitudes in sphere 2)
    """
    pos1 = [0.0]
    q1   = [Q]

    pos2 = [d]
    q2   = [-Q]

    for _ in range(max_iters):

        # Last charges
        q1_last, x1_last = q1[-1], pos1[-1]
        q2_last, x2_last = q2[-1], pos2[-1]

        # Distances from centers
        r1 = abs(x2_last - 0.0)
        r2 = abs(d - x1_last)

        # New image inside sphere 1
        q1_new = -q2_last * R / r1
        x1_new = R**2 / r1

        # New image inside sphere 2
        q2_new = -q1_last * R / r2
        x2_new = d - R**2 / r2

        if abs(q1_new) < tol and abs(q2_new) < tol:
            break

        q1.append(q1_new)
        pos1.append(x1_new)

        q2.append(q2_new)
        pos2.append(x2_new)

    return np.array(pos1), np.array(q1), np.array(pos2), np.array(q2)


# Compute E-field at (x,y)

def E_field(x, y, pos1, q1, pos2, q2):
    """
    Computes the electric field vector (Ex, Ey)
    created by all image charges.
    """
    Ex, Ey = 0.0, 0.0

    # Charges in sphere 1
    for xc, q in zip(pos1, q1):
        dx = x - xc
        dy = y - 0.0
        r2 = dx*dx + dy*dy + 1e-12
        Ex += q * dx / r2**1.5
        Ey += q * dy / r2**1.5

    # Charges in sphere 2
    for xc, q in zip(pos2, q2):
        dx = x - xc
        dy = y - 0.0
        r2 = dx*dx + dy*dy + 1e-12
        Ex += q * dx / r2**1.5
        Ey += q * dy / r2**1.5

    k = 1 / (4 * np.pi * eps0)
    return k * Ex, k * Ey


# generate plot

def task_a():

    # 1. Generate image charges
    pos1, q1, pos2, q2 = generate_image_charges(R, d, Q)

    print("Generated", len(q1), "image charges per sphere")

    # 2. Grid for E-field plotting
    gx = np.linspace(-2, d + 2, 250)
    gy = np.linspace(-4, 4, 250)
    X, Y = np.meshgrid(gx, gy)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)

    print("Computing electric field over grid...")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Ex[i,j], Ey[i,j] = E_field(X[i,j], Y[i,j], pos1, q1, pos2, q2)

    # Field magnitude for coloring
    Emag = np.log(np.sqrt(Ex**2 + Ey**2))

    # 3. Plotting
    plt.figure(figsize=(10,6))
    plt.streamplot(X, Y, Ex, Ey, color=Emag, cmap='inferno', density=1.2)

    # Draw spheres
    sph1 = plt.Circle((0,0), R, color='red', fill=False, linewidth=2)
    sph2 = plt.Circle((d,0), R, color='blue', fill=False, linewidth=2)
    plt.gca().add_patch(sph1)
    plt.gca().add_patch(sph2)

    plt.title("Electric Field of Two Conducting Spheres (Image Charge Method)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig("e_field_spheres.png", dpi=200)
    plt.show()

#########################################################################################
# TASK B                                                                                #
#########################################################################################

# Solutions for electrostatic boundary value problems in 2D domains
# Case 1-3 solved analytically, Case 4 solved numerically


# Cases 1-3: 

# --- Domain checks ---
# Case 1: Unit disk (r < 1)
def in_domain1(x, y):
    return x**2 + y**2 < 1.0
	
# Case 2: Annulus (1 < r < 2)
def in_domain2(x, y):
    r = np.sqrt(x**2 + y**2)
    return (r > 1.0) & (r < 2.0)

# Case 3: Ellipse (x^2/13^2 + y^2/5^2 < 1)
def in_domain3(x, y):
    return x**2 / 13.0**2 + y**2 / 5.0**2 < 1.0


# --- Exact solutions ---
# Case 1: Laplace's equation with homogeneous Dirichlet data on the unit disk has unique solution phi = 0
def phi_case1(x, y):
    return 0.0

# Case 2: Laplace's equation with phi = 1 on inner circle (r=1) and phi = pi on outer circle (r=2) has unique solution phi(r) = (((pi - 1) / log(2)) * log(r)) + 1
def phi_case2(x, y):
    r = np.sqrt(x**2 + y**2)
    A = (np.pi - 1.0) / np.log(2.0)
    return 1.0 + A * np.log(r)
# Case 3: Laplace's equation with homogeneous Dirichlet data on the ellipse has unique solution phi = 0
def phi_case3(x, y):
    return 0.0

# Case 4: rho(x)= exp(-1/(1-|x|^2)) for |x|<1, else 0
def phi_case4(GridX, GridY, mask, rho, psi):
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


def evaluate_phi(case_id, x, y):
    if case_id == 1:
        if not in_domain1(x, y):
            raise ValueError(f"Point ({x}, {y}) is outside the unit disk.")
        return phi_case1(x, y)
    elif case_id == 2:
        if not in_domain2(x, y):
            raise ValueError(f"Point ({x}, {y}) is outside the annulus.")
        return phi_case2(x, y)
    elif case_id == 3:
        if not in_domain3(x, y):
            raise ValueError(f"Point ({x}, {y}) is outside the ellipse.")
        return phi_case3(x, y)
    else:
        raise ValueError(f"Invalid case_id {case_id}. Expected 1, 2, or 3. Case 4 is solved numerically.")


# --- Plotting ---
# helper
def plot_phi(X, Y, phi, title):
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, phi, levels=25, cmap="viridis") # Contour plot of phi
    fig.colorbar(contour, ax=ax, label ="Potential $\phi$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title) 
    ax.set_aspect("equal")
    plt.show()


# plot each case
def plot_case1():
    # Unit disk, phi = 0
    x = np.linspace(-1.1, 1.1, 300)
    y = np.linspace(-1.1, 1.1, 300)
    X, Y = np.meshgrid(x, y)
    mask = in_domain1(X, Y) # if point is in unit disk
    phi = np.full_like(X, np.nan) 
    phi[mask] = phi_case1(X[mask], Y[mask]) # compute phi only for points in the disk
    plot_phi(X, Y, phi, "Case 1: Unit Disk") 


def plot_case2():
    # Annulus, phi(r) = (((pi - 1) / log(2)) * log(r)) + 1
    x = np.linspace(-2.1, 2.1, 300)
    y = np.linspace(-2.1, 2.1, 300)
    X, Y = np.meshgrid(x, y)
    mask = in_domain2(X, Y)
    phi = np.full_like(X, np.nan)
    phi[mask] = phi_case2(X[mask], Y[mask])
    plot_phi(X, Y, phi, "Case 2: Annulus")


def plot_case3():
    # Ellipse, phi = 0
    x = np.linspace(-13.5, 13.5, 300)
    y = np.linspace(-5.5, 5.5, 300)
    X, Y = np.meshgrid(x, y)
    mask = in_domain3(X, Y)
    phi = np.full_like(X, np.nan)
    phi[mask] = phi_case3(X[mask], Y[mask])
    plot_phi(X, Y, phi, "Case 3: Ellipse")


def plot_case4():
    N=500
    x = np.linspace(-2.5, 2.5, N)
    y = np.linspace(-2.5, 2.5, N)

    X, Y = np.meshgrid(x, y)

    rho = lambda x, y: np.exp(-1/max(0.0001,1-(x**2+y**2))) if (x**2+y**2) < 1 else 0
    psi = lambda x, y: 0

    mask = X==X

    phi = phi_case4(X, Y, mask, rho, psi)

    phi_plot = np.copy(phi)
    phi_plot[~mask] = np.nan

    plt.figure(figsize=(6, 5))
    # Use contourf to plot the filled contours of the potential
    contour = plt.contourf(X, Y, phi_plot, levels=50, cmap='inferno')
    plt.colorbar(contour, label='Potential $\phi$')
    plt.title("Case 4: Square Domain")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal') # Ensures our circle looks like a circle
    plt.show()

# --- Main Case ---
def task_b():
    print("Solutions for electrostatic boundary value problems")
    print("-" * 62)

    plot_case1()
    plot_case2()
    plot_case3()
    plot_case4()

#########################################################################################
# TASK C                                                                                #
#########################################################################################

class Point:
    ### class defining points and their distance
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z])
    
    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.pos - p2.pos)
    
    def dist_to(self, p):
        return self.__class__.dist(self,p)
    
def getForce(p1, plist):
    ### calculates the 'force' caused by repulsion between points
    F = np.array([0,0,0],dtype=float)
    
    for p2 in [p for p in plist if p != p1]:
        # skip if equal location
        if p1.dist_to(p2) != 0:
            # formula, is the same as the 'penalty' towards phi (1/distance), in the direction pointing away from the other point
            F += 1/p1.dist_to(p2) / np.linalg.norm(p1.pos - p2.pos) * (p1.pos - p2.pos)
    
    return F
    
def step(plist, learning_rate= 0.1, randomOffset=0.1):
    ### one optimization step
    newlist = []
    
    for p in plist:
        # new point is previous point plus learning rate * repulsive force and a random offset to get out of unstable equilibria
        new_pos = p.pos + learning_rate * getForce(p, plist) + randomOffset * np.random.normal(0, 0.5, 3)
        # normalize new point to be on the sphere
        new_pos /= np.linalg.norm(new_pos)
        newlist.append(Point(*new_pos))
        
    return newlist


def task_c(N, verbose=False):
    plist = []
    
    Nsteps = 20000//N
    
    # add N points
    for i in range(N):
        plist.append(Point(0,0,0))
    
    # do 'Nsteps' optimization steps, scale learning rate and random offset down to slowly settle in found equilibrium
    for i in range(Nsteps):
        if verbose:
            print('Iteration ', i+1, '/', Nsteps, ' started')
        plist = step(plist, 10**(-i/Nsteps), 10**(- 5 * i/Nsteps))
    
    # calculate final Phi
    phi = 0
    for i in range(N):
        for j in range(i):
            dist = Point.dist(plist[i], plist[j])
            phi += 1/dist
    
    if verbose:
        print('Calculated Phi for ', N, ' points: ', phi)
    
    
    # setup 3d plot
    plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Solution for {N} points: $\\phi={phi:.2f}$")
    
    # plot points
    x = [p.pos[0] for p in plist]
    y = [p.pos[1] for p in plist]
    z = [p.pos[2] for p in plist]
    ax.scatter(x,y,z, color='blue')
    
    # plot unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color='cyan')
    
    # show plot
    plt.show()
    
    return

#########################################################################################
# ENTRY POINT                                                                           #
#########################################################################################

if __name__ == "__main__":
    task_a()
    
    task_b()
    
    to_simulate = [2, 3, 4, 5, 6, 8, 12, 20, 32, 300]
    for N in to_simulate:
        task_c(N, verbose=True)