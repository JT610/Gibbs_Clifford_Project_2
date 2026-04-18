import numpy as np
import matplotlib.pyplot as plt

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


def main():
    plist = []
    N = 128
    Nsteps = 20000//N
    
    # add N points
    for i in range(N):
        plist.append(Point(0,0,0))
    
    # do 'Nsteps' optimization steps, scale learning rate and random offset down to slowly settle in found equilibrium
    for i in range(Nsteps):
        print('Iteration ', i+1, '/', Nsteps, ' started')
        plist = step(plist, 10**(-i/Nsteps), 10**(- 5 * i/Nsteps))
    
    # calculate final Phi
    phi = 0
    for i in range(N):
        for j in range(i):
            dist = Point.dist(plist[i], plist[j])
            phi += 1/dist
    
    print('Calculated Phi for ', N, ' points: ', phi)
    
    
    # setup 3d plot
    plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    
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


if __name__ == "__main__":
    main()