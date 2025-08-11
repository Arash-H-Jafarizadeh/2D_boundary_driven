import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



#### ED numerics #####################################################################################################################################

def kron_csr(x,y):
    return sp.sparse.kron(x,y,format='csr')


def checkerboard(dims):
    Lx,Ly = dims
    return [(i+j)%2 for j in range(Ly) for i in range(Lx)]


def random_config_sector(dims, k):
    N = np.prod(dims)
    
    assert 0 <= k <= N, "k must be between 0 and N"
    
    occupation_list = np.zeros(N, dtype=int)
    ones_indices = np.random.choice(N, size=k, replace=False)
    occupation_list[ones_indices] = 1
    
    return occupation_list



def single_shot(value, min, max):
    # sample = np.random.uniform(min, max)
    sample = min + (max-min) * np.random.random_sample()
    if sample < value:
        return max
    else:
        return min



def lattice_edges(N, M):

    edges = []
    for row in range(M):
        for col in range(N):
            idx = row * N + col
            if col < N - 1:  # Connect to right neighbor
                edges.append((row, idx, idx + 1))
            if row < M - 1:  # Connect to top neighbor
                edges.append((0, idx, idx + N))
    return edges



def circuit_edges(Nx, Ny):
    edges = [ [] for _ in range(4)]
    for row in range(Ny):
        for col in range(Nx):
            idx = row * Nx + col
            if col < Nx - 1 and col % 2 == 0.0 :  #Connection to horizontal neighbors
                edges[0].append((idx, idx + 1))
            if col < Nx - 1 and col % 2 == 1.0 : 
                edges[1].append((idx, idx + 1))
            if row < Ny - 1 and row % 2 == 0.0 :  #Connection to vertical neighbors
                edges[2].append((idx, idx + Nx))
            if row < Ny - 1 and row % 2 == 1.0 : 
                edges[3].append((idx, idx + Nx))

    return edges


#### PLOTS ##########################################################################################################################################        

def distinct_colors(n, color_map_name='jet'):
    cmap = plt.get_cmap(color_map_name)
    return [cmap(i / (n - 1)) for i in range(n)]
    
    
def current_from_correlation(input_C, dims, B_field = 0.0):
    Lx, Ly = dims
    L = Lx * Ly

    peier = lambda x: np.exp( 1j * B_field * x )

    J_array = np.zeros((2*L-Lx-Ly, 3), dtype=np.float64)
    for indx, (row, xx, yy) in enumerate(lattice_edges(Lx,Ly)):
        J_ij = -1j*( peier(row) * input_C[xx,yy] - peier(row).conj() * input_C[yy,xx] )  
        J_array[indx] = np.array([int(xx), int(yy), np.real(J_ij) ])

    return(J_array)