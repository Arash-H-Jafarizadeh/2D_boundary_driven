import numpy as np
import scipy.sparse as sps
import functools as ft
import sys
# import time as tt
# import matplotlib.pyplot as plt
# from numba import njit, prange
# from numba import njit, prange
sys.path.append('/gpfs01/home/ppzaj/python_projects/Quantinuum_Driven_Boundaries/source_code/') 
from basic_functions import *  # type: ignore

###################################################################################################################################################################################
################################################################################## Basic functions ################################################################################
###################################################################################################################################################################################
def kron_csr(x,y):
    return sps.kron(x,y,format='csr')

def op(a,j,L):

    op = sps.csr_array([[0,0],[1,0]]), sps.csr_array([[0,1],[0,0]]) 
        
    op_chain = [sps.csr_array(sps.eye(2**(j) )), op[(1+a)//2], sps.csr_array(sps.eye(2**(L-j-1) ))]
    
    out = ft.reduce(kron_csr, op_chain)
        
    return out


###################################################################################################################################################################################
################################################################################### 2D functions ##################################################################################
###################################################################################################################################################################################

def checkerboard(dims):
    Lx,Ly = dims
    return [(i+j)%2 for j in range(Ly) for i in range(Lx)]


def config_state(lisst):
    
    bits = [0,1], [1,0]
    
    bit_chain = [bits[x] for x in lisst]
    
    out = ft.reduce(np.kron, bit_chain)
    
    return(out) #sps.csr_array(out)



def adj_mat(dims, **kwargs):
    PBC = kwargs.get('PBC', False)
    
    Lx, Ly = dims
    
    Hy = sps.csr_array(np.eye(Ly, k=1)) + sps.csr_array(np.eye(Ly, k=-1))  + int(PBC)*sps.csr_array(np.eye(Ly, k=Ly-1))  + int(PBC)*sps.csr_array(np.eye(Ly, k=-Ly+1))    
    Hx = sps.csr_array(np.eye(Lx, k=1)) + sps.csr_array(np.eye(Lx, k=-1))  + int(PBC)*sps.csr_array(np.eye(Lx, k=Lx-1))  + int(PBC)*sps.csr_array(np.eye(Lx, k=-Lx+1))    
    
    adjs = sps.kronsum(Hx, Hy)
    
    return sps.triu(adjs, format='coo')



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



def Hamiltonian_2d(physical, dims, **kwargs):
    PBC = kwargs.get('PBC', False)
    B_field = kwargs.get('magnetic_field', 0.0)
    
    # print("magnetic field is: ", B_field)
    
    V, J = physical 
    Lx, Ly = dims
    L = Lx*Ly
    peier_phase = lambda x: np.exp( 1j * B_field * x )
    
    # adj_spar =  adj_mat(dims, **kwargs)
    # indexes = list(zip(adj_spar.coords[0],adj_spar.coords[1]))
    # hopping_list = [ sps.csr_array( op(+1,xx,L) @ op(-1,yy,L) + op(+1,yy,L) @ op(-1,xx,L) ) for xx, yy in indexes]
    # interac_list = [ sps.csr_array( op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) ) for xx, yy in indexes]
    
    indexes = lattice_edges(Lx, Ly)
    hopping_list = [ peier_phase(row) * op(+1,xx,L) @ op(-1,yy,L) + peier_phase(row).conj() * op(+1,yy,L) @ op(-1,xx,L) for row, xx, yy in indexes]
    interac_list = [ op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) for _, xx, yy in indexes]
       
    
    hopping_term = ft.reduce(lambda x,y:x+y, hopping_list)
    interac_term = ft.reduce(lambda x,y:x+y, interac_list)
    
    return -J * hopping_term + V * interac_term



def circuit_edges_1(Nx, Ny):
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



def circuit_edges_2(Nx, Ny):
    edges = [ [] for _ in range(4)]
    for row in range(Ny):
        for col in range(Nx):
            idx = row * Nx + col
            if col < Nx - 1 and (col + row) % 2 == 0.0 :  # Connect to right neighbor
                edges[0].append((idx, idx + 1))

            if col < Nx - 1 and (col + row) % 2 == 1.0 :  
                edges[1].append((idx, idx + 1))
                
            if row < Ny - 1 and (row + col) % 2 == 0.0 :  # Connect to top neighbor
                edges[2].append((idx, idx + Nx))

            if row < Ny - 1 and (row + col) % 2 == 1.0 :  
                edges[3].append((idx, idx + Nx))

    return edges




def trotter_circuit_2d(coupl, dims, **KwArgs):
    PBC = KwArgs.get('PBC', False)
    magnetic_field = KwArgs.get('magnetic_field', 0.0)

    V, _ = coupl 
    Lx, Ly = dims
    L = Lx*Ly

    peier_phase = lambda x: np.exp( 1j * magnetic_field * x ) # to be added to the circuit too!

    indx_1,indx_2,indx_3,indx_4 = circuit_edges_2(Lx, Ly)
    
    layer_list_1 = [-op(+1,xx,L) @ op(-1,yy,L) -op(+1,yy,L) @ op(-1,xx,L) + V* op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) for xx, yy in indx_1]
    layer_list_2 = [-op(+1,xx,L) @ op(-1,yy,L) -op(+1,yy,L) @ op(-1,xx,L) + V* op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) for xx, yy in indx_2]
    layer_list_3 = [-op(+1,xx,L) @ op(-1,yy,L) -op(+1,yy,L) @ op(-1,xx,L) + V* op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) for xx, yy in indx_3]
    layer_list_4 = [-op(+1,xx,L) @ op(-1,yy,L) -op(+1,yy,L) @ op(-1,xx,L) + V* op(+1,xx,L) @ op(-1,xx,L) @ op(+1,yy,L) @ op(-1,yy,L) for xx, yy in indx_4]    
    
    layer_term_1 = ft.reduce(lambda x,y:x+y, layer_list_1)
    layer_term_2 = ft.reduce(lambda x,y:x+y, layer_list_2)
    layer_term_3 = ft.reduce(lambda x,y:x+y, layer_list_3)
    layer_term_4 = ft.reduce(lambda x,y:x+y, layer_list_4)
    
    
    return layer_term_1 , layer_term_2 , layer_term_3 , layer_term_4




def kraus_operator(indx, Ls):
    X_0 = op(+1, 0, Ls) + op(-1, 0, Ls) 
    X_L = op(+1, Ls-1, Ls) + op(-1, Ls-1, Ls) 
    
    N_0 = op(+1, 0, Ls) @ op(-1, 0, Ls) 
    N_L = op(+1, Ls-1, Ls) @ op(-1, Ls-1, Ls) 
    
    if indx == 0:
        return sps.csr_array( sps.eye(2**Ls, format='csr'))
    if indx == 1:
        return sps.csr_array( X_0 @ (sps.eye(2**Ls, format='csr') - N_0) )
    if indx == 2:
        return sps.csr_array( N_0 )
    if indx == 3:
        return sps.csr_array( sps.eye(2**Ls, format='csr') - N_L) 
    if indx == 4:
        return sps.csr_array( X_0 @ (sps.eye(2**Ls, format='csr') - N_0) @ (sps.eye(2**Ls, format='csr') - N_L) )
    if indx == 5:
        return sps.csr_array( N_0 @ (sps.eye(2**Ls, format='csr') - N_L) )
    if indx == 6:
        return sps.csr_array( X_L @ N_L )
    if indx == 7:
        return sps.csr_array( X_0 @ (sps.eye(2**Ls, format='csr') - N_0) @ X_L @ N_L )
    if indx == 8:
        return sps.csr_array( N_0 @ X_L @ N_L )



def hraus_operator(indx, Ls):

    N_0 = op(+1, 0, Ls) @ op(-1, 0, Ls) 
    N_L = op(+1, Ls-1, Ls) @ op(-1, Ls-1, Ls) 
    
    if indx == 0:
        return sps.csr_array( sps.eye(2**Ls, format='csr'))
    if indx == 1:
        return sps.csr_array( sps.eye(2**Ls, format='csr') - N_0 )
    if indx == 2:
        return sps.csr_array( N_0 )
    if indx == 3:
        return sps.csr_array( sps.eye(2**Ls, format='csr') - N_L) 
    if indx == 4:
        return sps.csr_array( (sps.eye(2**Ls, format='csr') - N_0) @ (sps.eye(2**Ls, format='csr') - N_L) )
    if indx == 5:
        return sps.csr_array( N_0 @ (sps.eye(2**Ls, format='csr') - N_L) )
    if indx == 6:
        return sps.csr_array( N_L )
    if indx == 7:
        return sps.csr_array( (sps.eye(2**Ls, format='csr') - N_0) @ N_L )
    if indx == 8:
        return sps.csr_array( N_0 @ N_L )


def Ut_dens(dt, e_values, e_vectors):    
    out = e_vectors @ np.diag(np.exp(-1j * e_values * dt)) @ e_vectors.T.conj()
    return sps.csr_matrix(out)



def particle_counts(input_state, dimensions, **kwargs):
    sngl_sht = kwargs.get('single_shot', False)
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    Ns = np.zeros((L,) , dtype=np.float64) 
    for n in range(L):    
        N_th = op(+1,n,L) @ op(-1,n,L) 
        # Ns[n] += 1 - (input_state @ n_th @ input_state.conj() ).real
        n_th = (input_state @ N_th @ input_state.conj() ).real
    
        if sngl_sht:
            Ns[n] += single_shot(n_th, 0.0, 1.0)
        else:
            Ns[n] += n_th
    
    return(Ns)



# def particle_counts(input_state, dimensions, **kwargs):
#     Lx, Ly = dimensions
#     L = Lx * Ly
#     # input_state = input_state.toarray()
#     Ns = np.zeros((L,) , dtype=np.float64) 
#     for n in range(L):    
#         n_th = op(+1,n,L) @ op(-1,n,L) 
#         # Ns[n] += 1 - (input_state @ n_th @ input_state.conj() ).real
#         Ns[n] += np.real(input_state.conj() @ n_th @ input_state )        
#     return(Ns) #np.reshape(Ns, (Ly, Lx)) )



# def resolved_currents(input_state, dimensions, **kwargs):
#     # shaped = kwargs.get('shaped', True)    
#     Lx, Ly = dimensions
#     L = Lx * Ly
    
#     J_mat = np.zeros((L, L) , dtype=np.float64) 
#     for x in range(L):    
#         for y in range(L):    
#             J_ij = -1j*( op(+1,x,L) @ op(-1,y,L) - op(+1,y,L) @ op(-1,x,L) ) 
#             J_mat[x,y] += ( input_state @ J_ij @ input_state.conj() ).real
        
#     return( J_mat )


def resolved_currents_dict(input_state, dimensions, **kwargs):
    magnetic_field = kwargs.get('magnetic_field', 0.0)
    sngl_sht = kwargs.get('single_shot', False)

    Lx, Ly = dimensions
    L = Lx*Ly

    peier = lambda x: np.exp( 1j * magnetic_field * x )

    indexes = lattice_edges(Lx, Ly)

    J_array = np.empty((2*L-Lx-Ly, 3)) 
    for indx, (row, x, y) in enumerate(indexes):        
        var1 = -1j* peier(row)* input_state.conj() @ op(+1,x,L) @ op(-1,y,L) @ input_state
        var2 = +1j* peier(-row)* input_state.conj() @ op(+1,y,L) @ op(-1,x,L) @ input_state
        
        if sngl_sht:
            s_var_1 = single_shot( np.real(var1), -1.0, 1.0)
            s_var_2 = single_shot( np.real(var2), -1.0, 1.0)
            J_ij = s_var_1 + s_var_2 
        else:
            J_ij = var1 + var2 
        
        J_array[indx] = np.array([int(x), int(y), np.real( J_ij )])
        
    return( J_array )


# def resolved_currents_dict(input_state, dimensions, **kwargs):
#     B_field = kwargs.get('magnetic_field', 0.0)    
#     Lx, Ly = dimensions
#     L = Lx * Ly
#     peier = lambda x: np.exp( 1j * B_field * x )     
#     indexes = lattice_edges(Lx, Ly)    
#     J_array = np.empty((2*L-Lx-Ly, 3), dtype=np.float64) # [] 
#     for indx, (row, x, y) in enumerate(indexes):        
#         # J_ij = -1j*( peier(row)* op(+1,x,L) @ op(-1,y,L) - peier(row).conj()*op(+1,y,L) @ op(-1,x,L) ) 
#         # J_array[indx] = np.array([int(x), int(y), np.real( input_state.conj() @ J_ij @ input_state )])
#         J_1 = input_state.conj() @ op(+1,x,L) @ op(-1,y,L) @ input_state 
#         J_2 = input_state.conj() @ op(+1,y,L) @ op(-1,x,L) @ input_state 
#         J_ij = -1j*( peier(row) * J_1 - peier(row).conj() * J_2 ) 
#         J_array[indx] = np.array([int(x), int(y), np.real( J_ij )])
#     return( J_array )



def hopping_correlation(input_state, dimensions):
    
    Lx, Ly = dimensions
    L = Lx * Ly
        
    C_Mat = np.zeros((L,L) , dtype=np.complex128) 
    for x in range(L):    
        for y in range(L):    
            C_xy = op(+1,x,L) @ op(-1,y,L) 
            C_Mat[x,y] += input_state.conj() @ C_xy @ input_state 
        
    return(C_Mat)



def density_correlation(input_state, dimensions):
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    output = np.zeros((L,L) , dtype=np.float64) 
    for y in range(L):    
        for x in range(L):    
            N_X = input_state.conj() @ op(+1,x,L) @ op(-1,x,L) @ input_state
            N_Y = input_state.conj() @ op(+1,y,L) @ op(-1,y,L) @ input_state
            N_N = input_state.conj() @ op(+1,x,L) @ op(-1,x,L) @ op(+1,y,L) @ op(-1,y,L) @ input_state
            output[x,y] += np.real( N_N  - N_X *  N_Y )
        
    return(output)




# def sector_indexes_2D(dims, EVs):
    
#     Lx, Ly = dims
#     sctr_indxs = [[] for _ in range(Lx*Ly+1)]
#     for n in range(2**(Lx*Ly)):
#         pcns = particle_counts(EVs[:,n], Lx*Ly)
#         tpcns = np.round(np.sum(pcns), decimals=0)
#         sctr_indxs[int(tpcns)].append(n)
        
#     return sctr_indxs


# def total_particle_counts(dims):
    
#     Lx, Ly = dims
#     L = Lx * Ly
#     out = np.zeros((2**L,2**L), dtype=np.complex128)  
#     for n in range(1,L+1):    
#         out += op(+1,n,L) @ op(-1,n,L)     
    
#     return out.real



def sparse_simulation(max_time_steps, system_dimensions, physical_couplings, **Kwargs):

    ini_config = Kwargs.get('initial_state', np.resize( [0,1], np.prod(system_dimensions)) )#
    krs_function = Kwargs.get('kraus_function', kraus_operator)    
    K_rate = Kwargs.get('driving_rate', 0.5)#
    dt = Kwargs.get('dt', 0.03)#
    
    Lx, Ly = system_dimensions#    
    L = Lx * Ly#   
    
    ham = Hamiltonian_2d( physical_couplings, system_dimensions, **Kwargs)#

    N_0 = op(+1, 0, L) @ op(-1, 0, L)# 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L)# 

    ini_state = config_state( ini_config )#

    output_C = np.empty((max_time_steps, L, L), dtype=np.complex128)#    
    output_NN = np.empty((max_time_steps, L, L), dtype=np.float64)#    
    kraus_indexes = np.empty((max_time_steps,), dtype=np.int8)#

    for step in range(max_time_steps):
        
        obsr_C = hopping_correlation(ini_state, (Lx, Ly))
        output_C[step] = obsr_C         
        obsr_NN = density_correlation(ini_state, (Lx, Ly))
        output_NN[step] = obsr_NN         
        
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham, ini_state) #type:ignore #U_dt @ ini_state

        n0 = (ini_state.conj() @ N_0 @ ini_state ).real 
        nL = (ini_state.conj() @ N_L @ ini_state ).real 
        n0nL = (ini_state.conj() @ N_0 @ N_L @ ini_state ).real
        
        krs_probs = [(1-K_rate)**2, K_rate*(1-K_rate)*(1-n0), K_rate*(1-K_rate)*n0,
                    K_rate*(1-K_rate)*(1-nL), K_rate*K_rate*(1-n0-nL+n0nL), K_rate*K_rate*(n0-n0nL),
                    K_rate*(1-K_rate)*nL, K_rate*K_rate*(nL-n0nL), K_rate*n0nL ]
        
        krs_rates = np.cumsum(krs_probs)
        
        coin = np.random.uniform(0,1)  
                
        index = len(krs_rates[krs_rates < coin])
        ini_state = krs_function(index, L) @ ini_state
        
        norm = np.linalg.norm(ini_state) #np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
        
        ini_state = 1/norm * ini_state
        
        kraus_indexes[step] = index

    return(output_C, output_NN, kraus_indexes) 




def circuit_simulation(max_time_steps, system_dimensions, physical_couplings, **Kwargs):
    # function = Kwargs.get('observable', particle_counts)
    ini_config = Kwargs.get('initial_state', np.resize( [0,1], np.prod(system_dimensions)) )
    krs_function_type = Kwargs.get('kraus_function_type', "injecting")    
    # krs_function = Kwargs.get('kraus_function', kraus_operator)    
    K_rate = Kwargs.get('driving_rate', 0.5)
    dt = Kwargs.get('dt', 0.03) #Kwargs.get('dt', time_list[-1]/len(time_list))
    
    Lx, Ly = system_dimensions    
    L = Lx * Ly   
    
    if krs_function_type == "dephasing":
        krs_function = hraus_operator
    else:    
        krs_function = kraus_operator
    
    
    
    ham1, ham2, ham3, ham4 = trotter_circuit_2d( physical_couplings, system_dimensions, **Kwargs)

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = config_state( ini_config )#.toarray()   

    # output_N = np.empty((max_time_steps, L), dtype=np.float64) #[]
    # output_J = np.empty((max_time_steps, 2*L-Lx-Ly, 3), dtype=np.float64) #[]
    output_C = np.empty((max_time_steps, L, L), dtype=np.complex128) #[]
    output_NN = np.empty((max_time_steps, L, L), dtype=np.float64) #[]
    
    kraus_indexes = np.empty((max_time_steps,), dtype=np.int8) #[]
    for step in range(max_time_steps):
        
        # obsr_N = particle_counts(ini_state, (Lx, Ly), **Kwargs)         # output_N[step] = obsr_N        
        # obsr_J = currents_dict(ini_state, (Lx, Ly), **Kwargs)        # output_J[step] = obsr_J          
        
        obsr_C = hopping_correlation(ini_state, (Lx, Ly))
        output_C[step] = obsr_C
        obsr_NN = density_correlation(ini_state, (Lx, Ly))
        output_NN[step] = obsr_NN        
        
        
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham1, ini_state) #type: ignore
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham2, ini_state) #type: ignore
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham3, ini_state) #type: ignore
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham4, ini_state) #type: ignore


        n0 = (ini_state @ N_0 @ ini_state.conj() ).real 
        nL = (ini_state @ N_L @ ini_state.conj() ).real 
        n0nL = (ini_state @ N_0 @ N_L @ ini_state.conj() ).real
        
        krs_probs = [(1-K_rate)**2, K_rate*(1-K_rate)*(1-n0), K_rate*(1-K_rate)*n0,
                    K_rate*(1-K_rate)*(1-nL), K_rate*K_rate*(1-n0-nL+n0nL), K_rate*K_rate*(n0-n0nL),
                    K_rate*(1-K_rate)*nL, K_rate*K_rate*(nL-n0nL), K_rate*n0nL ]
        
        krs_rates = np.cumsum(krs_probs)
        
        coin = np.random.uniform(0,1)  
                
        index = len(krs_rates[krs_rates < coin])
        ini_state = krs_function(index, L) @ ini_state
        
        norm = np.linalg.norm(ini_state) #np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
        
        ini_state = 1/norm * ini_state
        
        ##### ~ gathering kraus stats
        kraus_indexes[step] = index # kraus_indexes.append(index) #index_set)  
        
    return(output_C, output_NN, kraus_indexes)




def normal_simulation(max_time_steps, dimensions, couplings, **Kwargs):
    # function = Kwargs.get('observable', particle_counts)
    ini_config = Kwargs.get('initial_state', np.resize( [0,1], np.prod(dimensions)) )
    K_rate = Kwargs.get('driving_rate', 0.05)
    dt = Kwargs.get('dt', 0.03)
    
    Lx, Ly = dimensions    
    L = Lx * Ly   
    
    ham = Hamiltonian_2d( couplings, dimensions, **Kwargs)

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = config_state( ini_config )#.toarray()   

    # output_array = []
    output_N = np.empty((max_time_steps, L), dtype=np.float64) #[]
    output_J = np.empty((max_time_steps, 2*L-Lx-Ly, 3), dtype=np.float64) #[]
    
    # output_C = np.empty((max_time_steps, L, L), dtype=np.float64) #[]
    # output_NN = np.empty((max_time_steps, L, L), dtype=np.float64) #[]
    
    kraus_indexes = np.empty((max_time_steps,), dtype=np.int8) #[]
    for step in range(max_time_steps):
        # obsrv = function(ini_state, (Lx, Ly), **Kwargs)
        # output_array.append(obsrv)        
        
        obsr_N = particle_counts(ini_state, (Lx, Ly), **Kwargs)
        output_N[step] = obsr_N        #.append(obsr_N)        
        obsr_J = resolved_currents_dict(ini_state, (Lx, Ly), **Kwargs)
        output_J[step] = obsr_J #.append(obsr_J)        
        
        # obsr_C = correlation(ini_state, (Lx, Ly))
        # output_C[step] = obsr_C        
        # obsr_NN = density_correlation(ini_state, (Lx, Ly))
        # output_NN[step] = obsr_NN #.append(obsr_J)        
        
        ini_state = sps.linalg.expm_multiply(-1j* dt * ham, ini_state) #type:ignore #U_dt @ ini_state


        n0 = (ini_state.conj() @ N_0 @ ini_state ).real 
        nL = (ini_state.conj() @ N_L @ ini_state ).real 
        n0nL = (ini_state.conj() @ N_0 @ N_L @ ini_state ).real
        
        krs_probs = [(1-K_rate)**2, K_rate*(1-K_rate)*(1-n0), K_rate*(1-K_rate)*n0,
                    K_rate*(1-K_rate)*(1-nL), K_rate*K_rate*(1-n0-nL+n0nL), K_rate*K_rate*(n0-n0nL),
                    K_rate*(1-K_rate)*nL, K_rate*K_rate*(nL-n0nL), K_rate*n0nL ]
        
        krs_rates = np.cumsum(krs_probs)
        
        coin = np.random.uniform(0,1)  
                
        index = len(krs_rates[krs_rates < coin])
        ini_state = kraus_operator(index, L, **Kwargs) @ ini_state
        
        norm = np.linalg.norm(ini_state) #np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
        
        ini_state = 1/norm * ini_state
        
        ##### ~ gathering kraus stats
        kraus_indexes[step] = index #.append(index) #index_set)  
        # kraus_probabilities.append(krs_prbb[index])  
        # kraus_norms.append(norm)  
        
    return(output_N, output_J, kraus_indexes)#, output_C, output_NN)




# def hermitian_simulation(max_time_steps, dimensions, couplings, **Kwargs):
#     function = Kwargs.get('observable', particle_counts)
#     ini_config = Kwargs.get('initial_state', np.resize( [0,1], np.prod(dimensions)) )
#     K_rate = Kwargs.get('driving_rate', 0.5)
#     dt = Kwargs.get('dt', 0.03)
    
#     Lx, Ly = dimensions    
#     L = Lx * Ly   
    
#     ham = Hamiltonian_2d( couplings, dimensions, **Kwargs)

#     N_0 = op(+1, 0, L) @ op(-1, 0, L) 
#     N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

#     ini_state = config_state( ini_config )#.toarray()   

#     # output_array = []
#     output_N = np.empty((max_time_steps, L), dtype=np.float64) #[]
#     output_J = np.empty((max_time_steps, 2*L-Lx-Ly, 3), dtype=np.float64) #[]
    
#     kraus_indexes = np.empty((max_time_steps,), dtype=np.int8) #[]

#     for step in range(max_time_steps):
        
#         # obsrv = function(ini_state, (Lx, Ly), **Kwargs)
#         # output_array.append(obsrv)        
        
#         obsr_N = particle_counts(ini_state, (Lx, Ly), **Kwargs)
#         output_N[step] = obsr_N      #append(obsr_N)        
#         obsr_J = resolved_currents_dict(ini_state, (Lx, Ly), **Kwargs)
#         output_J[step] = obsr_J      #append(obsr_J)        
        
#         ini_state = sps.linalg.expm_multiply(-1j* dt * ham, ini_state) #type: ignore  #U_dt @ ini_state

        
#         n0 = (ini_state @ N_0 @ ini_state.conj() ).real 
#         nL = (ini_state @ N_L @ ini_state.conj() ).real 
#         n0nL = (ini_state @ N_0 @ N_L @ ini_state.conj() ).real
        
#         krs_probs = [(1-K_rate)**2, K_rate*(1-K_rate)*(1-n0), K_rate*(1-K_rate)*n0,
#                     K_rate*(1-K_rate)*(1-nL), K_rate*K_rate*(1-n0-nL+n0nL), K_rate*K_rate*(n0-n0nL),
#                     K_rate*(1-K_rate)*nL, K_rate*K_rate*(nL-n0nL), K_rate*n0nL ]
        
#         krs_rates = np.cumsum(krs_probs)
        
#         coin = np.random.uniform(0,1)  
                
#         index = len(krs_rates[krs_rates < coin])
#         # ini_state = krs_operatos[index] @ ini_state
#         ini_state = hraus_operator(index, L, **Kwargs) @ ini_state
        
#         norm = np.linalg.norm(ini_state) #np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
#         ini_state = 1/norm * ini_state
        
#         ##### ~ gathering kraus stats
#         kraus_indexes[step] = index # kraus_indexes.append(index) #index_set)  
#         # kraus_probabilities.append(krs_prbb[index])  
#         # kraus_norms.append(norm)  
        
#         ##### ~ gathering In/Out flow stats
#         # in_out_flow.append([ particle_In[index], particle_Out[index] ])  

#     return(output_N, output_J, kraus_indexes)











# ########## notes to self (20250516): I dont think it worth it to make a function for time simulation. Doing a manual script may be much faster easier
def run_simulation(time_list, dimensions, couplings, **Kwargs):
    function = Kwargs.get('observable', particle_counts)
    ini_state = Kwargs.get('initial_state', config_state( np.resize( [0,1], np.prod(dimensions)) ))
    K_rate = Kwargs.get('driving_rate', 0.5)
    hermitian_kraus = Kwargs.get('hermitian_kraus', False)
    
    Lx, Ly = dimensions    
    L = Lx * Ly
    
    dt = time_list[-1]/len(time_list)
    
    ham = Hamiltonian_2d( couplings, dimensions, **Kwargs)
    
    U_dt = sps.linalg.expm(-1j * ham *dt) # type: ignore

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = ini_state.toarray()   
    
    # krs_operatos = [np.eye(2**L), op(+1,0,L), np.eye(2**L), np.eye(2**L), op(+1,0,L), np.eye(2**L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), op(-1,L-1,L)]
    # krs_operatos = [np.eye(2**L), op(+1,0,L), N_0, np.eye(2**L) - N_L, op(+1,0,L) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), N_0 @ op(-1,L-1,L)]
    
    if hermitian_kraus:
        X_0 = np.eye(2**L) 
        X_L = np.eye(2**L) 
    else:
        X_0 = op(+1, 0, L) + op(-1, 0, L) 
        X_L = op(+1, L-1, L) + op(-1, L-1, L) 
    
    krs_operatos = [np.eye(2**L), X_0 @ (np.eye(2**L) - N_0), N_0, 
                    np.eye(2**L) - N_L, X_0 @ (np.eye(2**L) - N_0) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L),
                    X_L @ N_L, X_0 @ (np.eye(2**L) - N_0) @ X_L @ N_L, N_0 @ X_L @ N_L ]

    particle_In = [0, 1, 0, 0, 1, 0, 0, 1, 0]
    particle_Out = [0, 0, 0, 0, 0, 0, -1, -1, -1]
    
    output_array = []

    # kraus_counts = []
    kraus_indexes = []
    kraus_probabilities = []
    kraus_norms = []
    
    in_flow = []
    out_flow = []

    for time in time_list:
        
        obsrv = function(ini_state, (Lx, Ly), **Kwargs)
        output_array.append(obsrv)        
        
        ini_state = U_dt @ ini_state
        
        
        n_0 = (ini_state @ N_0 @ ini_state.conj() ).real 
        n_L = (ini_state @ N_L @ ini_state.conj() ).real 

        kr_in = [1-K_rate, K_rate*(1-n_0), K_rate*n_0]
        kr_out = [1-K_rate, K_rate*(1-n_L), K_rate*n_L]
        
        krs_prbb = np.array([ a*b for a in kr_in for b in kr_out])
        krs_rates = np.cumsum(krs_prbb)
        
        
        coin = np.random.uniform(0,1)  
        
        index = len(krs_rates[krs_rates < coin])
        ini_state = krs_operatos[index] @ ini_state
        
        
        norm = np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
        
        ini_state = 1/norm * ini_state
        
        # kraus_counts.append([krs_prbb[index], norm])  
        index_set = np.zeros((9,), dtype=np.int32)
        index_set[index] = 1
         
        kraus_indexes.append(index_set)  
        kraus_probabilities.append(krs_prbb[index])  
        kraus_norms.append(norm)  
        
        in_flow.append(particle_In[index])  
        out_flow.append(particle_Out[index])  


    return(np.array(output_array), np.array(kraus_indexes), np.array([in_flow, out_flow]) )



def new_run_simulation(time_list, dimensions, couplings, **Kwargs):
    function = Kwargs.get('observable', particle_counts)
    ini_state = Kwargs.get('initial_state', config_state( np.resize( [0,1], np.prod(dimensions)) ))
    K_rate = Kwargs.get('driving_rate', 0.5)
    hermitian_kraus = Kwargs.get('hermitian_kraus', False)
    
    Lx, Ly = dimensions    
    L = Lx * Ly
    
    dt = time_list[-1]/len(time_list)
    
    ham = Hamiltonian_2d( couplings, dimensions, **Kwargs)
    
    U_dt = sps.linalg.expm(-1j * ham *dt) # type: ignore

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = ini_state.toarray()   
    
    # krs_operatos = [np.eye(2**L), op(+1,0,L), np.eye(2**L), np.eye(2**L), op(+1,0,L), np.eye(2**L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), op(-1,L-1,L)]
    # krs_operatos = [np.eye(2**L), op(+1,0,L), N_0, np.eye(2**L) - N_L, op(+1,0,L) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), N_0 @ op(-1,L-1,L)]
    
    if hermitian_kraus:
        X_0 = np.eye(2**L) 
        X_L = np.eye(2**L) 
    else:
        X_0 = op(+1, 0, L) + op(-1, 0, L) 
        X_L = op(+1, L-1, L) + op(-1, L-1, L) 
    
    krs_operatos = [np.eye(2**L), X_0 @ (np.eye(2**L) - N_0), N_0, 
                    np.eye(2**L) - N_L, X_0 @ (np.eye(2**L) - N_0) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L),
                    X_L @ N_L, X_0 @ (np.eye(2**L) - N_0) @ X_L @ N_L, N_0 @ X_L @ N_L ]

    particle_In = [0, 1, 0, 0, 1, 0, 0, 1, 0]
    particle_Out = [0, 0, 0, 0, 0, 0, -1, -1, -1]

    output_array_1 = []
    output_array_2 = []

    
    for time in time_list:
                
        obsrv1 = particle_counts(ini_state, (Lx, Ly), **Kwargs)
        output_array_1.append(obsrv1)        
        obsrv2 = resolved_currents_dict(ini_state, (Lx, Ly), **Kwargs)
        output_array_2.append(obsrv2)        
        
        ini_state = U_dt @ ini_state
        
        
        n_0 = (ini_state @ N_0 @ ini_state.conj() ).real 
        n_L = (ini_state @ N_L @ ini_state.conj() ).real 

        kr_in = [1-K_rate, K_rate*(1-n_0), K_rate*n_0]
        kr_out = [1-K_rate, K_rate*(1-n_L), K_rate*n_L]
        
        krs_prbb = np.array([ a*b for a in kr_in for b in kr_out])
        krs_rates = np.cumsum(krs_prbb)
        
        
        coin = np.random.uniform(0,1)  
        
        index = len(krs_rates[krs_rates < coin])
        ini_state = krs_operatos[index] @ ini_state
        
        
        norm = np.real(np.sqrt( np.real(ini_state @ ini_state.conj()) ))
        
        ini_state = 1/norm * ini_state

    return(np.array(output_array_1), np.array(output_array_2) )

