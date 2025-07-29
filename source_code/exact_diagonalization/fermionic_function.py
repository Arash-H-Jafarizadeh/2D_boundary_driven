import numpy as np
# import scipy as sp
# import time as tt
# import matplotlib.pyplot as plt
# from numba import njit, prange
# from numba import njit, prange

###################################################################################################################################################################################
################################################################################## Basic functions ################################################################################
###################################################################################################################################################################################

def fc(a,j,L): #this function has a problem, index j should start from 1 to L, not 0 to L-1 !
    spin = np.array([[0,1],[1,0]]), np.array([[0,-1j],[+1j,0]]), np.array([[1,0],[0,-1]])     
    out = 1
    for _ in range(j-1):
        #JW = np.kron(np.eye(2**(j-1)), np.kron(spin[abs(int(a)-1)], np.eye(2**(L-j))))
        out = np.kron(out,spin[2])
    op = 0.5*(spin[0]+a*1j*spin[1])
    # out = np.kron(out, np.kron(op, np.eye(2**(L-np.mod(j,L)-1))))
    out = np.kron(out, np.kron(op, np.eye(2**(L-j))))
    return out


def op(a,j,L):
    
    spin = np.array([[0,1],[1,0]]), np.array([[0,-1j],[+1j,0]]), np.array([[1,0],[0,-1]]) 
    
    out = 1
    for _ in range(j):
        out = np.kron(out, spin[2])
    
    op = 0.5*(spin[0]+a*1j*spin[1])

    out = np.kron(out, np.kron(op, np.eye(2**(L-j-1))))    
    return out


###################################################################################################################################################################################
################################################################################### 2D functions ##################################################################################
###################################################################################################################################################################################


def config_state(lisst):
    # L = len(lisst)
    # bits = [[1,0],[0,1]]
    bits = [[0,1],[1,0]]
    
    out = [1]
    for x in lisst:
        # out = np.kron(out, bits[np.mod(x,2)])
        out = np.kron(out, bits[x])
    return out



def Hubbard_Ham_2d(physical, dims, **kwargs):
    PBC = kwargs.get('PBC', False)

    V, J = physical[:2] 
    Lx, Ly = dims
    L = Lx*Ly
    
    Hy = np.eye(Ly, k=1, dtype=np.int16) + np.eye(Ly, k=-1, dtype=np.int16)  + int(PBC)*np.eye(Ly, k=Ly-1, dtype=np.int16)  + int(PBC)*np.eye(Ly, k=-Ly+1, dtype=np.int16)    
    Hx = np.eye(Lx, k=1, dtype=np.int16) + np.eye(Lx, k=-1, dtype=np.int16)  + int(PBC)*np.eye(Lx, k=Lx-1, dtype=np.int16)  + int(PBC)*np.eye(Lx, k=-Lx+1, dtype=np.int16)    
    adj_mat = np.kron(Hy, np.eye(Lx, dtype=np.int16)) + np.kron(np.eye(Ly, dtype=np.int16), Hx)

    out = np.zeros((2**L,2**L), dtype=np.complex128)
    
    # for xx in range(1, L+1):
    #     for yy in range(xx, L+1):
    #         out += -J * adj_mat[xx-1,yy-1] * (fc(+1,xx,L) @ fc(-1,yy,L) + fc(+1,yy,L) @ fc(-1,xx,L))
    #         out += +V/2 * adj_mat[xx-1,yy-1] * ( fc(+1,xx,L) @ fc(-1,xx,L) ) @ ( fc(+1,yy,L) @ fc(-1,yy,L) )
    
    for xx in range(L):
        for yy in range(xx, L):
            out += -J * adj_mat[xx,yy] * (op(+1,xx,L) @ op(-1,yy,L) + op(+1,yy,L) @ op(-1,xx,L))
            out += +V/2 * adj_mat[xx,yy] * ( op(+1,xx,L) @ op(-1,xx,L) ) @ ( op(+1,yy,L) @ op(-1,yy,L) )
    
    return out 



def Ut(dt, e_values, e_vectors):    
    return e_vectors @ np.diag(np.exp(-1j * e_values * dt)) @ e_vectors.T.conj()



def evolved_state(ini_state, time, e_values, e_vectors):

    U_t = e_vectors @ np.diag(np.exp(-1j*e_values*time)) @ e_vectors.T.conj()
    # ini_state = U_t @ ini_state
    return( U_t @ ini_state )



def particle_counts(input_state, dimensions, **kwargs):
    # shaped = kwargs.get('shaped', True)
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    input_state = np.array(input_state)
    
    Ns = np.zeros((L,) , dtype=np.float64) 
    for n in range(L):    
        n_th = op(+1,n,L) @ op(-1,n,L) 
        # Ns[n] += 1 - (input_state @ n_th @ input_state.conj() ).real
        Ns[n] += (input_state @ n_th @ input_state.conj() ).real
        
    return( np.reshape(Ns, (Ly, Lx)) )



def resolved_currents(input_state, dimensions, **kwargs):
    # shaped = kwargs.get('shaped', True)    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    # ii, jj = indexes 
    input_state = np.array(input_state)
    
    J_mat = np.zeros((L, L) , dtype=np.float64) 
    for x in range(L):    
        for y in range(L):    
            J_ij = +1j*( op(+1,x,L) @ op(-1,y,L) - op(+1,y,L) @ op(-1,x,L) ) 
            J_mat[x,y] += ( input_state @ J_ij @ input_state.conj() ).real
        
    return( J_mat )


def hopping_correlation(input_state, dimensions, **kwargs):
    # shaped = kwargs.get('shaped', True)
    
    Lx, Ly = dimensions
    L = Lx * Ly
    
    input_state = np.array(input_state)
    
    C_Mat = np.zeros((L,L) , dtype=np.float64) 
    for x in range(L):    
        for y in range(L):    
            C_xy = op(+1,x,L) @ op(-1,y,L) 
            C_Mat[x,y] += input_state @ C_xy @ input_state.conj() #.real
        
    return(C_Mat)


def density_correlation(input_state, dimensions, **kwargs):
    # shaped = kwargs.get('shaped', True)
    Lx, Ly = dimensions
    L = Lx * Ly
    
    # ii, jj = indexes 
    input_state = np.array(input_state)
    
    output = np.zeros((L,L) , dtype=np.float64) 
    for y in range(L):    
        for x in range(L):    
            N_N = op(+1,x,L) @ op(-1,x,L) @ op(+1,y,L) @ op(-1,y,L) 
            output[x,y] += (input_state @ N_N @ input_state.conj()).real
        
    return(output)




def sector_indexes_2D(dims, EVs):
    
    Lx, Ly = dims
    sctr_indxs = [[] for _ in range(Lx*Ly+1)]
    for n in range(2**(Lx*Ly)):
        pcns = particle_counts(EVs[:,n], Lx*Ly)
        tpcns = np.round(np.sum(pcns), decimals=0)
        sctr_indxs[int(tpcns)].append(n)
        
    return sctr_indxs


def total_particle_counts(dims):
    
    Lx, Ly = dims
    L = Lx * Ly
    out = np.zeros((2**L,2**L), dtype=np.complex128)  
    for n in range(1,L+1):    
        out += op(+1,n,L) @ op(-1,n,L)     
    
    return out.real



# ########## notes to self (20250516): I dont think it worth it to make a function for time simulation. Doing a manual script may be much faster easier


def run_simulation(time_list, dimensions, couplings, **Kwargs):
    function = Kwargs.get('observable', particle_counts)
    ini_state = Kwargs.get('initial_state', config_state( np.resize( [0,1], np.prod(dimensions)) ))
    inj_prob, ext_prob = Kwargs.get('driving_rates', [0.5, 0.5])
    
    Lx, Ly = dimensions    
    L = Lx * Ly
    
    
    ham = Hubbard_Ham_2d( couplings, dimensions, **Kwargs)
    e_list, v_mat = np.linalg.eigh(ham)

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = np.array(ini_state)
    
    dt = time_list[-1]/len(time_list)
    U_dt = v_mat @ np.diag(np.exp(-1j*e_list*dt)) @ v_mat.T.conj()
     
    output_array = []

    injections, extractions = [], []

    for time in time_list:

        obsrv = function(ini_state, (Lx, Ly), **Kwargs)
        output_array.append(obsrv)        
        
        # ini_state = evolved_state(ini_state, dt, e_list, v_mat, **Kwargs)
        ini_state = U_dt @ ini_state
        
        P_d, P_s = np.random.uniform(0,1), np.random.uniform(0,1)  

        if P_d <= inj_prob:
            n_0 = (ini_state @ N_0 @ ini_state.conj() ).real #n_counts[0,0]
            temp_P = np.random.uniform(0,1)

            if temp_P > n_0:
                # ini_state = op(+1,0,L) @ ini_state
                ini_state = (op(+1,0,L) + op(-1,0,L)) @ ini_state
                injections.append(time)
        
        if P_s <= ext_prob:
            n_L = (ini_state @ N_L @ ini_state.conj() ).real #n_counts[Ly-1,Lx-1]
            temp_P = np.random.uniform(0,1)
            
            if temp_P < n_L:
                # ini_state = op(-1,L-1,L) @ ini_state
                ini_state = (op(-1,L-1,L) + op(+1,L-1,L)) @ ini_state
                extractions.append(time)

    
    return(np.array(output_array), 
           ini_state, 
           np.array(injections), 
           np.array(extractions)
           )



# ########## notes to self (20250522): I added this function after talking to juan. This includes random selection of Krause operators.
 
def new_run_simulation(time_list, dimensions, couplings, **Kwargs):

    function = Kwargs.get('observable', particle_counts)
    ini_state = Kwargs.get('initial_state', config_state( np.resize( [0,1], np.prod(dimensions)) ))
    K_rate = Kwargs.get('driving_rate', 0.5)
    # X_type = Kwargs.get('X_type', True)
    
    Lx, Ly = dimensions    
    L = Lx * Ly
    
    ham = Hubbard_Ham_2d( couplings, dimensions, **Kwargs)
    e_list, v_mat = np.linalg.eigh(ham)

    N_0 = op(+1, 0, L) @ op(-1, 0, L) 
    N_L = op(+1, L-1, L) @ op(-1, L-1, L) 

    ini_state = np.array(ini_state)
    
    dt = time_list[-1]/len(time_list)
    U_dt = v_mat @ np.diag(np.exp(-1j*e_list*dt)) @ v_mat.T.conj()
     
    # krs_operatos = [np.eye(2**L), op(+1,0,L), np.eye(2**L), np.eye(2**L), op(+1,0,L), np.eye(2**L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), op(-1,L-1,L)]    
    # krs_operatos = [np.eye(2**L), op(+1,0,L), N_0, np.eye(2**L) - N_L, op(+1,0,L) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), N_0 @ op(-1,L-1,L)]

    X_0 = op(+1, 0, L) + op(-1, 0, L) 
    X_L = op(+1, L-1, L) + op(-1, L-1, L) 
    krs_operatos = [np.eye(2**L), X_0 @ (np.eye(2**L) - N_0), N_0, 
                    np.eye(2**L) - N_L, X_0 @ (np.eye(2**L) - N_0) @ (np.eye(2**L) - N_L), N_0 @ (np.eye(2**L) - N_L),
                    X_L @ N_L, X_0 @ (np.eye(2**L) - N_0) @ X_L @ N_L, N_0 @ X_L @ N_0 ]
    
    particle_In = [0, 1, 0, 0, 1, 0, 0, 1, 0]
    particle_Out = [0, 0, 0, 0, 0, 0, -1, -1, -1]
    
    output_array = []

    kraus_counts = []

    in_flow, out_flow = [], []

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
        
        # if X_type:
        #     krs_operatos = [np.eye(2**L), (op(+1,0,L) + op(-1,0,L)), np.eye(2**L), np.eye(2**L), (op(+1,0,L) + op(-1,0,L) ), np.eye(2**L),
        #         (op(-1,L-1,L) + op(+1,L-1,L)), (op(+1,0,L) + op(-1,0,L)) @ (op(-1,L-1,L) + op(+1,L-1,L)), (op(-1,L-1,L) + op(+1,L-1,L))]
        # else:
        #     krs_operatos = [np.eye(2**L), op(+1,0,L), np.eye(2**L), np.eye(2**L), op(+1,0,L), np.eye(2**L), op(-1,L-1,L), op(+1,0,L) @ op(-1,L-1,L), op(-1,L-1,L)]
        
        coin = np.random.uniform(0,1)  
        
        index = len(krs_rates[krs_rates < coin])
        ini_state = krs_operatos[index] @ ini_state
        # ini_state = krs_prbb[index]*(krs_operatos[index] @ ini_state)
        
        norm = np.sqrt( np.real(ini_state @ ini_state.conj()) )
        ini_state = 1/norm * ini_state
        
        kraus_counts.append([index, krs_prbb[index], norm])
        
        in_flow.append(particle_In[index])  
        out_flow.append(particle_Out[index]) 
        # print(f"    - - Particle in? {particle_In[index]}, index was: {index} ")
        # print(f"    - - Particle out? {particle_Out[index]}, index was: {index} ")
        
        # print(f"    - - old norm = {norm}, kraus_prob = {krs_prbb[index]}, new norm={np.real(ini_state @ ini_state.conj())}")
        # print(f"    - - old norm = {1/norm}, kraus_prob = {krs_prbb[index]}, new norm={( ini_state @ ini_state.conj() ).real}")

    return(np.array(output_array), np.array(kraus_counts), np.array([in_flow, out_flow]) )


