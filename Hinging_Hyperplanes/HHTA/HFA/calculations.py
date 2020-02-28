'This file contains all calculations regarding the hinge finding algorihtm'

import numpy as np
from sklearn.linear_model import LinearRegression
#from Hinging_Hyperplanes.visualisation.create_plots import plot_3D_scatter_plane

#%%
def remove_nan_values(data):
    data[0] = data[0][~np.isnan(data[1]).any(axis=1)]
    data[1] = data[1][~np.isnan(data[1])]
    data[1] = data[1][~np.isnan(data[0]).any(axis=1)] # if any value in any dimension is zero it needs to be removed
    data[0] = data[0][~np.isnan(data[0]).any(axis=1)]
    return data

#%%
def arbitrary_first_hinge(data):
    """Funktion to find an arbitrary first hinge
    X is a numpy ndarray that contains the X values of all datapoints the dimension of data[0] is d x n
    """
    n, d = data[0].shape
    max_val = np.zeros((d))
    min_val = np.zeros((d))
    for i in range(d):
        max_val[i] = np.max(data[0][:,i])
        min_val[i] = np.min(data[0][:,i])
    Delta = np.zeros((d))
    no_suitable_hinge = True
    count = 0
    while no_suitable_hinge: # just some number as break criterion
        for i in range(d):
            rand_val = min_val[i]+(max_val[i]-min_val[i])*(2*np.random.rand(1)-1) #to avoid 0 and a value higher then 1 should cause no issue
            if rand_val == 0:
                rand_val = rand_val+0.1*min_val[i]
            Delta[i] = 1/rand_val
        count +=1
        Delta[0] = 0
        data_minus, data_plus = separate_data(data, Delta)
        if (data_minus[1].size > d*2) and (data_plus[1].size > d*2):
            break
        if count > 60:
            print('No initial hinge was found')
            break

    return data_plus, data_minus,Delta

def separate_data(data,Delta):
    data_plus = list()
    data_minus = list()
    index_plus = np.dot(data[0],Delta)>0
    data_plus.append(data[0][index_plus[:],:])
    data_plus.append(data[1][index_plus])
    data_minus.append(data[0][~index_plus[:],:])
    data_minus.append(data[1][~index_plus])
    return data_minus, data_plus

def Hinge_finding_algorithm(data):
    'Hinge finding algorithm to find the optimal split of a data set into two subsets'
    n, d = data[0].shape
    'first data seperation'
    data_plus,data_minus,Delta = arbitrary_first_hinge(data)
    conv_loop_counter = 0
    not_converged = True
    while not_converged:
        in_data_set = False
        Delta_old = Delta
        lr = LinearRegression()
        lr.fit(X=data_plus[0][:,1:], y=data_plus[1])
        theta_plus = np.append(lr.intercept_,lr.coef_)
        #data_mesh_plus = calc_mesh_data_2D(data_plus, lr.coef_,lr.intercept_)
        lr = LinearRegression()
        lr.fit(X=data_minus[0][:,1:], y=data_minus[1])
        theta_minus = np.append(lr.intercept_,lr.coef_)
        #data_mesh_minus = calc_mesh_data_2D(data_minus, lr.coef_,lr.intercept_)
        #data_plot = [data_minus,data_plus]
        #data_mesh_plot = [data_mesh_minus,data_mesh_plus]
        #plot_3D_scatter_plane(data_plot, data_mesh_plot)
        Delta_Nt = theta_plus-theta_minus
        dN_loop_counter = 0
        while ~in_data_set:
            dN_loop_counter += 1
            mu = 1/dN_loop_counter # stepwise decrease of a damped Newton factor
            Delta = Delta_old + mu*(Delta_Nt-Delta_old)
            data_minus, data_plus = separate_data(data, Delta)
            if (data_minus[1].size > d * 2) and (data_plus[1].size > d * 2):
                break
            if dN_loop_counter > 60:
                #print('error in the damped newton method of the hinge finding algorithm')
                raise Exception("damped Newton algorithm does not converge")
        diff = np.cos(np.abs(np.divide(np.absolute(Delta) - np.abs(Delta_old),np.abs(Delta))))
        if min(diff) > 0.99:
            break
        conv_loop_counter += 1
        if conv_loop_counter > 60:
            raise Exception("HFA does not converge")
    return data_minus,data_plus

def calc_hinge_function_value(X,coeff,intersect):
    return np.dot(X,coeff)+intersect

def calc_mesh_data_2D(data,coeff,intersect):
    nx, ny = (50, 50)
    x = np.linspace(min(data[0][:,1]), max(data[0][:,1]), nx)
    y = np.linspace(min(data[0][:,2]), max(data[0][:,2]), ny)
    xv, yv = np.meshgrid(x, y)
    X = np.column_stack((xv.flatten(),yv.flatten()))
    Y = calc_hinge_function_value(X,coeff,intersect)
    data_mesh = list()
    data_mesh.append(X)
    data_mesh.append(Y)
    return data_mesh

def linear_regession(data):
    lr = LinearRegression()
    lr.fit(X=data[0][:, 1:], y=data[1])
    coef = np.append(lr.intercept_, lr.coef_)
    return coef, lr

#%%
if __name__ == "__main__":
    test0 = (np.random.rand(200,2)-0.5)*2
    test1 = np.zeros((200,1))
    test1[:,0] = np.exp(test0[:,0])
    data = list()
    data.append(test0)
    data.append(test1)
    data[0][3,0] = np.nan
    data[0][14,1] = np.nan
    data[1][2,0] = np.nan
    test = remove_nan_values(data)
    #data_plus, data_minus,Delta = arbitrary_first_hinge(data)
    data_minus,data_plus = Hinge_finding_algorithm(data)
    data = [data_plus,data_minus]

    #import Hinging_Hyperplanes.visualisation.create_plots as cp
    #cp.plot_scatter3D(data)
    a = 1