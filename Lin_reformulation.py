'Reformulation of 1-D Functions'
'Small framework for the reformulation of 1-D piecewise linear functions'
"""Basic idea is to seperate all convex areas for a simple formulation. Furthermore the function is reformed as log
formulation proposed by Vielma et al. (2009) Mixed-Integer Models for Nonseparable Piecewise-Linear Optimization: Unifying
Framework and Extensions.
"""
import numpy as np
from sympy import Le, Eq,Ge, symbols, Symbol
from sympy.abc import x,y,z # x and y are input variables and z is the outputvariable'
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib


"a small dictionary for testing"
def test_lin(num_pwle= 5):
    def FuelCell(El_prod):
        return 0.11 + 1.2 * El_prod + 0.99 * El_prod ** 2

    def Electrolysis(Output):
        return (Output/(-0.0498658434302489*Output**3 + 0.302140545845083*Output**2 + 0.715060196682739*Output + 0.0326651009024266))*0.629761721491849*Output
    def Sin_test(test_in):
        return math.cos((test_in*math.pi))


    x = list()
    y = list()
    pwlm = {}
    breakpoints = [i/num_pwle for i in range(num_pwle+1)]
    for i in np.linspace(0,1.6,num_pwle+1):
        x.append(i)
        y.append(Electrolysis(i)) ######### hier Funktion einfügen

    for i in range(len(x)-1):
        pwlm["Lin_el" + str(i)] = {}
        pwlm["Lin_el"+ str(i)]["start/endpoint"] = [x[i],x[i+1]]
        pwlm["Lin_el" + str(i)]["slope"] = (y[i + 1]-y[i])/(x[i + 1]-x[i])
        pwlm["Lin_el" + str(i)]["intersect"] = y[i] - x[i]*pwlm["Lin_el" + str(i)]["slope"]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    return pwlm


def seperate_con_nonconv(pwlm):
    """pwlm contains the piece wise linear model o the the function to be modeled. The model is a dictionary containing
    the linear elements of the model. The piecewise linear model is a dictionary with the values "start/endpoint", "slope"
    and "intersect".
    In the first step, the model is sorted by the start point to ensure, that the next loop iterates from left to write.
    """
    pwlm_array = np.zeros((len(pwlm),4)) # 4 values as each linear element is characterised by 4 constants
    iter = 0
    for pwle in pwlm:
        pwlm_array[iter, 0] = pwlm[pwle]["start/endpoint"][0]
        pwlm_array[iter, 1] = pwlm[pwle]["start/endpoint"][1]
        pwlm_array[iter, 2] = pwlm[pwle]["slope"]
        pwlm_array[iter, 3] = pwlm[pwle]["intersect"]
        iter  = iter+1
    pwlm_array = pwlm_array[pwlm_array[:, 0].argsort()]
    pwl_section = list()
    for i in range(len(pwlm_array)):
        if i > 0:
            if pwlm_array[i, 2] > pwlm_array[i-1, 2]:
                pwl_section[sections].append(pwlm_array[i,:])
            else:
                pwl_section.append(list())
                sections += 1
                pwl_section[sections].append(pwlm_array[i,:])
        else:
            pwl_section.append(list())
            pwl_section[0].append(pwlm_array[i, :])
            sections = 0

    return pwl_section, sections+1

def convex_formulation(linear_model,pwlm,input_var,output_var,bin_var,num_pwle):
    P_base = pwlm[0][3]+pwlm[0][2]*pwlm[0][0]
    upper_bound = list()
    lower_bound = list()
    variables = list()
    itter = 0 # Dirty fix
    for i in range(num_pwle, len(pwlm) + num_pwle, 1):
        upper_bound.append(pwlm[itter][1] - pwlm[itter][0])
        lower_bound.append(0)
        variables.append(Symbol("lin_seg_" + str(i)))
        linear_model["variables"].append(variables[itter])
        itter += 1
    linear_model["constraints"].append(Eq(sum(variables)+pwlm[0][0]*bin_var, input_var))
    linear_model["constraints"].append(Eq(
        sum(variables[i] * pwlm[i][2] for i in range(len(variables))) + P_base *
        bin_var, output_var))
    for i in range(len(upper_bound)):
        linear_model["constraints"].append(Le(variables[i],upper_bound[i]*bin_var))
        linear_model["bounds"][variables[i]] = (lower_bound[i], upper_bound[i])
    return linear_model

def sympy_formulation(pwl_section,sections,var_in,var_out,op_status):
    """As the example is very small: if the wohle area is convex: model written without binary variables. Otherwise: SOS2
    constraints, in further research it is to be examined if a part convex combination is applicable. Furthermore the
    ZigZag formulation might increase the complexity even further. """
    #bin_number = math.ceil(math.log(len(pwl_section),2))
    #for conv_subset in pwl_section:
    #    if len(conv_subset) > 1: # single element of a concave area
    linear_model = dict()
    linear_model["constraints"] = list()
    linear_model["variables"] = list()
    linear_model["bin_var"] = list()
    linear_model["bounds"] = dict()
    linear_model["inputs"] = list()
    linear_model["inputs"] = var_in
    linear_model["output"] = var_out
    linear_model["op_status"] = op_status
    section_count = 0
    num_pwle = 0
    exp_for_sum = list()
    add_var_o = list()
    add_var_u = list()
    slope_list = list()
    intersect_list = list()
    add_list = list()
    if sections > 1:
        'At first multiple choise formulation '
        for pwle in pwl_section:
            if len(pwle) > 1:
                linear_model["bin_var"].append(Symbol(f'op_var_{section_count}'))
                add_var_o.append(Symbol(f'conv_var_o{section_count}'))
                add_var_u.append(Symbol(f'conv_var_u{section_count}'))
                linear_model = convex_formulation(linear_model, pwle, Symbol(f'conv_var_u{section_count}'), Symbol(f'conv_var_o{section_count}'), Symbol(f'op_var_{section_count}'), num_pwle)
                for i in range(len(linear_model["variables"])-num_pwle): # Very in efficient but i don't kown how to write it different in python
                    slope_list.append(0)
                    add_list.append(0)
                intersect_list.append(0)
            else:
                linear_model["variables"].append(Symbol("lin_seg_" + str(num_pwle)))
                linear_model["bin_var"].append(Symbol(f'op_var_{section_count}'))
                linear_model["constraints"].append(Le(Symbol("lin_seg_" + str(num_pwle)), pwle[0][1]*Symbol(f'op_var_{section_count}')))
                linear_model["constraints"].append(
                    Ge(Symbol("lin_seg_" + str(num_pwle)), pwle[0][0] * Symbol(f'op_var_{section_count}')))
                slope_list.append(pwle[0][2])
                intersect_list.append(pwle[0][3])
                add_list.append(1)
                linear_model["bounds"][Symbol("lin_seg_" + str(num_pwle))] = (0, pwle[0][1])
            num_pwle = len(linear_model["variables"])
            section_count += 1
        linear_model["constraints"].append(Eq(sum(linear_model["bin_var"]), op_status))
        linear_model["constraints"].append(Eq(sum(linear_model["variables"][i]*add_list[i] for i in range(len(add_list))) +sum(add_var_u), var_in))
        linear_model["constraints"].append(Eq(sum(linear_model["variables"][i]*slope_list[i] for i in range(len(slope_list))) +
                                              sum(linear_model["bin_var"][i]*intersect_list[i] for i in range(len(intersect_list)))
                                              + sum(add_var_o), var_out))
        for var in add_var_o:
            linear_model["variables"].append(var)
            linear_model["bounds"][var] = (0,1000) #ToDo: Define appropriate boundaries
        for var in add_var_u:
            linear_model["variables"].append(var)
            linear_model["bounds"][var] = (0, 1000)  # ToDo: Define appropriate boundaries

    else:
        linear_model =  convex_formulation(linear_model, pwlm = pwl_section[0], input_var = linear_model["inputs"],
                                           output_var = linear_model["output"], bin_var=op_status, num_pwle = 0)


    return linear_model


def lin_example(inputs,output,op_status):
    pwlm = test_lin(5)
    pwl_section, sections = seperate_con_nonconv(pwlm)
    linear_model = sympy_formulation(pwl_section, sections, inputs, output, op_status)
    return linear_model

linear_model= lin_example(x,z,y)

a = 1




###### save the model to import it in the comando famework #########
name = 'Electrolyzer'
with open(f'Linear_model_{name}',"wb") as fp:
    pickle.dump(linear_model,fp)
    linear_model['constraints'][0]