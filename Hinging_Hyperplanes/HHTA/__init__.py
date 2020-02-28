'define the class of the linear model'
import Hinging_Hyperplanes.HHTA.HFA.calculations as calc
from Hinging_Hyperplanes.HHTA.HFA.calculations import Hinge_finding_algorithm
import Hinging_Hyperplanes.visualisation.create_plots as vis
import numpy as np


class Linear_model():
    """ Linear model of the input data.
    The aim is to approximate a dependency of a multidimensional input X  with an output y using a multivariate
    function: y = f(X)
    The approach of the hinging hyperplane tree algorithm is a data driven generation of a piecewise linear function
    to approximate f(X) with linear elements.
    The input data needs to be sorted in the following way:
    data is a list with two elements.
    data[0] is the X of the measured data. X is element of R^(n x d) where n is the number of measured points and d is
                                           the number of input dimensions
    data[1] is the y of the measured data. y is element of the R^n
    """
    def __init__(self,label,data):
        data = calc.remove_nan_values(data)  # remove all nan
        n, d = data[0].shape
        data[0] = np.concatenate((np.ones((n, 1)), data[0]), axis=1)
        self.original_data = data

        self.coeff, self.linear_model = calc.linear_regession(data)
        visualize = {}
        visualize["scatter"] = False
        visualize["linear_model"] = True
        visualize["convex_hull"] = True
        self.vis_settings  # settings for a later visualization

    def add_linear_element(self,data_subset):
        """ This function splits the data in data_subset into two data subsets.
        This is useful if we want to split one linear element manually  """
        data_minus, data_plus = Hinge_finding_algorithm(data_subset)
        coeff_minus, linear_model_minus = calc.linear_regession(data_minus)
        coeff_plus, linear_model_plus = calc.linear_regession(data_plus)


    def visualize(self):
        """ Visualization of the linear model
        Settings_vis is a dictionary with different settings for the visualization of the hinge function
        in the following the different settings are explained
        scatter = boolean : True is the data points should be plotted
        linear_model = boolean : True if the linear model should be plotted
        """
        settings_vis = vis.settings

        a = 1


