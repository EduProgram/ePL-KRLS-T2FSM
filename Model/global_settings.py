"""This module lists settings that are used throughout the toolkit."""

from decimal import Decimal
from numpy import linspace
import colour as cl

global_uod = (0, 10)  # universe of discourse boundaries
global_x_disc = 101
global_alpha_disc = 10
global_zlevel_disc = 4

# parameters of integrate the distance metrics (#Edu)
abs_e = 1.49e-08  # default is: 1.49e-08
rel_e = 1.49e-08  # default is: 1.49e-08
lim = 50  # default is: 50

# for visualisations
xlabel = r'$x$'
type_1_ylabel = r'$\mu(x)$'
type_2_ylabel = r'$u(x)$'
colours = ['#006767', '#AC0000', '#E9AF3B', '#34539C', '#E98A3B', '#660033']

# sets if fuzzy sets generated from data have normal or non-normal functions
normalise_generated_sets = True

# do not change this, use set_rounding() to change this instead
DECIMAL_ROUNDING = Decimal(10) ** -4

UNIQUE, HEATMAP, GREYSCALE = range(3)
type_2_3d_colour_scheme = GREYSCALE


def set_rounding(decimal_places):
    """Set the total decimal values returned in results."""
    global DECIMAL_ROUNDING
    DECIMAL_ROUNDING = Decimal(10) ** -decimal_places


def rnd(x):
    """Round the value x to set number of decimal values."""
    return Decimal(x).quantize(DECIMAL_ROUNDING)


def get_x_points():
    """Get a list of discretised points in the universe of discourse."""
    return [rnd(x)
            for x in linspace(global_uod[0], global_uod[1], global_x_disc)]


def get_y_points():
    """Get a list of discretised alpha-cut points."""
    return [rnd(Decimal(i)/global_alpha_disc)
            for i in range(global_alpha_disc, 0, -1)]


def get_z_points():
    """Get a list of discretised zLevels."""
    return [rnd(Decimal(i)/global_zlevel_disc)
            for i in range(global_zlevel_disc, 0, -1)]


def get_z_level_heatmap():
    """Get a list of hex colours for the heat map of zSlices."""
    return list(cl.Color('blue').range_to(cl.Color('red'), global_zlevel_disc)) #Edu


def get_z_level_greyscale():
    """Get a list of hex colours for the heat map of zSlices."""
    return list(cl.Color('#BBBBBB').range_to(cl.Color('#222222'), #Edu
                global_zlevel_disc))
