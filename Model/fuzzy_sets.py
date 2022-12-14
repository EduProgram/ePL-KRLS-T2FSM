"""This module is used to create a type-1 fuzzy set."""

import global_settings as gs
from decimal import Decimal
from numpy import linspace , e, power, log, sqrt, mean, std, diff
from bisect import bisect #'polling_t1_fuzzy_set'
from scipy.interpolate import lagrange #'polling_t1_fuzzy_set'
from collections import Counter, defaultdict #'generate_fuzzy_sets', 'discrete_t2_fuzzy_set'
import matplotlib.pyplot as plt #'visualisations'
import inspect #'general_t2_fuzzy_set'
from mpl_toolkits.mplot3d import Axes3D #'visualisations_3d'
from mpl_toolkits.mplot3d.art3d import Poly3DCollection #'visualisations_3d'
import mpl_toolkits.mplot3d.art3d as art3d #'visualisations_3d'


# Module 'fuzzy_exceptions'
#------------------------------------------------------------------------------
"""This module lists fuzzy set based exceptions.

These are regarding non-existent parts of fuzzy sets,
e.g. empty alpha-cuts, empty zLevels.
"""

class AlphaCutError(Exception):
    """The alpha-cut exceeds the height of the fuzzy set."""

    pass


class ZLevelError(Exception):
    """The zlevel exceeds the secondary height of the fuzzy set."""

    pass


###############################################################################
"""Plot graphs in fuzzy sets"""
###############################################################################

# Module 'visualisations'
#------------------------------------------------------------------------------
"""This module is used to plot graphs of fuzzy sets."""


def _plot_type1_set(fs, plot_points, colour_index):
    """Add a type-1 fuzzy set to plt."""
    plt.plot(plot_points,
             [fs.calculate_membership(x) for x in plot_points],
             gs.colours[colour_index],
             linewidth=3)


def _plot_discrete_t1(fs, colour_index):
    """Add a type-1 discrete fuzzy set to plt."""
    for p in fs.points.keys():
        plt.axvline(x=p,
                    ymin=0,
                    ymax=fs.calculate_membership(p),
                    color=gs.colours[colour_index],
                    linewidth=3,
                    alpha=0.5)


def _plot_interval_type2_set(fs, plot_points, colour_index,
                             colour_alpha='0.8'):
    """Add an interval type-2 fuzzy set to plt."""
    Y = [fs.calculate_membership(x) for x in plot_points]
    for i in range(len(Y)):
        Y[i] = (float(Y[i][0]), float(Y[i][1]))
    plt.plot(plot_points,
             [y[0] for y in Y],
             color=gs.colours[colour_index],
             linewidth=0)
    plt.plot(plot_points,
             [y[1] for y in Y],
             color=gs.colours[colour_index],
             linewidth=0)
    plot_points = [float(p) for p in plot_points]
    plt.fill_between(plot_points,
                     [y[0] for y in Y],
                     [y[1] for y in Y],
                     color=gs.colours[colour_index],
                     alpha=0.8) #Edu (Error: before is a string called
                                #    'colour_alpha', 'alpha' can't be a string)


def _plot_general_type2_set(fs, plot_points, colour_index):
    """Add a general type-2 fuzzy set to plt."""
    for z in fs.zlevel_coords:
        _plot_interval_type2_set(fs.zslice_functions[z], plot_points,
                                 colour_index, str(Decimal('0.8') * z)) #Teste


def _plot_type2_iaa_sets(fs, plot_points, colour_index):
    """Add a type-2 interval agreement approach set to plt."""
    noughts = [0 for x in plot_points]
    float_plot_points = [float(x) for x in plot_points]
    for mf in fs.membership_functions:
        ys = [float(mf.calculate_membership(x)) for x in plot_points]
        plt.fill_between(float_plot_points, noughts, ys,
                         color=gs.colours[colour_index],
                         alpha=0.5)


def _plot_discrete_t2(fs, colour_index):
    """Add a type-2 discrete fuzzy set."""
    Z = set([])
    for x in fs.points.keys():
        Z.update(fs.points[x].values())
    for z in sorted(list(Z)):
        colour_alpha = str(Decimal('0.8') * z)
        X = []
        YL = []
        YU = []
        for x in sorted(fs.points.keys()):
            yl, yu = fs.calculate_membership(x, z)
            if yu > 0:
                X.append(float(x))
                YL.append(float(yl))
                YU.append(float(yu))
        plt.fill_between(X, YL, YU,
                         color=gs.colours[colour_index],
                         alpha=colour_alpha)
        #plt.plot(X, YL)
        #plt.plot(X, YU)


def plot_sets(fuzzy_sets, filename=None):
    """Plot the given list of fuzzy sets.

    The fuzzy sets may be of any type.
    Discretisations and axis labels are set in the global_settings module.
    If filename is None, the plot is displayed.
    If a filename is given, the plot is saved to the given location.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colour_index = 0
    plot_points = gs.get_x_points()
    for fs in fuzzy_sets:
        if (fs.__class__.__name__ == 'FuzzySet' or
                fs.__class__.__name__ == 'PollingT1FuzzySet' or
                fs.__class__.__name__ == 'IAAT1FuzzySet'):
            _plot_type1_set(fs, plot_points, colour_index)
        elif fs.__class__.__name__ == 'DiscreteT1FuzzySet':
            _plot_discrete_t1(fs, colour_index)
        elif fs.__class__.__name__ == 'IntervalT2FuzzySet':
            _plot_interval_type2_set(fs, plot_points, colour_index)
        elif fs.__class__.__name__ == 'GeneralT2FuzzySet':
            _plot_general_type2_set(fs, plot_points, colour_index)
        elif fs.__class__.__name__ == 'T2AggregatedFuzzySet':
            _plot_type2_iaa_sets(fs, plot_points, colour_index)
        elif fs.__class__.__name__ == 'DiscreteT2FuzzySet':
            _plot_discrete_t2(fs, colour_index)
        else:
            print('Unknown how to plot', fs.__class__.__name__, 'object')
        colour_index = (colour_index + 1) % len(gs.colours)
    ax.set_ylim(0, 1.01)
    ax.set_xlim(gs.global_uod[0], gs.global_uod[1])
    plt.yticks(linspace(0, 1, 6))
    ax.set_xlabel(gs.xlabel, fontsize=22)
    if 'T2' in fs.__class__.__name__:
        ax.set_ylabel(gs.type_2_ylabel)
    else:
        ax.set_ylabel(gs.type_1_ylabel)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.15)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        

# Module 'visualisations_3d'
#------------------------------------------------------------------------------
"""This module is used to plot 3-dimensional graphs of type-2 fuzzy sets."""

def _darken_colour(hexstr, p):
    """Multiply the hex values by the percentage p."""
    hexstr = hexstr.replace('#', '')
    rgb = [(ord(c)*p)/255.0 for c in hexstr.decode('hex')]
    #rgb.append(0.5)
    return rgb


def _identify_direction_changes(l):
    """Identify the indexes of a list where the values change direction.

    i.e. changing from increasing value to decreasing value, or vice versa.
    Returns a list of indexes where each index is where the new direction
    begins in the list l.
    """
    if len(l) <= 2:
        return [len(l)]
    LOWER = 0
    HIGHER = 1

    def _check_cur_state(i1, i2):
        if l[i2] > l[i1]:
            return HIGHER
        else:
            return LOWER
    change_locations = []
    cur_state = _check_cur_state(0, 1)
    for i in range(1, len(l)-1):
        new_state = _check_cur_state(i, i+1)
        if cur_state != new_state:
            change_locations.append(i+1)
        cur_state = new_state
    return change_locations


def _append_reversed_list(li):
    """Return a new list as li.extend(li.reverse())."""
    new_li = li[:]
    li_reverse = li[:]
    li_reverse.reverse()
    new_li.extend(li_reverse)
    return new_li


def _plot_faces(ax, XL, YL, XU, YU, prev_z, cur_z, col):
    """Plot the FOU for the bottom and top of the zslice."""
    X = XL[:]
    Y = YL[:]
    X.extend(XU)
    Y.extend(YU)
    verts = [list(zip(X, Y, [prev_z for i in range(len(X))]))]
    p = Poly3DCollection(verts, edgecolors='#000000')
    p.set_facecolor(col)
    ax.add_collection3d(p)
    verts = [list(zip(X, Y, [cur_z for i in range(len(X))]))]
    p = Poly3DCollection(verts, edgecolors='#000000')
    p.set_facecolor(col)
    ax.add_collection3d(p)


def _plot_edge(ax, XL, YL, prev_z, cur_z, col):
    """Plot the inside and outside edge of the zslice.

    This joins the top and bottom parts plotted by _plot_faces.
    """
    dir_changes = _identify_direction_changes(YL)
    dir_changes.insert(0, 1)
    dir_changes.append(len(YL))
    for i in range(len(dir_changes)-1):
        start = dir_changes[i]-1
        end = dir_changes[i+1]
        X = _append_reversed_list(XL[start:end])
        Y = _append_reversed_list(YL[start:end])
        Z = [prev_z for i in range(end - start)]
        Z.extend([cur_z for i in range(end - start)])
        X.append(X[0])
        Y.append(Y[0])
        Z.append(Z[0])
        verts = [list(zip(X, Y, Z))]
        p = Poly3DCollection(verts, edgecolors='#000000')
        p.set_facecolor(col)
        ax.add_collection3d(p)


def plot_sets_3d(fuzzy_sets, filename=None): #Edu
    """Display a 3-dimensional plot of the given list of fuzzy sets.

    If filename is None, the plot is displayed.
    If a filename is given, the plot is saved to the given location.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if gs.type_2_3d_colour_scheme == gs.UNIQUE:
        colours = gs.colours
    elif gs.type_2_3d_colour_scheme == gs.HEATMAP:
        colours = gs.get_z_level_heatmap()
    elif gs.type_2_3d_colour_scheme == gs.GREYSCALE:
        colours = gs.get_z_level_greyscale()
    colour_index = 0
    for fs in fuzzy_sets:
        zlevels = fs.zlevel_coords[:]
        zlevels.insert(0, 0)
        for zi in range(1, len(zlevels)):
            XL = []
            XU = []
            YL = []
            YU = []
            for x in gs.get_x_points():
                yl, yu = fs.calculate_membership(x, zlevels[zi])
                # Note: If you have these if statements then spikes
                # in IAA won't be as apparent
                if yl > 0:
                    XL.append(float(x))
                    YL.append(float(yl))
                if yu > 0:
                    # add in reverse order to loop around the fuzzy set in a circle
                    XU.insert(0, float(x))
                    YU.insert(0, float(yu))
            prev_z = float(zlevels[zi-1])
            cur_z = float(zlevels[zi])
            if gs.type_2_3d_colour_scheme == gs.UNIQUE:
                col = _darken_colour(gs.colours[colour_index], 1-(0.5*cur_z))
            else:
                col = colours[colour_index].get_rgb()
            _plot_faces(ax, XL, YL, XU, YU, prev_z, cur_z, col)
            if len(XL) > 0:
                _plot_edge(ax, XL, YL, prev_z, cur_z, col)
            if len(XU) > 0:
                _plot_edge(ax, XU, YU, prev_z, cur_z, col)
            if gs.type_2_3d_colour_scheme != gs.UNIQUE:
                colour_index += 1
        if gs.type_2_3d_colour_scheme == gs.UNIQUE:
            colour_index += 1
        else:
            colour_index = 0
    ax.set_xlim(fs.uod[0], fs.uod[1])
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel(gs.xlabel, fontsize=18)
    ax.set_ylabel(gs.type_2_ylabel, fontsize=18)
    ax.set_zlabel(gs.type_1_ylabel, fontsize=18)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

        
###############################################################################
"""Creation of the fuzzy sets"""
###############################################################################

# Module 'fuzzy_set'
#------------------------------------------------------------------------------

class FuzzySet():
    """Create a type-1 fuzzy set."""

    def __init__(self, membership_function, uod=None):
        """Create a type-1 fuzzy set.

        membership_function: a membership function object.
        uod: the universe of discourse indicated by a two-tuple.
        """
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        self.membership_function = membership_function
        

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a Decimal value.
        """
        if x < self.uod[0] or x > self.uod[1]:
            return Decimal(0)
        mu = self.membership_function.calculate_membership(x)
        return mu

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        cut = self.membership_function.calculate_alpha_cut(alpha)
        # make sure the alpha-cut is within the universe of discourse
        if isinstance(cut[0], Decimal):  # convex alpha-cut
            return max(cut[0], self.uod[0]), min(cut[1], self.uod[1])
        elif isinstance(cut[0], tuple):  # non-convex alpha-cut
            return [(max(segment[0], self.uod[0]),
                     min(segment[1], self.uod[1]))
                    for segment in cut]

    def calculate_centroid(self):
        """Calculate the centroid x-value of the fuzzy set."""
        top = 0
        bottom = 0
        for x in linspace(self.uod[0], self.uod[1], gs.global_x_disc):
            x = Decimal(str(x))
            mu = self.membership_function.calculate_membership(x)
            top += x * mu
            bottom += mu
        return gs.rnd(top / bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu


# Module 'discrete_t1_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is used to create a discrete type-1 fuzzy set."""

class DiscreteT1FuzzySet():
    """Create a discrete type-1 fuzzy set."""

    def __init__(self, points):
        """Create a discrete type-1 fuzzy set using a dict of x,mu pairs."""
        self.points = points
        self.x_min = min(points.keys())
        self.x_max = max(points.keys())
        self.height = max(points.values())

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a Decimal value.
        """
        #x = Decimal(str(x))
        try:
            return gs.rnd(self.points[x])
        except KeyError:
            return 0

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        X_over_alpha = []
        alpha = gs.rnd(alpha)
        if alpha > max(self.points.values()):
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level',
                    max(self.points.values()))
        x_values = self.points.keys()
        x_values = sorted(x_values)
        # Record when the membership value passes the alpha cut
        results = []
        previous_hit = False
        for i in range(len(x_values)):
            if self.calculate_membership(x_values[i]) >= alpha:
                if not previous_hit:
                    results.append(gs.rnd(x_values[i]))
                    previous_hit = True
            else:
                if previous_hit:
                    results.append(gs.rnd(x_values[i-1]))
                previous_hit = False
        if self.calculate_membership(x_values[-1]) >= alpha and x_values[-1] not in results:
            results.append(gs.rnd(x_values[-1]))
        if len(results) == 1:
            results.append(results[0])
        if len(results) == 2:
            return results
        return list(zip(results[0::2], results[1::2]))


    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        new_points = {}
        for k, v in self.points.items():
            new_points[gs.rnd(k + x)] = v
        self.points = new_points
        self.x_min = self.x_min + x
        self.x_max = self.x_max + x

    def calculate_centroid(self):
        """Calculate the centroid x-value of the fuzzy set."""
        top = 0
        bottom = 0
        for x in self.points.keys():
            mu = self.points[x]
            top += x * mu
            bottom += mu
        return gs.rnd(top / bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu
        
        
# Module 'discrete_t2_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is used to create a discrete type-2 fuzzy set."""

class DiscreteT2FuzzySet():
    """Create a discrete type-2 fuzzy set."""

    def __init__(self, points, uod=None):
        """Create a discrete type-1 fuzzy set using a dict as {x: {mu: z}}."""
        self.points = points
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        # A dict of z:mu pairs for the height of each zslice
        self.zlevel_coords = set([])
        for x in self.points.keys():
            self.zlevel_coords.update(self.points[x].values())
        self.zlevel_coords = sorted(list(self.zlevel_coords))
        self.zslice_primary_heights = dict((z, [0,0]) for z in self.zlevel_coords)
        for x in sorted(self.points.keys()):
            for z in self.zlevel_coords:
                l, u = self.calculate_membership(x, z)
                self.zslice_primary_heights[z][0] = max(self.zslice_primary_heights[z][0], l)
                self.zslice_primary_heights[z][1] = max(self.zslice_primary_heights[z][1], u)

    def validate_zlevel(self, z):
        """Find the closest valid zlevel.

        Checks if the zlevel at z exists. If it exists then return z.
        If not, then return the closest zlevel that encompasses z.
        """
        z = gs.rnd(z)
        if z in self.zlevel_coords:
            return z
        else:
            points = self.zlevel_coords[:]
            if z > max(points):
                raise ZLevelError('zLevel ' + str(z) +
                                  ' is higher than the greatest zLevel at ' +
                                  str(max(points)))
            points.sort()
            for i in points:
                if i > z:
                    return i

    def calculate_membership(self, x, z):
        """Calculate the primary membership of x at the zlevel z."""
        x = gs.rnd(x)
        try:
            y_list = []
            for y in self.points[x].keys():
                if self.points[x][y] >= z:
                    y_list.append(y)
            if len(y_list) == 0:
                return 0, 0
            else:
                return (gs.rnd(min(y_list)), gs.rnd(max(y_list)))
        except KeyError:
            return 0, 0

    def calculate_secondary_membership(self, x, mu):
        """Calculate the secondary membership of x at primary membership y."""
        try:
            return self.points[x][mu]
        except KeyError:
            return 0

    def calculate_alpha_cut_lower(self, alpha, z=0):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        t1_points = {}
        for x in sorted(self.points.keys()):
            yl, yu = self.calculate_membership(x, z)
            if yu >= 0:
                t1_points[x] = yl
        t1_set = DiscreteT1FuzzySet(t1_points)
        return t1_set.calculate_alpha_cut(alpha)

    def calculate_alpha_cut_upper(self, alpha, z=0):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        t1_points = {}
        for x in self.points.keys():
            yl, yu = self.calculate_membership(x, z)
            if yu >= 0:
                t1_points[x] = yu
        t1_set = DiscreteT1FuzzySet(t1_points)
        return t1_set.calculate_alpha_cut(alpha)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu

    
# Module 'polling_t1_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is used to create a continuous version of a type-1 set."""

LAGRANGE = 0
LINEAR = 1

class PollingT1FuzzySet():
    """Create a type-1 fuzzy set using the polling technique."""

    def __init__(self, points, uod=None):
        """Create a discrete type-1 fuzzy set using a dict of x,mu pairs."""
        self.points = points
        self.x_min = min(self.points.keys())
        self.x_max = max(self.points.keys())
        self.height = max(self.points.values())
        self.interp_method = LINEAR
        # get lagrange polynomial
        X = [float(x) for x in sorted(points.keys())]
        Y = [float(points[x]) for x in sorted(points.keys())]
        self.lagrange_poly = lagrange(X, Y)
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        If x is not in self.points but exists between known x-values
        then linear interpolation is used to calculate its membership.
        Returns a Decimal value.
        """
        try:
            return gs.rnd(self.points[x])
        except KeyError:
            if self.interp_method is LINEAR:
                x_values = sorted(self.points.keys())
                if x < min(x_values) or x > max(x_values):
                    return Decimal(0)
                i = bisect(x_values, x)
                xl = x_values[i-1]
                xr = x_values[i]
                yl = self.points[xl]
                yr = self.points[xr]
                # cast as compatible data type
                if isinstance(xl, Decimal):
                    x = Decimal(x)
                else:
                    x = float(x)
                # Calculate the slope of the line from (xl, yl) to (xr, yr)
                slope = (yr - yl) / (xr - xl)
                return gs.rnd(slope * (x - xl) + yl)
            else:
                y = self.lagrange_poly(float(x))  # will not work with Decimal
                return gs.rnd(str(y))
                #return y

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        alpha = gs.rnd(alpha)
        if len(self.points) == 0:
            raise Exception('There are no elements within this fuzzy set')
        if alpha > max(self.points.values()):
            raise AlphaCutError('alpha level', alpha, 'is above max y level',
                                max(self.points.values()))
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        # create x_values based on the global settings and include the
        # x values used in the fuzzy set
        x_values = list(gs.get_x_points())
        for x in self.points.keys():
            if x not in x_values:
                x_values.append(x)
        x_values = sorted(x_values)
        # Record when the membership value passes the alpha cut
        results = []
        previous_hit = False
        for i in range(len(x_values)):
            if self.calculate_membership(x_values[i]) >= alpha:
                if not previous_hit:
                    results.append(gs.rnd(x_values[i]))
                    previous_hit = True
            else:
                if previous_hit:
                    results.append(gs.rnd(x_values[i-1]))
                previous_hit = False
        if len(results) == 1:
            results.append(results[0])
        if len(results) == 2:
            return results
        return list(zip(results[0::2], results[1::2]))

    def calculate_centroid(self):
        """Calculate the centroid x-value of the fuzzy set."""
        top = 0
        bottom = 0
        for x in self.points.keys():
            mu = self.points[x]
            top += x * mu
            bottom += mu
        return gs.rnd(top / bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu


# Module 'interval_t2_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is used to create an interval type-2 fuzzy set."""

class IntervalT2FuzzySet():
    """Create an interval type-2 fuzzy set."""

    def __init__(self, mf1, mf2, uod=None):
        """Create an interval type-2 fuzzy set.

        mf1: first membership function object
        mf2: second membership function object
        uod: the universe of discourse indicated by a two-tuple.
        Note, the lower and upper membership functions may be assigned
        in any order to mf1 and mf2.
        """
        if mf1.__class__ != mf2.__class__:
            raise Exception('Both membership functions ' +
                            'must be of the same type.')
        # special care has to be taken if using two Gaussian functions
        # with different means (unlike other cases the MFs will overlap)
        if (mf1.__class__.__name__ == 'Gaussian' and
                mf1.mean != mf2.mean):
            if mf1.height != mf2.height:
                raise Exception('Gaussian functions with different ' +
                                'mean values must have the same height.')
            self.gauss_diff_mean = True
            self.gauss_mean_values = (min(mf1.mean, mf2.mean),
                                      max(mf1.mean, mf2.mean))
            self.gauss_height = mf1.height
        else:
            self.gauss_diff_mean = False
            # check that one mf is a subset of the other
            if not ((mf1.x_min <= mf2.x_min and
                     mf1.x_max >= mf2.x_max and
                     mf1.height >= mf2.height) or
                    (mf1.x_min >= mf2.x_min and
                     mf1.x_max <= mf2.x_max and
                     mf2.height >= mf1.height)):
                raise Exception('One membership function must be a subset ' +
                                'of the other.')
        self.mf1 = mf1
        self.mf2 = mf2
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a two-tuple (lower, upper) of Decimal values.
        """
        if x < self.uod[0] or x > self.uod[1]:
            return (Decimal(0), Decimal(0))
        y1 = self.mf1.calculate_membership(x)
        y2 = self.mf2.calculate_membership(x)
        if (self.gauss_diff_mean and
                x >= self.gauss_mean_values[0] and
                x <= self.gauss_mean_values[1]):
            return min(y1, y2), self.gauss_height
        else:
            return min(y1, y2), max(y1, y2)

    def calculate_alpha_cut_lower(self, alpha):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if self.gauss_diff_mean:
            if self.mf1.mean < self.mf2.mean:
                lower_r = self.mf1.calculate_alpha_cut(alpha)[1]
                lower_l = self.mf2.calculate_alpha_cut(alpha)[0]
        else:
            if self.mf1.x_min < self.mf2.x_min:
                lower_l, lower_r = self.mf2.calculate_alpha_cut(alpha)
            else:
                lower_l, lower_r = self.mf1.calculate_alpha_cut(alpha)
        return max(lower_l, self.uod[0]), min(lower_r, self.uod[1])

    def calculate_alpha_cut_upper(self, alpha):
        """Calculate the alpha-cut of the upper membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if self.gauss_diff_mean:
            if self.mf1.mean < self.mf2.mean:
                cut = (self.mf1.calculate_alpha_cut(alpha)[0],
                       self.mf2.calculate_alpha_cut(alpha)[1])
            else:
                cut = (self.mf2.calculate_alpha_cut(alpha)[0],
                       self.mf1.calculate_alpha_cut(alpha)[1])
        else:
            if self.mf1.x_min < self.mf2.x_min:
                cut = self.mf1.calculate_alpha_cut(alpha)
            else:
                cut = self.mf2.calculate_alpha_cut(alpha)
        if isinstance(cut[0], Decimal):
            # convex cut
            return max(cut[0], self.uod[0]), min(cut[1], self.uod[1])
        else:
            # non-convex cut
            return [(max(subcut[0], self.uod[0]), min(subcut[1], self.uod[1]))
                    for subcut in cut]

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu

    def calculate_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Uses the Karnik Mendel algorithm.
        Returns a dict of two-tuples {z:(l, r)} indicating the
        boundaries  of the type-reduced set at each zlevel.
        """
        l = self._calculate_cos_boundary(right=False)
        r = self._calculate_cos_boundary(right=True)
        return l, r

    def calculate_overall_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Returns the centroid of the centre-of-sets type reduced result.
        """
        l, r = self.calculate_centre_of_sets()
        return (l + r) / Decimal(2)

    def _calculate_cos_boundary(self, right=True):
        """Compute the left or right boundary of the centre of sets.

        Uses the Karnik-Mendel centre-of-sets algorithm.
        right = True computes the right boundary,
        right = False computes the left centroid
        Process steps are as detailed in H. Hagras, "A hierarchical type-2
        fuzzy logic control architecture", IEEE Trans. Fuzz. Sys. 2004
        """
        x_values = [gs.rnd(x)
                    for x in linspace(self.uod[0], self.uod[1],
                                      gs.global_x_disc)]

        def h(x):
            return sum(self.calculate_membership(x)) / Decimal(2)

        def tri(x):
            y1, y2 = self.calculate_membership(x)
            return abs(y1 - y2) / Decimal(2)

        def find_e():
            """Find the index e where y_prime lies between e and e+1."""
            for e in range(len(x_values)-1):
                if x_values[e] <= y_prime and y_prime <= x_values[e+1]:
                    return e

        def get_double_prime():
            """Find the value of y_double_prime using steps 2 and 3."""
            # step 2
            e = find_e()
            # step 3
            top = Decimal(0)
            bottom = Decimal(0)
            for i in range(e+1):
                if right:
                    theta_value = (h(x_values[i]) - tri(x_values[i]))
                else:
                    theta_value = (h(x_values[i]) + tri(x_values[i]))
                top += (x_values[i] * theta_value)
                bottom += theta_value
            for i in range(e+1, len(x_values)):
                if right:
                    theta_value = (h(x_values[i]) + tri(x_values[i]))
                else:
                    theta_value = (h(x_values[i]) - tri(x_values[i]))
                top += (x_values[i] * theta_value)
                bottom += theta_value
            return gs.rnd(top / bottom)

        # step 1
        top = Decimal(0)
        bottom = Decimal(0)
        for x in x_values:
            top += (x * h(x))
            bottom += h(x)
        y_prime = gs.rnd(top / bottom)
        y_double_prime = 0
        while True:
            y_double_prime = get_double_prime()
            # step 4
            if y_prime == y_double_prime:
                return y_double_prime
            else:
                # step 5
                y_prime = y_double_prime


# Module 'general_t2_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is used to create a general type-2 fuzzy set."""

class GeneralT2FuzzySet():
    """Create a zSlices (alpha-plane) based general type-2 fuzzy set."""

    def __init__(self, mf1, mf2, zlevels_total=None, uod=None):
        """Create a general type-2 fuzzy set.

        mf1: first membership function object of lowest zslice
        mf2: second membership function object of lowest zslice
        zlevels_total: total number of zlevels
        uod: the universe of discourse indicated by a two-tuple.
        Note, the lower and upper membership functions may be assiged
        in any order to mf1 and mf2.
        """
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        if zlevels_total is None:
            self.zlevels_total = gs.global_zlevel_disc
        else:
            self.zlevels_total = zlevels_total
        self.zlevel_coords = []
        self._calculate_zlevel_coords()
        # Create the first zSlice to check it's valid,
        # then automatically generate the rest based on this.
        self.zslice_functions = {self.zlevel_coords[0]:
                                 IntervalT2FuzzySet(mf1, mf2)}
        self._generate_zslices()
        # A dict of z:mu pairs for the height of each zslice
        self.zslice_primary_heights = {}
        self._set_zslice_primary_heights()

    def _calculate_zlevel_coords(self):
        """Calculate the zlevel coordinates for each zslice."""
        # it's easier to construct the z-coordinates in reverse order
        self.zlevel_coords = [gs.rnd(Decimal(i)/self.zlevels_total)
                              for i in range(self.zlevels_total, 0, -1)]
        self.zlevel_coords.reverse()

    def _set_zslice_primary_heights(self):
        for z in self.zlevel_coords:
            lower_mf_height = min(self.zslice_functions[z].mf1.height,
                                  self.zslice_functions[z].mf2.height)
            upper_mf_height = max(self.zslice_functions[z].mf1.height,
                                  self.zslice_functions[z].mf2.height)
            self.zslice_primary_heights[z] = (lower_mf_height,
                                              upper_mf_height)

    def _generate_zslices(self):
        """Generate the zslices fuzzy sets."""
        # Find out what parameters the mf uses that will change
        # for each zslice.
        lowest_zslice = self.zslice_functions[self.zlevel_coords[0]]
        arg_names = inspect.getargspec(lowest_zslice.mf1.__init__)[0]
        arg_names.remove('self')
        args1 = [lowest_zslice.mf1.__dict__[var] for var in arg_names]
        args2 = [lowest_zslice.mf2.__dict__[var] for var in arg_names]
        shape = lowest_zslice.mf1.__class__
        # Calculate how much the MFs shift for each zslice.
        # coord_skew defines how much the coordinates of the UMF and
        # LMF change at each zSlice; 1 means no skew and 0 means
        # maximum skew. Think of it as a percentage of how far apart
        # the UMF and LMF will be from each other between each zLevel.
        if self.zlevels_total == 1:
            coord_skew = [1]
        else:
            x = self.zlevels_total - 1
            coord_skew = [Decimal(i)/x for i in range(x, -1, -1)]
        # Create the zslices IT2 FSs for each zlevel.
        # start from index 1 because __init__ already added the lowest zslice
        for z_index in range(1, len(coord_skew)):
            mf1_points = []
            mf2_points = []
            for i in range(len(args1)):
                spread = (args2[i] - args1[i]) / Decimal(2)
                altered_value = gs.rnd(spread * (1 - coord_skew[z_index]))
                mf1_points.append(args1[i] + altered_value)
                mf2_points.append(args2[i] - altered_value)
            try:
                zslice = IntervalT2FuzzySet(shape(*mf1_points),
                                            shape(*mf2_points),
                                            self.uod)
            except:
                #rounding error
                new_rounding = Decimal(str(gs.DECIMAL_ROUNDING*10)[:-1])
                mf1_points = [value.quantize(new_rounding)
                              for value in mf1_points]
                mf2_points = [value.quantize(new_rounding)
                              for value in mf2_points]
                zslice = IntervalT2FuzzySet(shape(*mf1_points),
                                            shape(*mf2_points),
                                            self.uod)
            self.zslice_functions[self.zlevel_coords[z_index]] = zslice

    def validate_zlevel(self, z):
        """Find the closest valid zlevel.

        Checks if the zlevel at z exists. If it exists then return z.
        If not, then return the closest zlevel that encompasses z.
        """
        z = gs.rnd(z)
        if z in self.zlevel_coords:
            return z
        else:
            points = self.zlevel_coords[:]
            if z > max(points):
                raise ZLevelError('zLevel ' + str(z) +
                                  ' is higher than the greatest zLevel at ' +
                                  str(max(points)))
            points.sort()
            for i in points:
                if i > z:
                    return i

    def calculate_membership(self, x, z):
        """Calculate the primary membership of x at the zlevel z."""
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_membership(x)

    def calculate_secondary_membership(self, x, y):
        """Calculate the secondary membership of x at primary membership y."""
        y = Decimal('%.4f' % y)
        for z in sorted(self.zlevel_coords, reverse=True):
            y1, y2 = self.zslice_functions[z].calculate_membership(x)
            if y1 <= y <= y2:
                return z
        return Decimal(0)

    def calculate_alpha_cut_lower(self, alpha, z):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_alpha_cut_lower(alpha)

    def calculate_alpha_cut_upper(self, alpha, z):
        """Calculate the alpha-cut of the upper membership function at z.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_alpha_cut_upper(alpha)

    def calculate_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Uses the Karnik Mendel algorithm.
        Returns a two-tuple indicating the boundaries of the type-reduced set.
        """
        type_reduced_set = {}
        for z in self.zlevel_coords:
            l, r = self.zslice_functions[z].calculate_centre_of_sets()
            type_reduced_set[z] = (l, r)
        return type_reduced_set

    def calculate_overall_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Returns the centroid of the centre-of-sets type reduced result.
        """
        intervals = self.calculate_centre_of_sets()
        top = 0
        bottom = 0
        for z in intervals.keys():
            centre = (intervals[z][0] + intervals[z][1]) / Decimal(2)
            top += z * centre
            bottom += z
        return top / bottom

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu

    def plot_set_3d(self):
        """Plot a 3-dimensional graph of the fuzzy set."""
        plot_sets_3d((self,)) #Edu


# Module 'general_t2_fuzzy_set'
#------------------------------------------------------------------------------
"""This module is for aggregating type-1 fuzzy sets into a type-2 fuzzy set.

The method of aggregation is the same as that used by the
interval aggrement approach, details of which are within
C. Wagner, S. Miller, J. M. Garibaldi, D. T. Anderson and T. C. Havens,
"From Interval-Valued Data to General Type-2 Fuzzy Sets,"
in IEEE Transactions on Fuzzy Systems, vol. 23, no. 2,
pp. 248-269, April 2015.
doi: 10.1109/TFUZZ.2014.2310734
"""

class T2AggregatedFuzzySet():
    """This class is for type-2 Interval Agreement Approach fuzzy sets."""

    def __init__(self, uod=None):
        """Initate a type-2 interval agreement approach fuzzy set."""
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        self.membership_functions = []
        self._total_membership_functions = 0
        self.zlevel_coords = []
        # A dict of z:mu pairs for the height of each zslice
        self.zslice_primary_heights = {}

    def add_membership_function(self, mf):
        """Add a type-1 membership function to the fuzzy set."""
        self.membership_functions.append(mf)
        self._total_membership_functions += 1
        # It's easier to calculate the zlevels in reverse order
        self.zlevel_coords = [gs.rnd(Decimal(i)/self._total_membership_functions)
                              for i in range(self._total_membership_functions,
                                             0, -1)]
        self.zlevel_coords.reverse()
        self._calculate_zslice_primary_heights()

    def _calculate_zslice_primary_heights(self):
        """Calculate the primary height of each zslice."""
        self.zslice_primary_heights = dict((z, (0, 0))
                                           for z in self.zlevel_coords)
        for z in self.zlevel_coords:
            self.zslice_primary_heights[z] = (
                    0, max([self.calculate_membership(x, z)[1]
                            for x in linspace(self.uod[0], self.uod[1],
                                              gs.global_x_disc)]))

    def validate_zlevel(self, z):
        """Find the closest valid zlevel.

        Checks if the zlevel at z exists. If it exists then return z.
        If not, then return the closest zlevel that encompasses z.
        """
        # Default to an interval type-2 fuzzy set if no zlevel is given.
        if z is None:
            return min(self.zlevel_coords)
        z = gs.rnd(z)
        if z in self.zlevel_coords:
            return z
        else:
            points = self.zlevel_coords[:]
            if z > max(points):
                raise ZLevelError('zLevel ' + str(z) +
                                  ' is higher than the greatest zLevel at ' +
                                  str(max(points)))
            points.sort()
            for i in points:
                if i > z:
                    return i

    def calculate_membership(self, x, z=None):
        """Calculate the primary membership of x at the zlevel z.

        For an interval type-2 fuzzy set, leave z as None.
        """
        z = self.validate_zlevel(z)
        x = Decimal(str(x))
        y_values = sorted([mf.calculate_membership(x)
                           for mf in self.membership_functions])
        y_max = 0
        for y in y_values:
            if self.calculate_secondary_membership(x, y) >= z:
                y_max = y
        # lower membership is 0 as a type-2 IAA doesn't strictly have a
        # proper lower membership function.
        return Decimal(0), y_max

    def calculate_secondary_membership(self, x, y):
        """Calculate the secondary membership value for the given x and y."""
        if y == 0:
            return 0
        x = Decimal(str(x))
        y = Decimal(str(y))
        mfs_within = 0
        for mf in self.membership_functions:
            if y <= mf.calculate_membership(x):
                mfs_within += 1
        return gs.rnd(Decimal(mfs_within) / self._total_membership_functions)

    def calculate_alpha_cut_upper(self, alpha, z=None):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        For an interval type-2 fuzzy set, leave z as None.
        Returns a list containing two-tuples
        (a list of cuts is always given as any alpha-cut may be non-convex)
        """
        z = self.validate_zlevel(z)
        if alpha > self.zslice_primary_heights[z][1]:
            raise AlphaCutError('alpha level', alpha, 'is above max y level',
                                self.zslice_primary_heights[z][1])
        x_values = []
        for mf in self.membership_functions:
            x_values.extend(mf.intervals.singleton_keys())
        x_values = sorted(x_values)
        test_values = x_values[:]
        # add inbetween values to spot discontinous intervals
        for i in range(len(x_values)-1):
            test_values.insert((i+1) * 2-1,
                               ((x_values[i+1] + x_values[i]) / 2))
        alpha_intervals = []
        current_interval = []
        for x in test_values:
            if self.calculate_membership(x, z)[1] >= alpha:
                current_interval.append(Decimal(str(x)))
            else:
                if len(current_interval) != 0:
                    alpha_intervals.append((Decimal(current_interval[0]),
                                            Decimal(current_interval[-1])))
                    current_interval = []
        if len(current_interval) != 0:
            alpha_intervals.append((Decimal(current_interval[0]),
                                    Decimal(current_interval[-1])))
        return alpha_intervals

    def calculate_alpha_cut_lower(self, alpha, z):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        """
        # Type-2 IAA doesn't strictly have a proper lower membership function.
        return (Decimal(0), Decimal(0))

    def calculate_centroid(self):
        """Calculate the centroid of the fuzzy set.

        Calculate the centroid of each zslice and take the weighted average.
        """
        result_top = 0
        result_bottom = 0
        for z in self.zlevel_coords:
            slice_top = 0
            slice_bottom = 0
            for x in linspace(self.uod[0], self.uod[1], gs.global_x_disc):
                x = Decimal(str(x))
                mu = self.calculate_membership(x, z)[1]
                slice_top += Decimal(x) * mu
                slice_bottom += mu
            result_top += z * (slice_top / slice_bottom)
            result_bottom += z
        return gs.rnd(result_top / result_bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        plot_sets((self,), filename) #Edu

    def plot_set_3d(self):
        """Plot a graph of the fuzzy set."""
        plot_sets_3d((self,)) #Edu


###############################################################################
"""Membership Functions of the fuzzy sets"""
###############################################################################


# Module 'trapezoidal'
#------------------------------------------------------------------------------
"""This module is used to create trapezoidal membership functions."""

class Trapezoidal():
    """Create a trapezoidal membership function."""

    def __init__(self, x_min, x_top_left, x_top_right, x_max, height=1):
        """Set the Trapezoidal membership function.

        x_min_base: bottom left coordinate
        x_top_left: top left coordinate
        x_top_right: top right coordinate
        x_max_base: bottom right coordinate
        height: scale the maximum membership value
        """
        if height <= 0 or height > 1:
            raise Exception('height must be within the range (0, 1]')
        if not (x_min <= x_top_left <= x_top_right <= x_max):
            raise Exception('Values must be ordered such that ' +
                            'x_min_base <= x_top_left <= ' +
                            'x_top_right <= x_max_base')
        # First two variables are renamed so that every membership
        # function has x_min and x_max defining the boundaries.
        self.x_min = Decimal(str(x_min))
        self.x_max = Decimal(str(x_max))
        self.x_top_left = Decimal(str(x_top_left))
        self.x_top_right = Decimal(str(x_top_right))
        self.height = Decimal(str(height))

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        x = Decimal(str(x))
        if self.x_min <= x and x <= self.x_top_left:
            try:
                return gs.rnd(self.height *
                              ((x - self.x_min) /
                                (self.x_top_left - self.x_min)))
            except:# ZeroDivisionError, DivisionByZero:
                # the base and top are the same resulting in a vertical line
                return self.height
        elif self.x_top_left <= x and x <= self.x_top_right:
            return self.height
        elif self.x_top_right <= x and x <= self.x_max:
            try:
                return gs.rnd(self.height *
                              ((self.x_max - x) /
                                (self.x_max - self.x_top_right)))
            except:# ZeroDivisionError, DivisionByZero:
                # the base and top are the same resulting in a vertical line
                return self.height
        else:
            return Decimal(0)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        alpha = Decimal(str(alpha))
        if alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        left_point = (self.x_min +
                      (self.x_top_left - self.x_min) *
                      (alpha / self.height))
        right_point = (self.x_max +
                        (self.x_top_right - self.x_max) *
                        (alpha / self.height))
        return (gs.rnd(left_point), gs.rnd(right_point))

    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        self.x_min += x
        self.x_max += x
        self.x_top_left += x
        self.x_top_right += x


# Module 'triangular'       
#------------------------------------------------------------------------------
"""This module is used to create triangular membership functions."""

class Triangular(Trapezoidal):
    """Create a triangular membership function."""

    def __init__(self, x_min, centre, x_max, height=1):
        """Create a triangular membership function.

        x_min: bottom left coordinate
        centre: x coordinate at the peak of the triangle
        x_max: bottom right coordinate
        height: highest membership at the centre.
        """
        self.centre = Decimal(centre)
        Trapezoidal.__init__(self, x_min, centre, centre, x_max, height)


# Module 'gaussian'
#------------------------------------------------------------------------------
"""This module is used to create Gaussian membership functions."""
        
class Gaussian():
    """Create a Gaussian distribution."""

    def __init__(self, mean, std_dev, height=1):
        """Set the Gaussian membership function.

        height scales the height of the mean.
        """
        if height <= 0 or height > 1:
            raise Exception('height must be within the range (0, 1]')
        if std_dev <= 0:
            raise Exception('std_dev must be greater than 0')
        self.height = Decimal(str(height))
        self.std_dev = Decimal(str(std_dev))
        self.mean = Decimal(str(mean))
        self.x_min = None
        self.x_max = None
        self._set_function_end_points()

    def _set_function_end_points(self):
        """Set self.x_min and self.x_max.

        A gaussian function never approaches zero but it's helpful to
        define the end points of the function for various calculations.
        Typically, the spread is std_dev * 4.
        """
        self.x_min = self.mean - (self.std_dev * 4)
        self.x_max = self.mean + (self.std_dev * 4)

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        x = Decimal(str(x))
        if x < self.x_min or x > self.x_max:
            return Decimal(0)
        y = self.height * power(
                    Decimal(e),
                    Decimal('-0.5') * (power(
                                        (x - self.mean) / self.std_dev,
                                        Decimal(2))))
        return gs.rnd(y)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        # set alpha as float because log won't work with Decimal
        part = Decimal(sqrt(abs(log(float(alpha)) / 0.5))) * self.height
        left_point = (self.std_dev * -part) + self.mean
        right_point = (self.std_dev * part) + self.mean
        return (gs.rnd(left_point), gs.rnd(right_point))

    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        self.mean += x
        self._set_function_end_points()


# Module 'interval_dict' (for 'iaa')        
#------------------------------------------------------------------------------
"""This module is used to create an interval-based dict."""

class IntervalDict(object):
    """This class stores a dict in which the keys are intervals."""

    def __init__(self, overwrite_with_max=True):
        """Initiate the interval dict.

        When overwrite_with_max = True:
        If a key has been assigned multiple values the max is returned.
        When overwrite_with_max = False:
        If a key has been assigned multiple values the sum is returned.
        """
        # store a list of [(interval, value)] mappings
        self._interval_value_pairs = []
        self.overwrite_with_max = overwrite_with_max

    def __setitem__(self, _slice, _value):
        """Assign _value to the continuous _slice.

        _slice must be a slice; e.g. [1:3]
        _value must be numerical
        """
        if not isinstance(_slice, slice):
            raise Exception('The key must be a slice object.')
        if _slice.start is None or _slice.stop is None:
            raise Exception('Both end points must be given')
        self._interval_value_pairs.append(([_slice.start, _slice.stop],
                                            _value))

    def __getitem__(self, _point):
        """Return the value of the singleton _point."""
        values = [0]
        for interval, value in self._interval_value_pairs:
            if self._is_within_interval(interval, _point):
                values.append(value)
        if self.overwrite_with_max:
            return max(values)
        else:
            return sum(values)

    def singleton_keys(self):
        """Return the list of key values as singletons."""
        item_list = []
        for interval, value in self._interval_value_pairs:
            if interval[0] not in item_list:
                item_list.append(interval[0])
            if interval[1] not in item_list:
                item_list.append(interval[1])
        return item_list

    def keys(self):
        """Return the list of intervals used as keys."""
        item_list = []
        for interval, value in self._interval_value_pairs:
            item_list.append(interval)
        return item_list

    def values(self):
        """Return the list of values stored."""
        value_list = []
        for interval, value in self._interval_value_pairs:
            value_list.append(value)
        return value_list

    def _is_within_interval(self, interval, point):
        """Check if point is within the given interval."""
        if isinstance(interval[0], tuple):
            for s in interval:
                if point >= s[0] and point <= s[1]:
                    return True
        else:
            if point >= interval[0] and point <= interval[1]:
                return True
        return False


# Module 'iaa'        
#------------------------------------------------------------------------------
"""This module is for applying the type-1 Interval Agreement Approach.

Details of the interval agreement approach are within
C. Wagner, S. Miller, J. M. Garibaldi, D. T. Anderson and T. C. Havens,
"From Interval-Valued Data to General Type-2 Fuzzy Sets,"
in IEEE Transactions on Fuzzy Systems, vol. 23, no. 2,
pp. 248-269, April 2015.
doi: 10.1109/TFUZZ.2014.2310734
"""

class IntervalAgreementApproach():
    """This class type-1 interval agreement approach membership function."""

    def __init__(self, normalise=False):
        """Create a membership function by the interval agreement approach."""
        self.normalise = normalise
        self.intervals = IntervalDict(overwrite_with_max=False)
        self._total_intervals = 0
        self._largest_value = 0  # largest value in the dict after summing
        self.height = 1

    def add_interval(self, interval):
        """Add an interval to the fuzzy set."""
        self.intervals[interval[0]:interval[1]] = 1
        self._total_intervals += 1
        self._largest_value = max([self.intervals[point] for point in
                                    self.intervals.singleton_keys()])
        self.height = self._largest_value / Decimal(self._total_intervals)

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        if self.normalise:
            mu = Decimal(self.intervals[x]) / self._largest_value
        else:
            mu = Decimal(self.intervals[x]) / self._total_intervals
        return gs.rnd(mu)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a list containing two-tuples
        (a list of cuts is always given as any alpha-cut may be non-convex)
        """
        if not self.normalise and alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        x_values = sorted(self.intervals.singleton_keys())
        test_values = x_values[:]
        # add inbetween values to spot discontinous intervals
        for i in range(len(x_values)-1):
            test_values.insert((i+1) * 2-1,
                                ((x_values[i+1] + x_values[i]) / 2))
        alpha_intervals = []
        current_interval = []
        for x in test_values:
            if self.calculate_membership(x) >= alpha:
                current_interval.append(Decimal(x))
            else:
                if len(current_interval) != 0:
                    alpha_intervals.append((Decimal(current_interval[0]),
                                            Decimal(current_interval[-1])))
                    current_interval = []
        if len(current_interval) != 0:
            alpha_intervals.append((Decimal(current_interval[0]),
                                    Decimal(current_interval[-1])))
        return alpha_intervals


###############################################################################
"""Generate fuzzy sets from data."""
###############################################################################


# Module 'generate_fuzzy_sets'       
#------------------------------------------------------------------------------

def _calculate_membership_values(data):
    """Calculate membership values for each data point.

    Each values membership is calculated as a proportion of how often
    it appears within the list of data points.
    """
    scale = Decimal(len(data))
    if gs.normalise_generated_sets:
        counter = Counter(data)
        scale = Decimal(counter.most_common()[0][1])
    else:
        scale = Decimal(len(data))
    return dict((gs.rnd(p), gs.rnd(data.count(p) / scale)) for p in set(data))


def generate_gaussian_t1_fuzzy_set(data):
    """Create a Gaussian distributed type-1 fuzzy set from the given data."""
    # mean and std won't work for Decimal data; ensure floats.
    float_data = [float(d) for d in data]
    return FuzzySet(Gaussian(Decimal(mean(float_data)), #Edu
                              Decimal(std(float_data))))


def generate_gaussian_t2_fuzzy_set(data):
    """Create a Gaussian distributed type-1 fuzzy set from the given data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        # mean and std won't work for Decimal data; ensure floats.
        float_subset = [float(d) for d in t1_subset]
        fs.add_membership_function(Gaussian(Decimal(mean(float_subset)),
                                            Decimal(std(float_subset))))
    return fs


def generate_discrete_t1_fuzzy_set(data):
    """Create a discrete type-1 fuzzy set from the given data."""
    return DiscreteT1FuzzySet(_calculate_membership_values(data)) #Edu


def generate_polling_t1_fuzzy_set(data):
    """Create a type-1 fuzzy set with interpolation from the given data."""
    return PollingT1FuzzySet(_calculate_membership_values(data)) #Edu


def generate_polling_t2_fuzzy_set(data):
    """Create a type-1 fuzzy set with interpolation from the given data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        fs.add_membership_function(PollingT1FuzzySet(
                _calculate_membership_values(t1_subset)))
    return fs


def generate_iaa_t1_fuzzy_set(data):
    """Create a type-1 interval agreement approach set from interval data."""
    mf = IntervalAgreementApproach(gs.normalise_generated_sets)
    for d in data:
        mf.add_interval(d)
    return FuzzySet(mf) #Edu


def generate_iaa_t2_fuzzy_set(data):
    """Create a type-2 interval agreement approach set from interval data."""
    fs = T2AggregatedFuzzySet()
    for t1_subset in data:
        fs.add_membership_function(generate_iaa_t1_fuzzy_set(t1_subset))
    return fs