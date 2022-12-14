from numpy import linspace, e, sqrt, linalg, corrcoef #Edu
from decimal import Decimal
import functools
from copy import deepcopy
from scipy import integrate
from fuzzy_sets import AlphaCutError, ZLevelError #Edu
import global_settings as gs #Edu
from fuzzy_sets import FuzzySet, DiscreteT1FuzzySet #'distance_it2', 'similarity_gt2'
from collections import defaultdict #'similarity_gt2'


#------------------------------------------------------------------------------
"""This module contains state-of-the-art measures."""

def ePL_KRLS(x, v):
    return (1 - ((linalg.norm(x - v))/x.shape[0]))

def ePL_KRLS_DISCO(x, v):
    return (1 - ((linalg.norm(x - v))/x.shape[0]))*((corrcoef(v, x)[0,1] + 1)/2)


# Module 'simirarity_t1'
#------------------------------------------------------------------------------
"""This module contains similarity measures for type-1 fuzzy sets."""

def pappis1(fs1, fs2):
    """Based on the maximum distance between membership values."""
    dist = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist = max(dist, abs(y1 - y2))
    return gs.rnd(1 - dist)


def pappis2(fs1, fs2):
    """The ratio between the negation and addition of membership values."""
    dist1 = 0
    dist2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist1 += abs(y1 - y2)
        dist2 += y1 + y2
    return gs.rnd(1 - (dist1 / dist2))


def pappis3(fs1, fs2):
    """Based on the average difference between membership values."""
    dist = 0
    n = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist += abs(y1 - y2)
        n += 1
    return gs.rnd(1 - (dist / n))


def jaccard(fs1, fs2):
    """Ratio between the intersection and union of the fuzzy sets."""
    sim1 = 0
    sim2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim1 += min(y1, y2)
        sim2 += max(y1, y2)
    return gs.rnd(sim1 / sim2)


def dice(fs1, fs2):
    """Based on the ratio between the intersection and cardinality."""
    sim1 = 0
    sim2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim1 += Decimal(2) * (min(y1, y2))
        sim2 += y1 + y2
    return gs.rnd(sim1 / sim2)


def zwick(fs1, fs2):
    """The maximum membership of the intersection of the fuzzy sets."""
    sim = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim = max(sim, min(y1, y2))
    return gs.rnd(sim)


def chen(fs1, fs2):
    """Ratio between the product of memberships and the cardinality."""
    top = 0
    fs1_squares = 0
    fs2_squares = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        top += (y1 * y2)
        fs1_squares += (y1 * y1)
        fs2_squares += (y2 * y2)
    return gs.rnd(top / max(fs1_squares, fs2_squares))


def vector(fs1, fs2):
    """Vector similarity based on the distance and similarity of shapes."""
    x_min = min(fs1.membership_function.x_min, fs2.membership_function.x_min)
    x_max = max(fs1.membership_function.x_max, fs2.membership_function.x_max)
    r = Decimal(4) / (x_max - x_min)
    dist = fs1.calculate_centroid() - fs2.calculate_centroid()
    # temporarily align the centroid of fs2 with fs1 to compare shapes
    fs2.membership_function.shift_membership_function(dist)
    sim1 = jaccard(fs1, fs2)
    # put the fuzzy set fs2 back to where it was
    fs2.membership_function.shift_membership_function(-dist)
    sim2 = pow(Decimal(e), -r * abs(dist))
    return gs.rnd(sim1 * sim2)


# Module 'similarity_gt2'
#------------------------------------------------------------------------------
"""This module contains similarity measures for general type-2 fuzzy sets."""

def jaccard_gt2(fs1, fs2): #Edu
    """Calculate the weighted average of the jaccard similarity on zslices."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        j_top = 0
        j_bottom = 0
        for x in gs.get_x_points():
            fs1_l, fs1_u = fs1.calculate_membership(x, z)
            fs2_l, fs2_u = fs2.calculate_membership(x, z)
            j_top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
            j_bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
        if j_top == 0 and j_bottom == 0:
            return 0
        top += z * (j_top / j_bottom)
        bottom += z
    if top == 0 and bottom == 0:
        return 0
    return gs.rnd(top / bottom)


def _zslice_jaccard(fs1, fs2, z):
    """Measure the jaccard similarity between two zslices."""
    j_top = 0
    j_bottom = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x, z)
        fs2_l, fs2_u = fs2.calculate_membership(x, z)
        j_top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
        j_bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    if j_bottom == 0:
        return Decimal(0)
    return gs.rnd(j_top / j_bottom)


def zhao_crisp(fs1, fs2):
    """Like jaccard, but the result is the standard average; not weighted."""
    sim = 0
    for z in gs.get_z_points():
        sim += _zslice_jaccard(fs1, fs2, z)
    return gs.rnd(sim / gs.global_zlevel_disc)


def hao_fuzzy(fs1, fs2):
    """Calculate the jaccard similarity given as type-1 fuzzy set."""
    sim_fs = defaultdict(int)
    for z in gs.get_z_points():
        sim = _zslice_jaccard(fs1, fs2, z)
        sim_fs[sim] = max(sim_fs[sim], z)
    fs = DiscreteT1FuzzySet(sim_fs)
    return fs


def hao_crisp(fs1, fs2):
    """Calculate the centroid of hao_fuzzy(fs1, fs2)."""
    sim_fs = hao_fuzzy(fs1, fs2)
    top = 0
    bottom = 0
    for y in sim_fs.points.keys():
        z = sim_fs.points[y]
        y = y.quantize(Decimal(10) ** -2)
        top += y * z
        bottom += z
    return top / bottom


def yang_lin(fs1, fs2):
    """Calculate the average jaccard similarity for each vertical slice."""
    result = 0
    n = 0
    for x in gs.get_x_points():
        top = 0
        bottom = 0
        y_points = gs.get_y_points()
        for y in y_points:
            z1 = fs1.calculate_secondary_membership(x, y)
            z2 = fs2.calculate_secondary_membership(x, y)
            top += min(y*z1, y*z2)
            bottom += max(y*z1, y*z2)
        top /= sum(y_points)
        bottom /= sum(y_points)
        if bottom > 0:  # if z values were present
            result += top/bottom
            n += 1
    if n == 0:
        return 0
    return gs.rnd(result / n)


def mohamed_abdaala(fs1, fs2):
    """Based on the the jaccard similarity for each vertical slice."""
    result = 0
    for x in gs.get_x_points():
        fs1_slices = 0
        fs2_slices = 0
        for y in gs.get_y_points():
            fs1_slices += 1 - y * fs1.calculate_secondary_membership(x, y)
            fs2_slices += 1 - y * fs2.calculate_secondary_membership(x, y)
        result += min(fs1_slices, fs2_slices) / max(fs1_slices, fs2_slices)
    return gs.rnd(result / gs.global_x_disc)


def hung_yang(fs1, fs2):
    """Based on the Hausdorff distance between vertical slice pairs."""
    distance = Decimal(0)
    for x in gs.get_x_points():
        top = Decimal(0)
        bottom = Decimal(0)
        for z in gs.get_z_points():
            y1_l, y1_u = fs1.calculate_membership(x, z)
            y2_l, y2_u = fs2.calculate_membership(x, z)
            hausdorff = max(abs(y1_l - y2_l), abs(y1_u - y2_u))
            top += hausdorff * z
            bottom += z
        try:
            distance += top / bottom
        except ZeroDivisionError:
            pass
    return gs.rnd(1 - (distance / gs.global_x_disc))


def wu_mendel(fs1, fs2):
    """Geometric approach."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        for x in gs.get_x_points():
            fs1_l, fs1_u = fs1.calculate_membership(x, z)
            fs2_l, fs2_u = fs2.calculate_membership(x, z)
            top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
            bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    return gs.rnd(top / bottom)


# Module 'similarity_it2'
#------------------------------------------------------------------------------
"""This module contains similarity measures for interval type-2 fuzzy sets."""

def zeng_li(fs1, fs2):
    """Based on the average distance between the membership values."""
    result = 0
    # restrict the universe of discourse because the
    # measure doesn't follow the property overlapping.
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        result += abs(fs1_l - fs2_l) + abs(fs1_u - fs2_u)
    result /= (2 * gs.global_x_disc)
    result = 1 - result
    return gs.rnd(result)


def gorzalczany(fs1, fs2):
    """Based on the highest membership where the fuzzy sets overlap."""
    max_of_min_lower_values = 0
    max_of_min_upper_values = 0
    max_lower_fs1 = 0
    max_upper_fs1 = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        max_of_min_lower_values = max(max_of_min_lower_values,
                                      min(fs1_l, fs2_l))
        max_of_min_upper_values = max(max_of_min_upper_values,
                                      min(fs1_u, fs2_u))
        max_lower_fs1 = max(max_lower_fs1, fs1_l)
        max_upper_fs1 = max(max_upper_fs1, fs1_u)
    measure1 = gs.rnd(max_of_min_lower_values / max_lower_fs1)
    measure2 = gs.rnd(max_of_min_upper_values / max_upper_fs1)
    return min(measure1, measure2), max(measure1, measure2)


def bustince(fs1, fs2, t_norm_min=True):
    """Based on the inclusion of one fuzzy set within the other."""
    yl_ab = 1
    yl_ba = 1
    yu_ab = 1
    yu_ba = 1
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        yl_ab = min(yl_ab, min(1 - fs1_l + fs2_l,
                               1 - fs1_u + fs2_u))
        yl_ba = min(yl_ba, min(1 - fs2_l + fs1_l,
                               1 - fs2_u + fs1_u))
        yu_ab = min(yu_ab, max(1 - fs1_l + fs2_l,
                               1 - fs1_u + fs2_u))
        yu_ba = min(yu_ba, max(1 - fs2_l + fs1_l,
                               1 - fs2_u + fs1_u))
    return min(yl_ab, yl_ba), min(yu_ab, yu_ba)


def jaccard_it2(fs1, fs2): #Edu
    """Ratio between the intersection and union of the fuzzy sets."""
    top = 0
    bottom = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
        bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    return gs.rnd(top / bottom)


def zheng(fs1, fs2):
    """Similar to jaccard; based on the intersection and union of the sets."""
    top_a = 0
    top_b = 0
    bottom_a = 0
    bottom_b = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        top_a += min(fs1_u, fs2_u)
        top_b += min(fs1_l, fs2_l)
        bottom_a += max(fs1_u, fs2_u)
        bottom_b += max(fs1_l, fs2_l)
    return gs.rnd(Decimal('0.5') * ((top_a / bottom_a) + (top_b / bottom_b)))


def vector_it2(fs1, fs2): #Edu
    """Vector similarity based on the distance and similarity of shapes."""
    fs1_c = fs1.calculate_overall_centre_of_sets()
    fs2_c = fs2.calculate_overall_centre_of_sets()
    dist = fs1_c - fs2_c
    # temporarily align the centroid of fs2 with fs1 to compare shapes
    fs2.mf1.shift_membership_function(dist)
    fs2.mf2.shift_membership_function(dist)
    # find out the support of the union of the fuzzy sets
    # and weight the absolute distance by this support
    x_min = min(max(fs1.uod[0], fs1.mf1.x_min),
                max(fs1.uod[0], fs1.mf2.x_min),
                max(fs2.uod[0], fs2.mf1.x_min),
                max(fs2.uod[0], fs2.mf2.x_min))
    x_max = max(min(fs1.uod[1], fs1.mf1.x_max),
                min(fs1.uod[1], fs1.mf2.x_max),
                min(fs2.uod[1], fs2.mf1.x_max),
                min(fs2.uod[1], fs2.mf2.x_max))
    r = Decimal(4) / (x_max - x_min)
    proximity = pow(Decimal(e), -r * abs(dist))
    shape_difference = jaccard(fs1, fs2)
    # put the fuzzy set fs2 back to where it was
    fs2.mf1.shift_membership_function(-dist)
    fs2.mf2.shift_membership_function(-dist)
    return gs.rnd(shape_difference * proximity)


# Module 'distance_t1'
#------------------------------------------------------------------------------
"""This module contains distance measures for type-1 fuzzy sets."""

def _hausdorff(fs1, fs2, alpha):
    """Calculate the Hausdorff distance at the given alpha cut."""
    fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
    fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
    return max(abs(fs1_min - fs2_min), abs(fs1_max - fs2_max))


def _minkowski_r1(fs1, fs2, alpha):
    """Calculate the minkwoski distance where r = 1 at the given alpha cut."""
    fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
    fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
    return abs(fs1_min - fs2_min), abs(fs1_max - fs2_max)


def centroid(fs1, fs2):
    """Calculate the diffefence between the centroids of the fuzzy sets."""
    return abs(fs1.calculate_centroid() - fs2.calculate_centroid())


def ralescu1(fs1, fs2):
    """Calculate the average Hausdorff distance over all alpha-cuts."""
    def haus(alpha):
        return _hausdorff(fs1, fs2, alpha)
    a, b = integrate.quad(haus, 0, 1, epsabs=gs.abs_e, epsrel=gs.rel_e, limit=gs.lim) #Edu
    return gs.rnd(a)


def ralescu2(fs1, fs2):
    """Calculate the maximum Hausdorff distance over all alpha-cuts."""
    result = Decimal(0)
    for alpha in gs.get_y_points():
        dist = _hausdorff(fs1, fs2, alpha)
        result = max(result, dist)
    return gs.rnd(result)


def chaudhuri_rosenfeld(fs1, fs2):
    """Calculate the weighted average of Hausdorff distances."""
    top = Decimal(0)
    bottom = Decimal(0)
    for alpha in gs.get_y_points():
        dist = _hausdorff(fs1, fs2, alpha)
        top += alpha * dist
        bottom += alpha
    return gs.rnd(top / bottom)


def chaudhuri_rosenfeld_nn(fs1, fs2, e=0.5):
    """Calculate the weighted average of Hausdorff distances for non-normal."""
    def _normalise(fs):
        if fs.__class__.__name__ == 'DiscreteT1FuzzySet':
            fs.points = dict((x, y/fs.height) for x, y in fs.points.items())
        else:
            fs.membership_function.height = 1
    dist1 = 0
    n = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist1 += abs(y1 - y2)
        n += 1
    # normalise the fuzzy sets, use deepcopy to originals aren't altered
    fs1n = deepcopy(fs1)
    fs2n = deepcopy(fs2)
    _normalise(fs1n)
    _normalise(fs2n)
    dist2 = chaudhuri_rosenfeld(fs1n, fs2n)
    return gs.rnd(dist2 + (Decimal(e) * (dist1 / n)))


def _grzegorzewski_non_inf_p(fs1, fs2, p=2):
    """Use for Grzegorzewski distance where 1 <= p < infty."""
    def get_left_dist(alpha):
        fs1_l, fs1_u = fs1.calculate_alpha_cut(alpha)
        fs2_l, fs2_u = fs2.calculate_alpha_cut(alpha)
        return pow(fs1_l - fs2_l, p)

    def get_right_dist(alpha):
        fs1_l, fs1_u = fs1.calculate_alpha_cut(alpha)
        fs2_l, fs2_u = fs2.calculate_alpha_cut(alpha)
        return pow(fs1_u - fs2_u, p)
    left_dist, b = integrate.quad(get_left_dist, 0, 1, epsabs=gs.abs_e, epsrel=gs.rel_e, limit=gs.lim) #Edu
    right_dist, b = integrate.quad(get_right_dist, 0, 1, epsabs=gs.abs_e, epsrel=gs.rel_e, limit=gs.lim) #Edu
    return Decimal(left_dist), Decimal(right_dist)


def _grzegorzewski_inf_p(fs1, fs2):
    """Use for Grzegorzewski distance where p is infinity."""
    left_dist = Decimal(0)
    right_dist = Decimal(0)
    for alpha in gs.get_y_points():
        l, r = _minkowski_r1(fs1, fs2, alpha)
        left_dist = max(left_dist, l)
        right_dist = max(right_dist, r)
    return left_dist, right_dist


def grzegorzewski_non_inf_pq(fs1, fs2, p=2, q=0.5):
    """Grzegorzewski distance where 1 <= p < infty and q is used.

    q is used to weight the distance at alpha cuts.
    (1-q) weight for left distance, (q) weight for right distance.
    """
    p = Decimal(p)
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p)
    left_dist = (1 - Decimal(q)) * left_dist
    right_dist = 1 * right_dist
    distance = (left_dist + right_dist) ** (1 / p)
    return gs.rnd(distance)


def grzegorzewski_non_inf_p(fs1, fs2, p=2):
    """Grzegorzewski distance where 1 <= p < infty and q is not used."""
    p = Decimal(p)
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p)
    left_dist = left_dist ** (1 / p)
    right_dist = right_dist ** (1 / p)
    return gs.rnd(max(left_dist, right_dist))


def grzegorzewski_inf_q(fs1, fs2, q=0.5):
    """Grzegorzewski distance where p is infinity and q is used.

    q is used to weight the distance at alpha cuts.
    (1-q) weight for left distance, (q) weight for right distance.
    """
    q = Decimal(q)
    left_dist, right_dist = _grzegorzewski_inf_p(fs1, fs2)
    left_dist = (1 - q) * left_dist
    right_dist = q * right_dist
    return gs.rnd(left_dist + right_dist)


def grzegorzewski_inf(fs1, fs2):
    """Grzegorzewski distance where p is infinity and q is not used."""
    left_dist, right_dist = _grzegorzewski_inf_p(fs1, fs2)
    return gs.rnd(max(left_dist, right_dist))


def ban(fs1, fs2):
    """Minkowski based distance."""
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p=2)
    return gs.rnd((left_dist + right_dist) ** Decimal('0.5'))


def allahviranloo(fs1, fs2, c=0.5, f=lambda a: a):
    """Distance based on the average width and centre of the fuzzy sets."""
    def average_value(fs, alpha):
        l, r = fs.calculate_alpha_cut(alpha)
        return (Decimal(c) * l) + (Decimal(c) * r)

    def width(fs, alpha):
        l, r = fs.calculate_alpha_cut(alpha)
        return (r - l) * Decimal(f(alpha))
    fs1_average, b = integrate.quad(lambda y: average_value(fs1, y),
                                    0, 1, limit=50)
    fs2_average, b = integrate.quad(lambda y: average_value(fs2, y),
                                    0, 1, limit=50)
    fs1_width, b = integrate.quad(lambda y: width(fs1, y), 0, 1, limit=50)
    fs2_width, b = integrate.quad(lambda y: width(fs2, y), 0, 1, limit=50)
    centre_difference = (Decimal(fs1_average) - Decimal(fs2_average)) ** 2
    width_difference = (Decimal(fs1_width) - Decimal(fs2_width)) ** 2
    return gs.rnd(sqrt(centre_difference + width_difference))


def yao_wu(fs1, fs2):
    """Calculate the average Minkowski (r=1) distance."""
    def dist(alpha):
        fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
        fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
        diff = fs1_min + fs1_max - fs2_min - fs2_max
        return diff
    a, b = integrate.quad(dist, 0, 1, epsabs=gs.abs_e, epsrel=gs.rel_e, limit=gs.lim) #Edu
    return gs.rnd(Decimal('0.5') * Decimal(a))


def _get_directional_distance(cut1, cut2):
    """Calcualte the directional minkwoski distance between cuts."""
    # Check they're continuous (i.e. not a tuple of tuples)
    if isinstance(cut1[0], Decimal) and isinstance(cut2[0], Decimal):
        return (cut2[0] + cut2[1] - cut1[0] - cut1[1]) / 2
    else:
        # Attempt to flatten the alpha cuts in case they are non-convex
        try:
            cut1 = [val for sublist in cut1 for val in sublist]
        except TypeError:
            pass
        try:
            cut2 = [val for sublist in cut2 for val in sublist]
        except TypeError:
            pass
        diffs = []
        # Take the lists in pairs
        for i in zip(cut1[0::2], cut1[1::2]):
            for j in zip(cut2[0::2], cut2[1::2]):
                diffs.append((j[0] + j[1] - i[0] - i[1]) / 2)
        return functools.reduce(lambda x, y: x + y, diffs) / len(diffs)


def mcculloch(fs1, fs2):
    """Calculate the weighted Minkowski (r=1) directional distance."""
    top = 0
    bottom = 0
    diff = 0
    for alpha in gs.get_y_points():
        try:
            cut1 = fs1.calculate_alpha_cut(alpha)
            cut2 = fs2.calculate_alpha_cut(alpha)
            diff = _get_directional_distance(cut1, cut2)
            top += alpha * diff
            bottom += alpha
        except AlphaCutError:
            pass
    return gs.rnd(top / bottom)


# Module 'distance_gt2'
#------------------------------------------------------------------------------
"""This module contains distance measures for general type-2 fuzzy sets."""

def _zslice_distance(fs1, fs2, z):
    """Calcualte the directional minkwoski distance between zslices."""
    top = bottom = Decimal(0)
    count = True
    z1 = fs1.validate_zlevel(z)
    z2 = fs2.validate_zlevel(z)
    mfs = ((fs1.calculate_alpha_cut_lower, fs2.calculate_alpha_cut_lower),
           (fs1.calculate_alpha_cut_upper, fs2.calculate_alpha_cut_upper))
    mf_maxes = ((fs1.zslice_primary_heights[z1][0],
                 fs2.zslice_primary_heights[z2][0]),
                (fs1.zslice_primary_heights[z1][1],
                 fs2.zslice_primary_heights[z2][1]))
    if (fs1.zslice_primary_heights[z1][1] == 0 or
            fs2.zslice_primary_heights[z1][1] == 0):
        raise ZLevelError('Empty zSlice.')
    for mf_index in (0, 1):
        mf_pair = mfs[mf_index]
        for alpha in gs.get_y_points():
            try:
                count = True
                fs1_cut = mf_pair[0](alpha, z)
                fs2_cut = mf_pair[1](alpha, z)
                diff = _get_directional_distance(fs1_cut, fs2_cut) #Edu
            except AlphaCutError:
                # If both sets have a null alpha cut then ignore it
                # (count=False). If only one is empty then replace it with
                # the cut at its height.
                count = False
                try:
                    fs1_cut = mf_pair[0](alpha, z)
                    count = True
                except AlphaCutError:
                    alpha2 = mf_maxes[mf_index][0]
                    fs1_cut = mf_pair[0](alpha2, z)
                try:
                    fs2_cut = mf_pair[1](alpha, z)
                    count = True
                except AlphaCutError:
                    alpha2 = mf_maxes[mf_index][1]
                    fs2_cut = mf_pair[1](alpha2, z)
                if count:
                    diff = _get_directional_distance(fs1_cut, #Edu
                                                                 fs2_cut)
            # count is zero if the alpha cut is empty for both fuzzy sets
            if count:
                top += Decimal(str(alpha)) * diff
                bottom += alpha
    return gs.rnd(top / bottom)


def mcculloch_gt2(fs1, fs2): #Edu
    """Calculate the weighted Minkowski (r=1) directional distance."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        try:
            # Note: You can't just compare zslice_functions using distance_it2
            # as IAA and discrete sets aren't easily constructed that way.
            dist = _zslice_distance(fs1, fs2, z)
            top += z * dist
            bottom += z
        except ZLevelError:
            pass  # there is nothing at this zSlice
    return gs.rnd(top / bottom)


# Module 'distance_it2'
#------------------------------------------------------------------------------
"""This module contains distance measures for interval type-2 fuzzy sets."""

def figueroa_garcia_alpha(fs1, fs2):
    """Calculate the absolute difference between alpha-cuts."""
    def dist(alpha):
        fs1_cut_lower = fs1.calculate_alpha_cut_lower(alpha)
        fs1_cut_upper = fs1.calculate_alpha_cut_upper(alpha)
        fs2_cut_lower = fs2.calculate_alpha_cut_lower(alpha)
        fs2_cut_upper = fs2.calculate_alpha_cut_upper(alpha)
        return (Decimal(str(alpha)) *
                (abs(fs1_cut_upper[0] - fs2_cut_upper[0]) +
                 abs(fs1_cut_lower[0] - fs2_cut_lower[0]) +
                 abs(fs1_cut_upper[1] - fs2_cut_upper[1]) +
                 abs(fs1_cut_lower[1] - fs2_cut_lower[1])))
    a, b = integrate.quad(dist, 0, 1)
    return gs.rnd(a)


def figueroa_garcia_centres_hausdorff(fs1, fs2):
    """Calculate the hausdorff distance between the centre-of-sets."""
    fs1_centre = fs1.calculate_centre_of_sets()
    fs2_centre = fs2.calculate_centre_of_sets()
    return max(abs(fs1_centre[0] - fs2_centre[0]),
               abs(fs1_centre[1] - fs2_centre[1]))


def figueroa_garcia_centres_minkowski(fs1, fs2):
    """Calculate the absolute difference between the centre-of-sets."""
    fs1_centre = fs1.calculate_centre_of_sets()
    fs2_centre = fs2.calculate_centre_of_sets()
    return (abs(fs1_centre[0] - fs2_centre[0]) +
            abs(fs1_centre[1] - fs2_centre[1]))


def mcculloch_it2(fs1, fs2): #Edu
    """Calculate the weighted Minkowski (r=1) directional distance."""
    def order_lower_upper(fs):
        if ((fs.mf1.x_min <= fs.mf2.x_min and
                fs.mf1.x_max >= fs.mf2.x_max and
                fs.mf1.height >= fs.mf2.height)):
            return fs.mf1, fs.mf2
        else:
            return fs.mf2, fs.mf1
    fs1_lower_mf, fs1_upper_mf = order_lower_upper(fs1)
    fs2_lower_mf, fs2_upper_mf = order_lower_upper(fs2)
    return gs.rnd((mcculloch(FuzzySet(fs1_lower_mf), #edu
                                         FuzzySet(fs2_lower_mf)) +
                   mcculloch(FuzzySet(fs1_upper_mf), #Edu
                                         FuzzySet(fs2_upper_mf))) /
                Decimal(2))


# Module 'compatibility_t1'
#------------------------------------------------------------------------------
"""This module contains a compatibility measure for type-1 fuzzy sets.

See http://ieeexplore.ieee.org/abstract/document/6891672/ for further details.
"""


def compatibility(fs1, fs2, w0=0.7, w1=0.3):
    """Calculate weighted average of dissimilarity and directional distance.

    w0 and w1 are the weights given to dissimilarity and distance, resp.
    Result is in the range [-1, 1].
    0 is for identical sets and -1 and 1 are maximum distance.
    """
    w0 = Decimal(str(w0))
    w1 = Decimal(str(w1))
    similarity = jaccard(fs1, fs2) #Edu
    distance = mcculloch(fs1, fs2) #Edu
    max_distance = max(fs1.uod[1], fs2.uod[1]) - min(fs1.uod[0], fs1.uod[0])
    distance = distance / max_distance
    dissimilarity = 1 - similarity
    # make both have same sign
    if distance < 0:
        dissimilarity = -dissimilarity
    return gs.rnd((w0 * dissimilarity) + (w1 * distance))


# Module 'entropy_t1'
#------------------------------------------------------------------------------
"""This module contains entropy measures for type-1 sets.""" #Edu

def kosko(fs):
    """Calculate the degree to which the fuzzy set is fuzzy."""
    inc1 = 0
    inc2 = 0
    for x in gs.get_x_points():
        mu = fs.calculate_membership(x)
        inc1 += min(mu, 1 - mu)
        inc2 += max(mu, 1 - mu)
    return gs.rnd(inc1 / inc2)


# Module 'entropy_it2'
#------------------------------------------------------------------------------
"""This module contains entropy measures for interval type-2 fuzzy sets.""" #Edu

def szmidt_pacprzyk(fs):
    """Calculate the ratio between the upper & lower membership functions."""
    ent1 = 0
    ent2 = 0
    for x in gs.get_x_points():
        l, u = fs.calculate_membership(x)
        ent1 += 1 - max(1 - u, l)
        ent2 += 1 - min(1 - u, l)
    return gs.rnd((ent1 / ent2) / Decimal(gs.global_x_disc))


def zeng_li_entropy(fs): #Edu
    """Calculate entropy based on the sum of upper and lower memberships.""" #Edu
    result = 0
    for x in gs.get_x_points():
        l, u = fs.calculate_membership(x)
        result += abs(u + l - 1)
    return gs.rnd(1 - (result / Decimal(gs.global_x_disc)))


# Module 'inclusion_t1'
#------------------------------------------------------------------------------
"""This module contains inclusion (subsethood) measures for type-1 sets."""

def sanchez(fs1, fs2):
    """Calcualte the degree to which fs1 is contained within fs2."""
    inc1 = 0
    inc2 = 0
    for x in gs.get_x_points():
        mu1 = fs1.calculate_membership(x)
        inc1 += min(mu1, fs2.calculate_membership(x))
        inc2 += mu1
    return gs.rnd(inc1 / inc2)