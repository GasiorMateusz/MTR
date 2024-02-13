import argparse
import copy
import csv
import math

import matplotlib.pyplot as plt
from numpy import mean

from curve_generator import CurveGenerator
from ReinschAlgorithm import kSmooth
from lib import mathLib

from lib.array import Array, det
from lib.mathLib import linspace, euclidean_distance, euclidean_norm

T = "t"

BINORMALNY_ = "Binormalny "

NORMALNY_ = "Normalny "

STYCZNY_ = "Styczny "

TORSJA = "Torsja"

KRYWIZNA = "Krzywizna"

DL_LUKU = 'Długość łuku'
chart_name_sufix = ""

class Consts:
    average = "Wygładzanie średnie"
    randomized = "Punkty bazowe"
    ideal_path = "Ścieżka idealna"
    spline = "Splajn"
    base_vectors = ["Wektory styczne", "Wektory normalne", "Wektory Binormalne"]
    vectors_colors = ["red", "blue", "black"]
    color_ideal = "orange"
    color_spline = "blue"
    color_random = "green"
    color_average = "red"
    color_equidistant = "pink"


class Frenet:

    def __init__(self, curveSmoothing):
        self.cs = curveSmoothing

    def r_der(self, k, arc):
        smoothData = self.cs.spline_list
        pt = []
        Xout = self.cs.nodal_arcs
        t = arc
        t_k = Xout[k]
        if (t - t_k) == 0:
            t_k = 0
        for dim in range(len(self.cs.spline_list[0])):
            a_1 = smoothData[0][dim]["coeff"]["a"][k]
            b_1 = smoothData[0][dim]["coeff"]["b"][k]
            c_1 = smoothData[0][dim]["coeff"]["c"][k]

            pt.append(3 * a_1 * (t - t_k) ** 2 + 2 * b_1 * (t - t_k) + c_1)
        return pt

    def r_der_2(self, k, arc):
        smooth = self.cs.spline_list
        pt = []
        Xout = self.cs.nodal_arcs
        for dim in range(len(self.cs.spline_list[0])):
            a_1 = smooth[0][dim]["coeff"]["a"][k]
            b_1 = smooth[0][dim]["coeff"]["b"][k]
            t = arc
            t_k = Xout[k]
            pt.append(6 * a_1 * (t - t_k) + 2 * b_1)
        return pt

    def r_der_3(self, k, arc):
        smooth = self.cs.spline_list
        pt = []
        Xout = self.cs.Xout
        for dim in range(len(self.cs.spline_list[0])):
            a_1 = smooth[0][dim]["coeff"]["a"][k]
            t = arc
            t_k = Xout[k]
            pt.append(6 * a_1)
        return pt

    def r_der_2_norm(self, k, arc):
        vector = self.r_der_2(k, arc)
        magnitude = (sum(v ** 2 for v in vector)) ** 0.5
        return magnitude

    def tangent_unit_vector(self, k, arc):
        vector = self.r_der(k, arc)
        magnitude = (sum(v ** 2 for v in vector)) ** 0.5
        return [v / magnitude for v in vector]

    def normal_unit_vector(self, k, arc):
        vector = self.r_der_2(k, arc)
        magnitude = (sum(v ** 2 for v in vector)) ** 0.5
        if magnitude ==0:
            return [0,0,0]
        return [v / magnitude for v in vector]

    @staticmethod
    def binormal_unit_vector(tangent, normal):
        return (Array(tangent) @ Array(normal)).tolist()

    def curvature(self, k, arc):
        return self.r_der_2_norm(k, arc)

    def torsion(self, k, arc, binormal):
        dddot_r = Array(self.r_der_3(k, arc))
        ddot_r = Array(self.r_der_2(k, arc))
        dot_r = Array(self.r_der(k, arc))
        numerator_matrix = Array(
            [[Array.dotProd(dddot_r, dddot_r), Array.dotProd(dddot_r, ddot_r), Array.dotProd(dddot_r, dot_r)],
             [Array.dotProd(ddot_r, dddot_r), Array.dotProd(ddot_r, ddot_r), Array.dotProd(ddot_r, dot_r)],
             [Array.dotProd(dot_r, dddot_r), Array.dotProd(dot_r, ddot_r), Array.dotProd(dot_r, dot_r)]])
        if det(numerator_matrix) > 0:
            numerator = math.sqrt(det(numerator_matrix))
        else:
            numerator = 0
        denominator = det(Array([[Array.dotProd(ddot_r, ddot_r), Array.dotProd(dot_r, ddot_r)],
                                 [Array.dotProd(ddot_r, dot_r), Array.dotProd(dot_r, dot_r)]]))
        if denominator == 0:
            return 0
        return numerator / denominator

    def frenet_base(self, k, arc):

        tangent = self.tangent_unit_vector(k, arc)
        normal = self.normal_unit_vector(k, arc)
        curvature = self.curvature(k, arc)
        binormal = self.binormal_unit_vector(tangent, normal)
        torsion = self.torsion(k, arc, binormal)

        return [tangent, normal, binormal], torsion, curvature

    def anchor_pt(self, k, arc):  # r(t)
        smoothData = self.cs.spline_list
        pt = []
        Xout = self.cs.nodal_arcs
        for dim in range(len(self.cs.spline_list[0])):
            a_1 = smoothData[0][dim]["coeff"]["a"][k]
            b_1 = smoothData[0][dim]["coeff"]["b"][k]
            c_1 = smoothData[0][dim]["coeff"]["c"][k]
            d_1 = smoothData[0][dim]["coeff"]["d"][k]
            t = arc
            t_k = Xout[k]
            dt = t - t_k
            pt.append(a_1 * (t - t_k) ** 3 + b_1 * (t - t_k) ** 2 + c_1 * (t - t_k) + d_1)

        return pt

    @staticmethod
    def find_k_for_x_in_Xout(sm, x):
        for k, xk in enumerate(sm.nodal_XRef):
            if xk > x:
                if k == 0:
                    return 0
                return k - 1
        return len(sm.nodal_XRef) - 1

    def frenet_base_anchored(self, sm, base_id):
        arc = sm.equidistant_arcs[base_id]
        k = sm.equidistant_pts_k[base_id]
        base_vectors, torsion, curvature = self.frenet_base(k, arc)
        anchored_vectors = []
        anchor_pt = self.anchor_pt(k, arc)

        for vector in base_vectors:
            anchored_vector = [anchor_pt[i] + vector[i] for i in range(len(vector))]
            anchored_vectors.append([anchor_pt, anchored_vector])

        return anchored_vectors, [torsion, curvature]


class CurveSmoothing3D:
    trim = False
    frenet_bases_multiplicator = 1
    arc_length_precision = 10e-3
    iterations_nb_limit = 2
    initial_sigma = 0.1  # 0 = interpolation
    sigma_max_limit = 0.6
    sigma_min_limit = 0.14
    scale_factor = 10e-15
    original_list = None
    average_smoothing_neighbours_nb = 24
    segment_size = 10

    points_density_coeff = 1
    eq_pts_dens = 1
    eq_pts_number = None

    input_list = []
    input_set = []

    average_set = []
    serial_set = []
    spline_set = []

    average_list = []
    serial_list = []
    spline_list = []
    nodal_arcs = []
    denser_arcs = []
    sigma = []
    equidistant_pts = []

    @classmethod
    def from_argparse(cls, args):
        cls.trim = args.trim
        cls.frenet_bases_multiplicator = args.frenet_bases_multiplicator
        cls.arc_length_precision = args.arc_length_precision
        cls.iterations_nb_limit = args.iterations_nb_limit
        cls.initial_sigma = args.initial_sigma
        cls.sigma_max_limit = args.sigma_max_limit
        cls.sigma_min_limit = args.sigma_min_limit
        cls.scale_factor = args.scale_factor
        cls.original_list = args.original_list
        cls.average_smoothing_neighbours_nb = args.average_smoothing_neighbours_nb
        cls.segment_size = args.segment_size
        cls.points_density_coeff = args.points_density_coeff
        cls.eq_pts_number = args.eq_pts_number

    def execute(self):
        if self.eq_pts_number is None:
            self.eq_pts_number = len(self.input_set)

        if self.average_smoothing_neighbours_nb is not None:
            self.average_set, self.average_list = sm.smoothing_average(self.input_list,
                                                                       self.average_smoothing_neighbours_nb
                                                                       )
            set_to_sort, list_to_sort = self.average_set, self.average_list
        else:
            set_to_sort, list_to_sort = self.input_set, self.input_list
        if self.segment_size is not None:
            sorted_input_set = self.flatten_extend(sm.sort(set_to_sort, self.segment_size))
            sorted_input_list = self.set_to_list(sorted_input_set)
        else:
            sorted_input_set = set_to_sort
            sorted_input_list = list_to_sort
        self.average_set, self.sorted_input_list = sorted_input_set, sorted_input_list
        diagrams_preliminary(self)
        self.serial_set, self.serial_list = self.serial_no_parametrized_smoothing(
            sorted_input_set,
            self.initial_sigma,
            self.points_density_coeff
        )

        self.sigma = []
        for dim in range(len(sorted_input_list)):
            self.sigma.append(self.initial_sigma)
        self.nodal_arcs, arcs, self.spline_list, self.sigma, Xout = self.arc_length_parametrized_smoothing(
            nodal_arcs=self.serial_set,
            randomized_xy=sorted_input_set,
            sigma=self.sigma,
            iterations_nb_limit=self.iterations_nb_limit,
            points_density_coeff=self.points_density_coeff,
            arc_length_precision=self.arc_length_precision
        )

        trim_nb = 2
        if self.trim:
            self.trim_data(self.spline_list, trim_nb, self.points_density_coeff)
            self.nodal_arcs = self.nodal_arcs[trim_nb:-trim_nb]
            self.denser_arcs = arcs[trim_nb * self.points_density_coeff:-trim_nb * self.points_density_coeff]
            self.nodal_XRef = self.nodal_XRef[trim_nb:-trim_nb]
        else:
            self.denser_arcs = arcs
        self.Xout = Xout
        self.equidistant_pts, self.equidistant_arcs, self.equidistant_pts_k = self.equidistant_points(
            arcs_dens=self.denser_arcs,
            eq_pts_num=self.eq_pts_number,
            smooth=self.spline_list
        )
        self.equidistant_pts_list = Array(self.equidistant_pts).transpose()

    @staticmethod
    def trim_data(smooth, trim_size, dens_coeff):
        for i, der in enumerate(smooth):
            for j, dim in enumerate(der):
                smooth[i][j]["smoothData"] = smooth[i][j]["smoothData"][
                                             trim_size * dens_coeff:-trim_size * dens_coeff]
                smooth[i][j]["coeff"]['d'] = smooth[i][j]["coeff"]['d'][trim_size:-trim_size]
                smooth[i][j]["coeff"]['c'] = smooth[i][j]["coeff"]['c'][trim_size:-trim_size]
                smooth[i][j]["coeff"]['b'] = smooth[i][j]["coeff"]['b'][trim_size:-trim_size]
                smooth[i][j]["coeff"]['a'] = smooth[i][j]["coeff"]['a'][trim_size:-trim_size]

    def find_nodal_indx(self, nodal_arcs, arc):

        for j in range(len(nodal_arcs)):
            if nodal_arcs[j] <= arc:
                i = j
        if nodal_arcs[i] == nodal_arcs[-1]:
            i = i - 1
        return i

    @staticmethod
    def find_neighbour(arcs, s_desired):
        for i in range(0, len(arcs) - 1):
            if arcs[i] >= s_desired:
                ind_closest_point = i
                if abs(s_desired - arcs[i - 1]) < abs(arcs[i] - s_desired):
                    ind_closest_point = i - 1
                return ind_closest_point
        return len(arcs) - 1

    def equidistant_points(self, arcs_dens, eq_pts_num, smooth):
        eq_lengths = linspace(arcs_dens[0], arcs_dens[-1], eq_pts_num).tolist()

        pts = Array([smooth[0][0]["smoothData"], smooth[0][1]["smoothData"], smooth[0][2]["smoothData"]]).transpose()

        eq_points = []
        eq_points_k = []
        eq_len = []
        for idx, eq_length_arc in enumerate(eq_lengths):
            closest_point_idx = self.find_neighbour(arcs_dens, eq_length_arc)  # closest nodal arc to arc
            eq_points.append(
                self.arc_interpolation(s_arcs=arcs_dens,
                                       s_desired=eq_length_arc,
                                       k=closest_point_idx,
                                       r=smooth[0],
                                       r_der=smooth[1],
                                       r_der_2=smooth[2],
                                       r_der_3=smooth[3]
                                       )
            )
            eq_points_k.append(
                self.find_nodal_indx(self.nodal_arcs, eq_length_arc)
            )
            eq_len.append(eq_length_arc)
        return eq_points, eq_len, eq_points_k

    @staticmethod
    def arc_interpolation(s_arcs, s_desired, k, r, r_der, r_der_2, r_der_3=None):
        if k <= 2:
            return [r[0]["smoothData"][k], r[1]["smoothData"][k], r[2]["smoothData"][k]]
        elif k >= (len(s_arcs) - 2):
            return [r[0]["smoothData"][k], r[1]["smoothData"][k], r[2]["smoothData"][k]]
        else:
            h1 = s_arcs[k + 1] - s_arcs[k]
            h2 = s_arcs[k + 2] - s_arcs[k + 1]
            H1 = s_arcs[k] - s_arcs[k - 1]
            H2 = s_arcs[k - 1] - s_arcs[k - 2]

        a = s_arcs[k]
        sigma = s_desired - a

        a0 = - ((h1 - sigma) * (H1 + sigma) * (-2 * h2 + sigma) * (2 * H2 + sigma)) / (4 * h1 * H1 * h2 * H2)

        b1 = - ((2 * h2 - sigma) * sigma * (H1 + sigma) * (2 * H2 + sigma)) / (
                h1 * (h1 + H1) * (h1 - 2 * h2) * (h1 + 2 * H2))

        c1 = ((h1 - sigma) * (2 * h2 - sigma) * sigma * (2 * H2 + sigma)) / (
                H1 * (h1 + H1) * (H1 + 2 * h2) * (H1 - 2 * H2))

        b2 = - ((h1 - sigma) * sigma * (H1 + sigma) * (2 * H2 + sigma)) / (
                4 * h2 * (-h1 + 2 * h2) * (H1 + 2 * h2) * (h2 + H2))

        c2 = ((h1 - sigma) * sigma * (H1 + sigma) * (-2 * h2 + sigma)) / (
                4 * (H1 - 2 * H2) * H2 * (h2 + H2) * (h1 + 2 * H2))

        def interpolating_func(r_der_3=None):
            interpolated_point = []
            for dim in range(len(r)):
                if r_der_3 is not None:
                    o3 = 1 / 6 * (b1 * h1 ** 3 - c1 * H1 ** 3 + 8 * b2 * h2 ** 3 - 8 * c2 * H2 ** 3) * \
                         r_der_3[dim]["smoothData"][k]
                else:
                    o3 = 0
                interpolated_point.append((a0 + b1 + b2 + c1 + c2) * r[dim]["smoothData"][k]
                                          + (b1 * h1 - c1 * H1 + 2 * b2 * h2 - 2 * c2 * H2) * r_der[dim]["smoothData"][
                                              k]
                                          + 0.5 * (b1 * h1 ** 2 + c1 * H1 ** 2 + 4 * b2 * h2 ** 2 + 4 * c2 * H2 ** 2) *
                                          r_der_2[dim]["smoothData"][k] + o3
                                          )
            return interpolated_point

        return interpolating_func(r_der_3)

    @staticmethod
    def closest_pts_to_equidistant(arcs, points_number, smooth):
        eq_lenghts = linspace(arcs[0], arcs[-1], points_number)
        current_length_idx = 1
        eq_dst_pts = []
        for dim in range(len(smooth)):
            eq_dst_pts.append([smooth[dim]["smoothData"][0]])
        eq_dst_pts_arc = [0]
        for i in range(1, len(arcs)):
            if arcs[i] > eq_lenghts[current_length_idx]:
                ind_closest_point = i
                if eq_lenghts[current_length_idx] - arcs[i - 1] < arcs[i] - eq_lenghts[current_length_idx]:
                    ind_closest_point = i - 1
                eq_dst_pts_arc.append(arcs[ind_closest_point])
                current_length_idx += 1
                for dim in range(len(smooth)):
                    eq_dst_pts[dim].append(smooth[dim]["smoothData"][ind_closest_point])
        for dim in range(len(smooth)):
            eq_dst_pts[dim].append(smooth[dim]["smoothData"][-1])
        eq_dst_pts_arc.append(arcs[-1])

        return eq_dst_pts

    def smoothing_average(self, input_data, neigbours_nb=24):
        input_data = Array(input_data).transpose()
        smoothed_points = [self.geometric_center(tmp_point, input_data, neigbours_nb) for tmp_point in input_data]
        return copy.deepcopy(smoothed_points), Array(copy.deepcopy(smoothed_points)).transpose()

    def geometric_center(self, tmp_point, input_points, r=1, neighbours_nb=24):
        indices = self.find_nearest_neighbors(input_points, tmp_point, neighbours_nb, r)
        weights = self.calculate_weights(input_points, indices, tmp_point, r)
        weight_standard = sum(weights)
        geometric_center = self.calculate_geometric_center(weights, input_points, indices, weight_standard)

        return geometric_center

    @staticmethod
    def length(smooth, i):
        tmp = []
        for dimension in range(len(smooth)):
            tmp.append(smooth[dimension]["smoothData"][i] ** 2)
        return (sum(tmp)) ** 0.5

    @staticmethod
    def list_of_arcs(para, smooth):
        s = [0]
        dt = para[1]
        for i in range(1, len(para)):
            s.append(s[- 1] + CurveSmoothing3D.length(smooth, i) * dt)
        return s

    def serial_no_parametrized_smoothing(self, input_data, init_sigma, points_density_coeff):
        serial_no = [_ for _ in range(len(input_data))]
        sigma = []
        for i in range(len(input_data[0])):
            sigma.append(init_sigma)
        arc_para, arcs, sigma, dens_smooth, Xout = self.smoothing_iteration(serial_no, input_data,
                                                                            points_density_coeff,
                                                                            sigma
                                                                            )
        return arc_para, dens_smooth[0]

    @staticmethod
    def arcs_in_nodal_pts(arcs, dens_coeff):
        double_density_number = len(arcs)
        single_density_number = int((double_density_number + dens_coeff - 1) / dens_coeff)
        s = [0]
        for i in range(1, single_density_number):
            s.append(arcs[i * dens_coeff])
        return s

    @staticmethod
    def denserXref(nodalRef, dens_coeff):
        xRef = []
        step = nodalRef[1] / dens_coeff
        for x in nodalRef:
            if x == nodalRef[-1]:
                xRef.append(x)
            else:
                for i in range(dens_coeff):
                    xRef.append(x + i * step)
        return xRef

    def smoothing_iteration(self, parametrization, input_data, dens_coeff, sigma, xRef=None):
        is_Xref_none = xRef is None
        coor_with_par, smooth = [[], [], []], [[], [], []]
        for coor in range(len(coor_with_par)):
            coor_with_par[coor] = [[parametrization[i], input_data[i][coor]] for i in range(len(input_data))]
        if xRef is None:
            self.nodal_XRef = linspace(parametrization[0], parametrization[-1], len(input_data))
            xRef = sm.denserXref(sm.nodal_XRef, dens_coeff)
            prec = 10e12
            xRef = [math.floor(x * prec) / prec for x in xRef]
            self.nodal_XRef = [math.floor(x * prec) / prec for x in self.nodal_XRef]
            if (xRef[0::dens_coeff] != self.nodal_XRef): raise Exception("Xref doesn't include nodal points")
        for coor in range(len(coor_with_par)):
            smooth[coor] = kSmooth(coor_with_par[coor], xRef, sigma[coor])
            sigma[coor] = self.weights_based_on_curvature_and_torsion(smooth[coor]["smoothData"][2])
        dens_smooth = []
        for derivative in range(4):
            dens_smooth.append(
                [{"smoothData": copy.deepcopy(i["smoothData"][derivative]), "coeff": copy.deepcopy(i["coeff"])} for i in
                 smooth]
            )
        if is_Xref_none:
            arcs = self.list_of_arcs(xRef, dens_smooth[1])
            nodal_arcs = self.arcs_in_nodal_pts(arcs, dens_coeff)
            print("Length :", nodal_arcs[-1])
        else:
            return dens_smooth
        return nodal_arcs, arcs, sigma, dens_smooth, xRef

    def arc_length_parametrized_smoothing(self, nodal_arcs, randomized_xy, iterations_nb_limit,
                                          sigma, points_density_coeff, arc_length_precision):
        iteration = 1
        prev_iteration_length = nodal_arcs[-1] + 1
        while abs(nodal_arcs[-1] - prev_iteration_length) > arc_length_precision and iteration < iterations_nb_limit:
            prev_iteration_length = nodal_arcs[-1]
            nodal_arcs, arcs, sigma, dens_smooth, Xout = self.smoothing_iteration(nodal_arcs, randomized_xy,
                                                                                  points_density_coeff,
                                                                                  sigma
                                                                                  )
            iteration += 1

            print("iteration : ", iteration)
            print("Length :", nodal_arcs[-1])
            print("precision : ", round(abs(nodal_arcs[-1] - prev_iteration_length), 5))
            print("precision limit : ", arc_length_precision)
        return nodal_arcs, arcs, dens_smooth, sigma, Xout

    @staticmethod
    def normalize(values: list[float], min_range=0.0, max_range=1.0) -> list[float]:
        max_value = max(values)
        min_value = min(values)
        if max_value == min_value:
            for i in range(len(values)):
                values[i] = min_range
        else:
            for i in range(len(values)):
                values[i] = (values[i] - min_value) * (max_range - min_range) / (max_value - min_value) + min_range
        return values

    def weights_based_on_curvature_and_torsion(self, second_derivative: list[float]):
        """
        :param second_derivative: list of second points
            scale_factor Współczynnik skalujący
        :return: weights of points based on curvature and torsion
        """
        # Modyfikacja wag na podstawie krzywizny
        epsilon = 1e-6  # Mała wartość zapobiegająca dzieleniu przez zero
        weights = []
        for value in second_derivative:
            weights.append(self.scale_factor / (abs(value) + epsilon))
        weights = self.normalize(weights, self.sigma_min_limit,
                                 self.sigma_max_limit
                                 )  # reasonable min value is 0.1 and max i 1.1
        return weights

    @staticmethod
    def closest_neighbour(point, points):
        """
        Znajduje najbliższego sąsiada dla danego punktu wśród innych punktów.

        :param point: Tuple reprezentujący współrzędne punktu (x, y, z).
        :param points: Lista punktów do analizy.
        :return: Najbliższy sąsiad w postaci tuple (x, y, z).
        """
        closest = None
        closest_idx = 0
        min_distance = float('inf')  # Początkowo ustaw odległość na nieskończoność

        for neighbour_idx, neighbour in enumerate(points):
            distance = euclidean_norm(Array(point) - Array(neighbour))
            if distance < min_distance:
                min_distance = distance
                closest = neighbour
                closest_idx = neighbour_idx

        return closest, closest_idx

    def sort(self, points, segment_size):
        start, end = self.find_segment_ends(points)
        return self.sort_by_segments(points, segment_size, points.index(start))

    def sort_by_segments(self, points, segment_size, start_ind=0):
        points_left_to_sort = copy.deepcopy(points)
        segments = []
        target = points_left_to_sort[start_ind]
        while len(points_left_to_sort) > 0:
            segment, points_left_to_sort = self.find_n_nearest_neighbors(target, points_left_to_sort, segment_size)
            sorted_segment_set, _ = self.sort_by_closest_neighbour(segment)
            seg_ends = self.find_segment_ends(sorted_segment_set)
            sorted_segment_set, _ = self.sort_by_closest_neighbour(segment, segment.index(seg_ends[0]))
            target, _ = self.closest_neighbour(seg_ends[1],
                                               points_left_to_sort
                                               )
            segments.append(sorted_segment_set)
        return segments

    def find_segment_ends(self, points):

        max_distance = 0
        max_distance_points = None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = euclidean_norm(Array(points[i]) - Array(points[j]))

                if distance > max_distance:
                    max_distance = distance
                    max_distance_points = (points[i], points[j])

        return max_distance_points

    @staticmethod
    def sort_by_closest_neighbour(points, start_point_idx=None):
        points_left_to_sort = copy.deepcopy(points)
        if start_point_idx is None:
            start_point_idx = 0
        sorted_points = [points_left_to_sort[start_point_idx]]
        sorted_indexes = [start_point_idx]
        del points_left_to_sort[start_point_idx]

        while len(points_left_to_sort) > 0:
            neighbour_searched_for = sorted_points[-1]
            neighbour, closest_idx = CurveSmoothing3D.closest_neighbour(neighbour_searched_for,
                                                                        points_left_to_sort
                                                                        )
            sorted_points.append(neighbour)
            sorted_indexes.append(closest_idx)
            points_left_to_sort.pop(closest_idx)

        return sorted_points, 0

    def find_n_nearest_neighbors(self, target_point, points, n):
        if len(points) < n:
            n = len(points)
        distances = [(euclidean_norm(Array(target_point) - Array(point)), point) for point in points]
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [point[1] for point in distances[:n]]
        points[:] = [point for point in points if point not in nearest_neighbors]

        return nearest_neighbors, points

    def print_frenet_base(self, frenet_bases_arcs, ax, sm):

        print("total length " + str(sm.denser_arcs[-1]))
        torsion_curvature = []
        frenet = Frenet(self)
        bases = []
        for base_id in range(len(frenet_bases_arcs)):

            vectors, t_c = frenet.frenet_base_anchored(sm, base_id)
            bases.append(vectors)
            torsion_curvature.append(t_c)
            if ax != None:
                if base_id % sm.frenet_bases_multiplicator == 0:
                    for vector_id, vector in enumerate(vectors):
                        ax.plot([pt[0] for pt in vector], [pt[1] for pt in vector], [pt[2] for pt in vector],
                                label=(Consts.base_vectors[vector_id]),
                                color=Consts.vectors_colors[vector_id]
                                )
        return torsion_curvature, bases

    @staticmethod
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    @staticmethod
    def set_to_list(set):
        lists = [[] for _ in range(len(set[0]))]
        for i, point in enumerate(set):
            for j, coor in enumerate(point):
                lists[j].append(coor)
        return lists

    def find_nearest_neighbors(self, input_points, tmp_point, neighbours_nb, distance_upper_bound):
        nearest_neighbors = []
        distances = []

        for i, point in enumerate(input_points):
            distance = euclidean_distance(tmp_point, point)
            if distance <= distance_upper_bound:
                nearest_neighbors.append(point)
                distances.append(distance)
        sorted_neighbors = [neighbor for _, neighbor in sorted(zip(distances, nearest_neighbors))][:neighbours_nb]
        indices = [input_points.tolist().index(i) for i in sorted_neighbors]
        return indices

    @staticmethod
    def calculate_weights(input_points, indices, tmp_point, r):
        weights = []

        for index in indices:
            point = input_points[index]
            distance_squared = sum((a - b) ** 2 for a, b in zip(point, tmp_point))
            weight = mathLib.e ** (-distance_squared / r ** 2)
            weights.append(weight)

        return weights

    @staticmethod
    def calculate_geometric_center(weights, input_points, indices, weight_standard):
        weighted_points = []

        for weight, index in zip(weights, indices):
            point = input_points[index]
            weighted_point = [weight * coord for coord in point]
            weighted_points.append(weighted_point)

        weighted_sum = [sum(coords) for coords in zip(*weighted_points)]
        geometric_center = [coord / weight_standard for coord in weighted_sum]

        return geometric_center


def diagrams_preliminary(sm):
    fig, axs = plt.subplots(1, 2, figsize=(chart_size))

    id = 0
    axs[id] = fig.add_subplot(1, 2, id + 1, projection='3d')
    axs[id].set_title(Consts.randomized + " vs " + Consts.ideal_path)
    axs[id].scatter(
        sm.input_list[0], sm.input_list[1], sm.input_list[2], label=Consts.randomized, color=Consts.color_random
    )
    axs[id].plot(
        sm.original_list[0], sm.original_list[1], sm.original_list[2], label=Consts.ideal_path, color=Consts.color_ideal
    )
    axs[id].legend()

    id = 1
    axs[id] = fig.add_subplot(1, 2, id + 1, projection='3d')
    axs[id].set_title(Consts.average + " vs " + Consts.ideal_path)
    axs[id].plot(
        sm.original_list[0], sm.original_list[1], sm.original_list[2], label=Consts.ideal_path, color=Consts.color_ideal
    )
    axs[id].scatter(
        sm.sorted_input_list[0], sm.sorted_input_list[1], sm.sorted_input_list[2], label=Consts.average,
        color=Consts.color_average
    )
    axs[id].legend()
    plt.tight_layout()
    for ax in axs:
        set_axes_equal(ax)
    plt.show()


def diagrams_smoothing(sm):
    spline = sm.spline_list
    accurate_smooth = sm.original_list
    fig, axs = plt.subplots(1, 3, figsize=(chart_size))

    id = 0
    axs[id] = fig.add_subplot(1, 3, id + 1, projection='3d')
    axs[id].set_title(Consts.randomized + ' vs ' + Consts.spline)
    axs[id].plot(spline[0][0]["smoothData"], spline[0][1]["smoothData"],
                 spline[0][2]["smoothData"],
                 label=Consts.spline,
                 color=Consts.color_spline
                 )
    axs[id].scatter(
        sm.sorted_input_list[0], sm.sorted_input_list[1], sm.sorted_input_list[2], label=Consts.randomized,
        color=Consts.color_average
    )
    axs[id].legend()

    id = 1
    axs[id] = fig.add_subplot(1, 3, id + 1, projection='3d')
    axs[id].set_title(Consts.spline + ' vs ' + Consts.ideal_path)
    axs[id].scatter(
        sm.spline_list[0][0]["smoothData"], sm.spline_list[0][1]["smoothData"], sm.spline_list[0][2]["smoothData"],
        label=Consts.spline,
        color=Consts.color_spline
    )
    axs[id].plot(accurate_smooth[0], accurate_smooth[1],
                 accurate_smooth[2],
                 label=Consts.ideal_path,
                 color=Consts.color_ideal
                 )
    axs[id].legend()

    id = 2
    axs[id] = fig.add_subplot(1, 3, id + 1, projection='3d')
    axs[id].set_title("Splajn i bazy Freneta")
    torsion_and_curvature, bases = sm.print_frenet_base(sm.equidistant_arcs, axs[id], sm)
    axs[id].plot(spline[0][0]["smoothData"], spline[0][1]["smoothData"],
                 spline[0][2]["smoothData"],
                 label=Consts.spline,
                 color=Consts.color_spline
                 )
    for ax in axs:
        set_axes_equal(ax)
    plt.show()
    return torsion_and_curvature


def diagrams_torsion_curvature(sm, torsion_curvature):
    spline = sm.spline_list
    accurate_smooth = sm.original_list
    fig, axs = plt.subplots(1, 3, figsize=(chart_size))
    torsion_curvature = Array(torsion_curvature).transpose()

    id = 0
    axs[id].set_title("Torsja i krzywizna vs długość łuku")
    axs[id].plot(
        sm.equidistant_arcs, torsion_curvature[0], label=TORSJA
    )
    axs[id].plot(
        sm.equidistant_arcs, torsion_curvature[1], label=KRYWIZNA
    )
    axs[id].set_xlabel(DL_LUKU)
    axs[id].legend()

    id = 1
    axs[id].set_title("Torsja vs długość łuku")
    axs[id].plot(
        sm.equidistant_arcs, torsion_curvature[0], label=TORSJA
    )
    axs[id].set_xlabel(DL_LUKU)
    axs[id].legend()

    id = 2
    axs[id].set_title('Krzywizna vs długość łuku')
    axs[id].plot(
        sm.equidistant_arcs, torsion_curvature[1], label=KRYWIZNA
    )
    axs[id].set_xlabel(DL_LUKU)
    axs[id].legend()
    plt.tight_layout()
    plt.show()


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def fitting_test_charts(sm, t):
    spline = sm.spline_list
    t_dens = linspace(t[0], t[-1], len(sm.denser_arcs))

    fig, axs = plt.subplots(1, 3, figsize=(chart_size))

    id = 0
    axs[id] = fig.add_subplot(1, 3, id + 1)
    axs[id].set_title("[t(s), x(s)]")
    axs[id].scatter(spline[0][0]["smoothData"], t_dens,
                    label=Consts.spline,
                    color=Consts.color_ideal
                    )
    axs[id].scatter(
        sm.input_list[0], t, label="t",
        color=Consts.color_average
    )
    axs[id].legend()

    id = 1
    axs[id] = fig.add_subplot(1, 3, id + 1)
    axs[id].set_title("[t(s), y(s)]")

    axs[id].scatter(spline[0][1]["smoothData"], t_dens,
                    label=Consts.spline,
                    color=Consts.color_ideal
                    )
    axs[id].scatter(
        sm.input_list[1], t, label="t",
        color=Consts.color_average
    )
    axs[id].legend()

    id = 2
    axs[id] = fig.add_subplot(1, 3, id + 1)
    axs[id].set_title("[t(s), z(s)]")
    axs[id].scatter(spline[0][2]["smoothData"], t_dens,
                    label=Consts.spline,
                    color=Consts.color_ideal
                    )
    axs[id].scatter(
        sm.input_list[2], t, label="t",
        color=Consts.color_average
    )
    plt.show()


def bases_test_charts(sm):
    torsion_and_curvature, bases = sm.print_frenet_base(sm.equidistant_arcs, None, sm)
    tangents = []
    normals = []
    binormals = []
    dim = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 3, figsize=(chart_size))
    for base in bases:
        tangents.append(base[0][1])
        normals.append(base[1][1])
        binormals.append(base[2][1])

    for i in range(3):
        id = i

        axs[0, id].scatter([row[0] for row in tangents], t,
                           label=STYCZNY_ + dim[i],
                           color=Consts.color_ideal
                           )
        axs[0, id].scatter(
            sm.input_list[0], t, label=T,
            color=Consts.color_average
        )
        axs[0, id].legend()

    for i in range(3):
        id = i

        axs[1, id].scatter([row[1] for row in tangents], t,
                           label=NORMALNY_ + dim[i],
                           color=Consts.color_ideal
                           )
        axs[1, id].scatter(
            sm.input_list[1], t, label=T,
            color=Consts.color_average
        )
        axs[1, id].legend()

    for i in range(3):
        id = i
        axs[2, id].scatter([row[2] for row in tangents], t,
                           label=BINORMALNY_ + dim[i],
                           color=Consts.color_ideal
                           )
        axs[2, id].scatter(
            sm.input_list[2], t, label=T,
            color=Consts.color_average
        )
        axs[2, id].legend()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Algorytm smoothing spline.')
    parser.add_argument('--trim', '-t', action='store_true', help='Usunięcie pierwszy i ostatnich punktów węzłowych')
    parser.add_argument('--frenet_bases_multiplicator', '-fbm', type=float, default=1,
                        help='Współczynnik widocznych baz')
    parser.add_argument('--arc_length_precision', '-alp', type=float, default=10e-3, help='Dokładność obliczeń')
    parser.add_argument('--iterations_nb_limit', '-inl', type=int, default=2, help='Maksymalna liczba iteracji')
    parser.add_argument('--initial_sigma', '-is', type=float, default=0.1, help='Początkowa wartość sigma')
    parser.add_argument('--sigma_max_limit', '-sml', type=float, default=0.6, help='Maksymalny zakres sigma')
    parser.add_argument('--sigma_min_limit', '-smil', type=float, default=0.14, help='Minimalny zakres sigma')
    parser.add_argument('--scale_factor', '-sf', type=float, default=10e-15, help='Współczynnik skalowania sigma')
    parser.add_argument('--original_list', '-ol', type=str, default=None, help='Plik zawierający ścieżkę idealna dla porównania')
    parser.add_argument('--average_smoothing_neighbours_nb', '-asn', type=int, default=None,
                        help='Liczba sąsiadów dla wygładzania średniego')
    parser.add_argument('--segment_size', '-ss', type=int, default=None, help='Liczebność segmentu sortowania')
    parser.add_argument('--points_density_coeff', '-pdc', type=int, default=1,
                        help='Współczynnik gęstości punktów smooth')
    # parser.add_argument('--eq_pts_dens', '-eqd', type=int, default=1, help='Description of eq_pts_dens argument')
    parser.add_argument('--eq_pts_number', '-eqn', type=int, default=3, help='Liczba baz')
    parser.add_argument('--noise', '-n', action='store_true', help='Szum demonstracyjnej krzywej')
    parser.add_argument('--demo', '-d', action='store_true', help='Demonstracyjna krzywa')
    parser.add_argument('--input_set', '-ins', type=str, default=None,
                        help='Plik zawierający listę punktów wejściowych')
    parser.add_argument('--parametrization', '-par', type=str, default=None,
                        help='Plik zawierający parametryzację')

    args = parser.parse_args()
    return args

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(list(map(float, row)))
    return data


if __name__ == "__main__":
    args = parse_arguments()
    sm = CurveSmoothing3D()
    if (args.demo):
        generator = CurveGenerator().Spirala_2
        (sm.input_set, sm.input_list, sm.original_list, sm.average_smoothing_neighbours_nb, sm.segment_size,
         sm.initial_sigma, sm.sigma_max_limit, sm.sigma_min_limit, sm.eq_pts_number, t) = generator(args.noise)
        chart_name = generator.__name__
        sm.frenet_bases_multiplicator = 1
        sm.points_density_coeff = 5
    else:
        sm.from_argparse(args)
        chart_name = "Krzywa"
        t = Array(read_csv(args.parametrization))
        sm.input_set = Array(read_csv(args.input_set))
        sm.original_list = Array(read_csv(args.original_list)).transpose()
        sm.input_list = Array.transpose(sm.input_set)
        flatten_t = []
        for row in t:
            flatten_t.append(row[0])
        t = flatten_t

        print(t)

    chart_size = 20, 10
    sm.execute()
    t_c = diagrams_smoothing(sm)
    fitting_test_charts(sm, t)
    bases_test_charts(sm)
    diagrams_torsion_curvature(sm, t_c)
