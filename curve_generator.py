import copy
import random

import numpy as np
from lib.array import Array
from lib.import_gpx import import_gpx


class CurveGenerator:

    @staticmethod
    def Klepsydra_0(nCycles=1, npointsPerCycle=12 * 72, eps=0.2):
        const_12 = 12.0
        const_6 = 6
        const_3 = 3

        ex = [np.cos(2 * np.pi * np.random.random()), np.sin(2 * np.pi * np.random.random())]
        ey = lambda v: [v[0] / np.sqrt(v[0] ** 2 + v[1] ** 2), v[1] / np.sqrt(v[0] ** 2 + v[1] ** 2)]
        ey = ey([1, 1] - np.dot([1, 1], ex) * np.array(ex))

        rc = const_6 * (2 * np.random.random(2) - np.array([1, 2]))
        ampl = np.exp(const_3 * np.random.random())

        rpoints0, rpoints1 = [], []

        for _ in range(nCycles * npointsPerCycle):
            phi = np.random.uniform(0, 2 * np.pi * nCycles)
            x_noise = rc[0] + ampl * (1 + eps * np.random.uniform(-1, 1)) * (
                    ex[0] * np.cos(phi) + ey[0] * np.sin(2 * phi))
            y_noise = rc[1] + ampl * (1 + eps * np.random.uniform(-1, 1)) * (
                    ex[1] * np.cos(phi) + ey[1] * np.sin(2 * phi))
            x = rc[0] + ampl * (ex[0] * np.cos(phi) + ey[0] * np.sin(2 * phi))
            y = rc[1] + ampl * (ex[1] * np.cos(phi) + ey[1] * np.sin(2 * phi))
            rpoints0.append([phi, x_noise, y_noise])
            rpoints1.append([phi, x, y])

        rpoints0 = sorted(rpoints0, key=lambda x: x[0])
        rpoints1 = sorted(rpoints1, key=lambda x: x[0])
        rpoints0 = [[(const_12 / len(rpoints0)) * idx, x, y] for idx, [_, x, y] in enumerate(rpoints0)]
        rpoints1 = [[(const_12 / len(rpoints1)) * idx, x, y] for idx, [_, x, y] in enumerate(rpoints1)]

        rpoints0 = np.array(rpoints0)
        rpoints1 = np.array(rpoints1)

        X_n, Y_n = np.transpose([[row[1], row[2]] for row in rpoints0])
        X, Y = np.transpose([[row[1], row[2]] for row in rpoints1])
        Z = np.zeros(len(X_n))

        list = [X_n.tolist(), Y_n.tolist(), Z.tolist()]
        set = np.transpose(list)
        ideal_list = [X.tolist(), Y.tolist(), Z.tolist()]
        average_sm_neighbour_nb = 24
        segment_size = None
        if eps != 0:
            initial_sigma = 0.1
            sigma_min = 0.09
        else:
            initial_sigma = 0.01
            sigma_min = 0.01
        sigma_max = 1
        eq_pts_num = 100
        return set, list, ideal_list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def Torus_0(phi=100, step=0.1, eps=0.0):
        phi_values = np.arange(0, phi, step)
        np.random.shuffle(phi_values)
        random_values = (np.random.random((len(phi_values), 3)) * 0.5 - 0.25) * eps
        const_3 = 3
        const_7 = 7
        xyz_pairs = np.array([
            [np.cos(phi) * (const_3 + np.cos(const_7 * phi)), np.sin(phi) * (const_3 + np.cos(const_7 * phi)),
             np.sin(const_7 * phi)] + random
            for phi, random in zip(phi_values, random_values)
        ]
        )
        phi_values = np.arange(0, phi, step)
        const_3 = 3
        const_7 = 7
        xyz_pairs_ideal = Array([
            [np.cos(phi) * (const_3 + np.cos(const_7 * phi)), np.sin(phi) * (const_3 + np.cos(const_7 * phi)),
             np.sin(const_7 * phi)]
            for phi in phi_values
        ]
        )

        xyz_pairs = np.transpose(xyz_pairs)
        randomized_xyz_list = copy.deepcopy(xyz_pairs)
        randomized_points_set = [[xyz_pairs[0][i], xyz_pairs[1][i], xyz_pairs[2][i]] for i in
                                 range(len(xyz_pairs[0]))]

        list = randomized_xyz_list
        set = randomized_points_set
        toroidal_list = xyz_pairs_ideal.transpose()
        average_sm_neighbour_nb = 12
        segment_size = 30
        initial_sigma = 0.05
        sigma_max = 0.6
        sigma_min = 0.05
        eq_pts_num = 400
        return set, list.tolist(), toroidal_list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    # test
    @staticmethod
    def Helisa_0(e=0.5, n=7, points_num=500, noise=False):
        t = np.linspace(0, n, points_num)
        if noise:
            k = 7
            eps = (np.random.random((len(t))) * 1 / k - 1 / k / 2) * 1
        else:
            eps = 0
        x = (1 + e * np.sin(2 * np.pi * t)) * np.cos(2 * np.pi * t / n) * (1 + eps)
        y = (1 + e * np.sin(2 * np.pi * t)) * np.sin(2 * np.pi * t / n) * (1 + eps)
        z = e * np.cos(2 * np.pi * t) * (1 + eps)

        list = [x.tolist(), y.tolist(), z.tolist()]
        set = Array(list).transpose()
        average_sm_neighbour_nb = None
        segment_size = None
        if noise:
            eps = 0
            x = (1 + e * np.sin(2 * np.pi * t)) * np.cos(2 * np.pi * t / n) * (1 + eps)
            y = (1 + e * np.sin(2 * np.pi * t)) * np.sin(2 * np.pi * t / n) * (1 + eps)
            z = e * np.cos(2 * np.pi * t) * (1 + eps)

            ideal = [x.tolist(), y.tolist(), z.tolist()]
        else:
            ideal = list

        initial_sigma = 0
        sigma_max = 0
        sigma_min = 0
        eq_pts_num = points_num
        return set, list, ideal, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num, t

    @staticmethod
    def Pochylony_okrag(num_points=300, radius=1, tilt_angle=45, noise=0.1):
        theta = np.linspace(0, 2 * np.pi, num_points)
        rotation_matrix = np.array([[np.cos(np.radians(tilt_angle)), 0, np.sin(np.radians(tilt_angle))],
                                    [0, 1, 0],
                                    [-np.sin(np.radians(tilt_angle)), 0, np.cos(np.radians(tilt_angle))]]
                                   )
        points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
        tilted_points = np.dot(rotation_matrix, points)
        tilted_points_without_noise = tilted_points
        if noise:
            random_values = np.random.normal(0, noise, tilted_points.shape)
            tilted_points = tilted_points + random_values
            initial_sigma = 0.1
            sigma_min = 0.05
        else:
            initial_sigma = 0.02
            sigma_min = 0.01
        sigma_max = 1.1
        set = np.transpose(tilted_points).tolist()
        list = tilted_points.tolist()
        circle_list = tilted_points_without_noise.tolist()
        average_sm_neighbour_nb = None
        segment_size = None
        eq_pts_num = 100
        return set, list, circle_list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def Spirala_0(radius=5, shape_factor=1, coil_distance_factor=0.5, num_coils=1, noise=False, noise_intensity=0.1):
        t = np.linspace(0, 2 * np.pi * num_coils, 500)
        x = radius * np.cos(t) * np.cos(shape_factor * t)
        y = radius * np.sin(t) * np.cos(shape_factor * t)
        z = coil_distance_factor * t

        if noise:

            noise = np.random.random((3, t.size)) * 2 - 1
            noise *= noise_intensity
            noise = noise.tolist()

            x += noise[0]
            y += noise[1]
            z += noise[2]
            segment_size = None
            initial_sigma = 0.1
            sigma_max = 1.1
            sigma_min = 0.05
            eq_pts_num = 100
            average_sm_neighbour_nb = 10
        else:
            segment_size = None
            average_sm_neighbour_nb = None
            initial_sigma = 0.02
            sigma_max = 1.1
            sigma_min = 0.01
            eq_pts_num = 100

        spiral_list = [
            radius * np.cos(t) * np.cos(shape_factor * t),
            radius * np.sin(t) * np.cos(shape_factor * t),
            coil_distance_factor * t
        ]
        spiral_list = [spiral_list[0].tolist(), spiral_list[1].tolist(), spiral_list[2].tolist()]

        list = [x.tolist(), y.tolist(), z.tolist()]
        set = Array(list).tolist()

        return set, list, spiral_list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def Spirala_stozkowa(rnd_pts_num=600, R=1.5, helix_pts_num=1000):
        phi_max = 4
        phi_min = -4
        points = []
        for _ in range(rnd_pts_num):
            phi = np.pi * np.random.uniform(phi_min, phi_max)
            x = np.random.rand()
            y = np.random.rand()
            z = np.random.rand()

            point = np.array([
                phi * np.cos(phi) + R * z ** (1 / 3) * (2 * (1 - x) * x * np.cos(2 * np.pi * y)),
                phi * np.sin(phi) + R * np.sin(2 * np.pi * y) * 2 * (1 - x) * x,
                2 * phi + R * (1 - 2 * x)
            ]
            )
            points.append(point)

        helix_phi = np.linspace(phi_min * np.pi, phi_max * np.pi, helix_pts_num)
        helix_set = np.array([
            helix_phi * np.cos(helix_phi),
            helix_phi * np.sin(helix_phi),
            2 * helix_phi
        ]
        )
        list = np.array(points).T.tolist()
        set = [p.tolist() for p in points]
        average_sm_neighbour_nb = 200
        segment_size = 10
        initial_sigma = 0.1
        sigma_min = 0.3
        sigma_max = 0.6
        eq_pts_num = 200
        return set, list, helix_set, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def biale_morza():
        set, list = import_gpx(
            "lib/biale_morza.gpx")
        average_sm_neighbour_nb = None
        segment_size = None
        eq_pts_num = 100
        initial_sigma = 0.1
        sigma_min = 0.1
        sigma_max = sigma_min * 10
        return set, list, list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def Dane_GPS():
        set, list = import_gpx(
            "lib/kurdwanow.gpx")
        average_sm_neighbour_nb = None
        segment_size = None
        eq_pts_num = 100
        initial_sigma = 0.05
        sigma_min = 0.02
        sigma_max = 1
        return set, list, list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def piaski_0():
        set, list = import_gpx(
            "lib/piaski.gpx")
        average_sm_neighbour_nb = None
        segment_size = None
        eq_pts_num = 100
        initial_sigma = 5
        sigma_min = 3
        sigma_max = sigma_min * 2

        return set, list, list, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num

    @staticmethod
    def Spirala_2(noise=False):
        t_min = -4 * np.pi
        t_max = 4 * np.pi

        dt = 2 * np.pi / 12

        x_points = []
        y_points = []
        z_points = []

        t = t_min
        _t = []
        while t <= t_max:
            eps = 0
            if noise:
                eps = random.uniform(-0.1, 0.1)
            x = np.cos(t) * (1 + eps)
            y = np.sin(t) * (1 + eps)
            z = t * (1 + eps) / (2 * np.pi)

            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
            _t.append(t)
            t += dt

        list = [x_points, y_points, z_points]
        set = Array(list).transpose()
        ideal = list
        average_sm_neighbour_nb = None
        segment_size = None
        eq_pts_num = len(x_points)
        initial_sigma = 0.
        sigma_min = 0.
        sigma_max = 0
        return set, list, ideal, average_sm_neighbour_nb, segment_size, initial_sigma, sigma_max, sigma_min, eq_pts_num, _t
