import itertools

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("error")

lookup_hp = {
    'LightGCN': ['factors', 'n_layers', 'lr'],
    'NGCF': ['factors', 'n_layers', 'lr'],
    'ItemKNN': ['nn', 'sim'],
    'UserKNN': ['nn', 'sim'],
    'BPRMF': ['f', 'lr'],
    'NeuMF': ['factors', 'lr']
}


class ObjectivesSpace:
    def __init__(self, df, functions, model_name, norm):
        self.model_name = model_name
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()
        self.norm = norm

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return pd.DataFrame(self.points[0], columns=self._constr_obj()).sort_values(by=list(self.functions.keys())[1])

    def get_dominated(self):
        return pd.DataFrame(self.points[1], columns=self._constr_obj())

    def get_nondominated_per_hp(self, norm=False):
        temp_dict = {}
        return_dict = {}
        if norm:
            pts = self.get_nondominated_norm()
        else:
            pts = self.get_nondominated()
        hps = lookup_hp[self.model_name]
        for hp in hps:
            pts[hp] = pts['model'].map(lambda x: x[x.find(hp) + len(str(hp) + '='):].split('_')[0])
            temp_dict[hp] = list(pts[hp].unique())
            return_dict[hp] = {}
        for hp in hps:
            hp_values = temp_dict[hp]
            for value in hp_values:
                return_dict[hp][value] = pts.loc[pts[hp] == value]
        return return_dict

    def get_dominated_per_hp(self, norm=False):
        temp_dict = {}
        return_dict = {}
        if norm:
            pts = self.get_dominated_norm()
        else:
            pts = self.get_dominated()
        hps = lookup_hp[self.model_name]
        for hp in hps:
            pts[hp] = pts['model'].map(lambda x: x[x.find(hp) + len(str(hp) + '='):].split('_')[0])
            temp_dict[hp] = list(pts[hp].unique())
            return_dict[hp] = {}
        for hp in hps:
            hp_values = temp_dict[hp]
            for value in hp_values:
                return_dict[hp][value] = pts.loc[pts[hp] == value]
        return return_dict

    @staticmethod
    def create_line(pts):
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def _get_distances(self, line_pts, pts):
        non_dom_line = line_pts.copy()
        non_dom_line = self.order_points([list(x) for x in list(non_dom_line)])
        line = self.create_line(non_dom_line)
        all_pts = pts.copy()
        distances = {}
        i = 0
        for point in all_pts:
            try:
                distances[(i, tuple(point))] = min([self.point_to_segment_distance(point, x[0], x[1]) for x in line])
            except ValueError:
                try:
                    distances[(i, tuple(point))] = self.compute_norm(point, np.array(non_dom_line[0]), self.norm)
                except ValueError:
                    pass
            i += 1
        return distances


    # @staticmethod
    '''
    def point_to_segment_distance(self, P, A, B):
        # Vector from A to B
        AB = B - A
        # Vector from A to P
        AP = P - A
        # Project point P onto the line AB
        t = np.dot(AP, AB) / np.dot(AB, AB)
        # Clamp t to the range [0, 1] to find the closest point on the segment
        t = max(0, min(1, t))
        # Closest point on the segment
        closest_point = A + t * AB
        # Distance from P to the closest point on the segment
        # distance = np.linalg.norm(P - closest_point)
        distance = self.compute_norm(P, closest_point, self.norm)
        return distance
    '''

    def point_to_segment_distance(self, P, A, B):
        """
        Calculates the EXACT distance from point P to segment [A, B]
        for L1, L2, and L-inf norms without numerical approximation.
        """
        AB = B - A
        # Vector P - A
        PA = P - A

        # --- CASE 1: Euclidean (L2) ---
        if self.norm == 2 or self.norm == 'l2':
            denom = np.dot(AB, AB)
            if denom == 0:  # A and B are the same point
                return np.linalg.norm(PA)
            t = np.dot(PA, AB) / denom
            t = np.clip(t, 0, 1)
            closest = A + t * AB
            return np.linalg.norm(P - closest)

        # --- CASE 2: Manhattan (L1) ---
        elif self.norm == 1 or self.norm == 'l1':
            # Candidates are 0, 1, and the 'kinks' where error in one dim is 0.
            # Solve: PA[i] - t * AB[i] = 0  =>  t = PA[i] / AB[i]

            # Avoid division by zero where AB is 0 (segment is flat in that dim)
            with np.errstate(divide='ignore', invalid='ignore'):
                t_candidates = PA / AB

            # Filter valid candidates
            valid_t = t_candidates[np.isfinite(t_candidates)]
            valid_t = valid_t[(valid_t > 0) & (valid_t < 1)]

            # Add boundaries
            candidates = np.concatenate(([0.0, 1.0], valid_t))

            # Evaluate distance at all candidates and take the min
            min_dist = float('inf')
            for t in candidates:
                # Calculate L1 distance at this t
                dist = np.sum(np.abs(PA - t * AB))
                if dist < min_dist:
                    min_dist = dist
            return min_dist

        # --- CASE 3: Chebyshev (L-Infinity) ---
        elif self.norm == np.inf or self.norm == 'inf':
            # Candidates are 0, 1, and intersections where |err_i| == |err_j|
            candidates = [0.0, 1.0]
            dims = len(P)

            # We must check pairs of dimensions (i, j)
            # Eq 1: (PA_i - t*AB_i) = (PA_j - t*AB_j)  => t(AB_j - AB_i) = PA_j - PA_i
            # Eq 2: (PA_i - t*AB_i) = -(PA_j - t*AB_j) => t(AB_j + AB_i) = PA_j + PA_i

            for i, j in itertools.combinations(range(dims), 2):
                # Check Intersection 1
                denom1 = AB[j] - AB[i]
                if denom1 != 0:
                    t1 = (PA[j] - PA[i]) / denom1
                    if 0 < t1 < 1: candidates.append(t1)

                # Check Intersection 2
                denom2 = AB[j] + AB[i]
                if denom2 != 0:
                    t2 = (PA[j] + PA[i]) / denom2
                    if 0 < t2 < 1: candidates.append(t2)

            # Evaluate distance at all candidates
            min_dist = float('inf')
            for t in candidates:
                # Calculate L-inf distance at this t
                # dist = max(|PA - t*AB|)
                dist = np.max(np.abs(PA - t * AB))
                if dist < min_dist:
                    min_dist = dist
            return min_dist

        else:
            raise NotImplementedError("Only L1, L2, and L-inf supported exactly.")

    def compute_norm(self, point1, point2, norm):
        # Compute the Euclidean distance
        distance = np.linalg.norm(point1 - point2, ord=norm)
        return distance

    def order_points(self, points):
        """Order a list of 3D points to form a continuous line."""
        if len(points) < 2:
            return points
        ordered_points = [points[0]]  # Start with the first point
        remaining_points = points[1:]
        while remaining_points:
            # Find the nearest neighbor to the last point in the ordered list
            last_point = ordered_points[-1]
            nearest_point = min(remaining_points, key=lambda p: self.compute_norm(np.array(last_point), np.array(p), self.norm))
            # Add the nearest point to the ordered list and remove it from remaining points
            ordered_points.append(nearest_point)
            remaining_points.remove(nearest_point)
        return np.array([np.array(xi) for xi in ordered_points])

    def _minmax_normalization(self, pts, line_pts, all_pts):
        non_dom_line = line_pts.copy()
        all_pts = all_pts.copy()
        pts = pts.copy()
        for i in range(0, all_pts.shape[1]):
            try:
                pts[:, i] = (pts[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
            except RuntimeWarning:
                pts[:, i] = 0
            except IndexError:
                pass
            try:
                non_dom_line[:, i] = (non_dom_line[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
            except RuntimeWarning:
                non_dom_line[:, i] = 0
        return non_dom_line, pts

    def mean_std(self, distances):
        mean = np.fromiter(distances.values(), dtype=float).mean()
        variance = ((np.fromiter(distances.values(), dtype=float) - mean) ** 2).sum() / (
                np.fromiter(distances.values(), dtype=float).shape[0] - 1)
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def get_nondominated_norm(self):
        non_dom, dom = self.points[0].copy(), self.points[1].copy()
        pts = np.concatenate((self.points[0], self.points[1]))
        non_dom[:, 1:] = self._minmax_normalization(pts[:, 1:].astype('float'), non_dom[:, 1:].astype('float'),
                                                    np.concatenate((non_dom[:, 1:].astype('float'),
                                                                    dom[:, 1:].astype('float'))))[0]
        return pd.DataFrame(non_dom, columns=self._constr_obj()).sort_values(by=list(self.functions.keys())[1])

    def get_dominated_norm(self):
        non_dom, dom = self.points[0].copy(), self.points[1].copy()
        pts = np.concatenate((self.points[0], self.points[1]))
        non_dom[:, 1:], pts[:, 1:] = self._minmax_normalization(pts[:, 1:].astype('float'),
                                                                non_dom[:, 1:].astype('float'),
                                                                np.concatenate((non_dom[:, 1:].astype('float'),
                                                                                dom[:, 1:].astype('float'))))
        non_dom = set(map(tuple, non_dom))
        dom = np.array([row for row in pts if tuple(row) not in non_dom])
        return pd.DataFrame(dom, columns=self._constr_obj())

    def get_statistics(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        pts = np.concatenate((self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')))
        if normalization:
            line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
            distances = self._get_distances(line_pts, all_pts)
        else:
            distances = self._get_distances(non_dom, pts)
        # print(np.fromiter(distances.values(), dtype=float))
        return self.mean_std(distances)


    def get_statistics_per_hp(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        non_dom_hp, dom_hp = self.get_nondominated_per_hp(), self.get_dominated_per_hp()
        hps = lookup_hp[self.model_name]
        stats_hp = {}
        for hp in hps:
            stats_hp[hp] = {}
            values = set()
            for el in dom_hp[hp].keys():
                values.add(el)
            for el in non_dom_hp[hp].keys():
                values.add(el)
            for value in values:
                try:
                    if non_dom.shape[1] == 2:
                        pts = np.concatenate((non_dom_hp[hp][value].values[:, 1:3].astype('float'), dom_hp[hp][value].values[:, 1:3].astype('float')))
                    elif non_dom.shape[1] == 3:
                        pts = np.concatenate((non_dom_hp[hp][value].values[:, 1:4].astype('float'),
                                              dom_hp[hp][value].values[:, 1:4].astype('float')))
                except KeyError:
                    try:
                        if non_dom.shape[1] == 2:
                            pts = non_dom_hp[hp][value].values[:, 1:3].astype('float')
                        elif non_dom.shape[1] == 3:
                            pts = non_dom_hp[hp][value].values[:, 1:4].astype('float')
                    except KeyError:
                        if non_dom.shape[1] == 2:
                            pts = dom_hp[hp][value].values[:, 1:3].astype('float')
                        elif non_dom.shape[1] == 3:
                            pts = dom_hp[hp][value].values[:, 1:4].astype('float')
                if normalization:
                    line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
                    distances = self._get_distances(line_pts, all_pts)
                else:
                    distances = self._get_distances(non_dom, pts)
                stats_hp[hp][value] = self.mean_std(distances)
        return stats_hp


    def get_distances(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        pts = np.concatenate((self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')))
        if normalization:
            line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
            distances = self._get_distances(line_pts, all_pts)
        else:
            distances = self._get_distances(non_dom, pts)
        return np.fromiter(distances.values(), dtype=float)


    def get_distances_per_hp(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        non_dom_hp, dom_hp = self.get_nondominated_per_hp(), self.get_dominated_per_hp()
        hps = lookup_hp[self.model_name]
        stats_hp = {}
        for hp in hps:
            stats_hp[hp] = {}
            values = set()
            for el in dom_hp[hp].keys():
                values.add(el)
            for el in non_dom_hp[hp].keys():
                values.add(el)
            for value in values:
                try:
                    if non_dom.shape[1] == 2:
                        pts = np.concatenate((non_dom_hp[hp][value].values[:, 1:3].astype('float'), dom_hp[hp][value].values[:, 1:3].astype('float')))
                    elif non_dom.shape[1] == 3:
                        pts = np.concatenate((non_dom_hp[hp][value].values[:, 1:4].astype('float'),
                                              dom_hp[hp][value].values[:, 1:4].astype('float')))
                except KeyError:
                    try:
                        if non_dom.shape[1] == 2:
                            pts = non_dom_hp[hp][value].values[:, 1:3].astype('float')
                        elif non_dom.shape[1] == 3:
                            pts = non_dom_hp[hp][value].values[:, 1:4].astype('float')
                    except KeyError:
                        if non_dom.shape[1] == 2:
                            pts = dom_hp[hp][value].values[:, 1:3].astype('float')
                        elif non_dom.shape[1] == 3:
                            pts = dom_hp[hp][value].values[:, 1:4].astype('float')
                if normalization:
                    line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
                    distances = self._get_distances(line_pts, all_pts)
                else:
                    distances = self._get_distances(non_dom, pts)
                stats_hp[hp][value] = distances
        return stats_hp


