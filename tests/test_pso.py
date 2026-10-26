"""Lightweight smoke tests for PSO helpers."""
import math
import unittest
from relief_optimization.pso import haversine_km, get_bounds_from_sites, fitness_function


class PSOTests(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_km(26.91, 75.79, 26.91, 75.79), 0)

    def test_symmetric_distance(self):
        self.assertAlmostEqual(haversine_km(26.91, 75.79, 26.93, 75.82),
                               haversine_km(26.93, 75.82, 26.91, 75.79))

    def test_bounds(self):
        sites = [{"lat": 26.90, "lon": 75.80}, {"lat": 27.00, "lon": 75.90}]
        bounds = get_bounds_from_sites(sites, 0.01, 0.02)
        for observed, expected in zip(bounds, (26.89, 27.01, 75.78, 75.92)):
            self.assertAlmostEqual(observed, expected)

    def test_fitness_finite(self):
        sites = [{"lat": 26.91, "lon": 75.79, "priority": 7},
                 {"lat": 26.93, "lon": 75.82, "priority": 9}]
        score = fitness_function([26.915, 75.795, 26.925, 75.815],
                                 sites, supply_vector=[400, 500])
        self.assertTrue(math.isfinite(score))


if __name__ == "__main__":
    unittest.main()
