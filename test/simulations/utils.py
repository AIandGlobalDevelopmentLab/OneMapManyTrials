import unittest
import numpy as np
import pandas as pd
from simulations.utils import generate_population, get_ate, estimate_ate_with_pred_dists, calculate_errors, CDF_model

class TestUtils(unittest.TestCase):

    def test_generate_population(self):
        def dummy_img_proxy_func(Y):
            return Y

        population = generate_population(dummy_img_proxy_func, n_samples=100)
        self.assertEqual(len(population), 100)
        self.assertIn('C', population.columns)
        self.assertIn('A', population.columns)
        self.assertIn('Y', population.columns)
        self.assertIn('p_A_given_C', population.columns)
        self.assertIn('X', population.columns)

    def test_get_ate(self):
        t = np.array([1, 0, 1, 0])
        y_preds = np.array([2.0, 1.0, 2.5, 1.5])
        p_t = np.array([0.8, 0.2, 0.8, 0.2])
        ate = get_ate(t, y_preds, p_t)
        self.assertAlmostEqual(ate, 0.0, places=5)

    def test_estimate_ate_with_pred_dists(self):
        y_pred_dists = np.random.rand(100, 1000)
        t_trial = np.random.randint(0, 2, 100)
        p_A_trial = np.random.rand(100)
        ate_estimates = estimate_ate_with_pred_dists(y_pred_dists, t_trial, p_A_trial)
        self.assertEqual(len(ate_estimates), 1000)

    def test_calculate_errors(self):
        ate_estimates = np.random.rand(1000, 100)
        modeling_error, sampling_error = calculate_errors(ate_estimates)
        self.assertTrue(modeling_error > 0)
        self.assertTrue(sampling_error > 0)

    def test_CDF_model(self):
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_cal = np.random.rand(50, 10)
        y_cal = np.random.rand(50)
        X_test = np.random.rand(20, 10)

        model = CDF_model()
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)
        y_pred = model.point_predict(X_test)
        self.assertEqual(len(y_pred), 20)

if __name__ == '__main__':
    unittest.main()