import unittest

from fraud_detection_demo import FEATURES, make_synthetic_transactions


class SyntheticDataTests(unittest.TestCase):
    def test_schema_and_row_count(self):
        frame = make_synthetic_transactions(n=1000, seed=7)
        self.assertEqual(frame.shape[0], 1000)
        self.assertTrue(set(FEATURES + ["is_fraud"]).issubset(frame.columns))
        self.assertTrue(set(frame["is_fraud"].unique()).issubset({0, 1}))

    def test_seed_is_reproducible(self):
        one = make_synthetic_transactions(n=1000, seed=17)
        two = make_synthetic_transactions(n=1000, seed=17)
        self.assertTrue(one.equals(two))

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            make_synthetic_transactions(n=50)


if __name__ == "__main__":
    unittest.main()
