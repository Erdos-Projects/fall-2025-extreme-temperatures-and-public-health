#runs all modelling scripts in sequence
import os, runpy
base = os.path.dirname(__file__)
steps = [
  '1_prep_data.py',
  '2_train_test_split.py',
  '3_train_models.py',
  '4_test_and_plot.py',
  '5_goodness_of_fit_straight_line.py',
  '6_goodness_of_fit_second_order.py',
  '7_statistical_tests.py',
  '8_demonstration.py',
  '9_simple_model_comparison.py']

for i in steps:
    print(f"Running script {i}")
    runpy.run_path(os.path.join(base, i))