# Files:
The (minimum) required data files are:
 
- `P6-40-correct.csv`: Ground truth file for Day 40 of participant 6
- `P6-41-correct.csv`: Ground truth file for Day 41 of participant 6
- `occupancy_results_m1.csv`: Occupancy output for the DF model
- `occupancy_results_m3.csv`: Occupancy output for the TLM model
- `occupancy_results_m4.csv`: Occupancy output for the HMM model

The required python script files are:

- `merge_GT.py`
- `gt_vs_occupancy_model.py`

# Python libraries
The version of python used is 3.5. Packages required are:

- sys
- numpy
- pandas
- dateutil.parser
- datetime
- pytz
- pandas_ml
- sklearn

# Scripts
## merge_GT
This script merges two days of ground truth such as `Day1.csv` and `Day2.csv` into a single file `GT.csv`. The ground truth file can then be used to compare with the model outputs with `gt_vs_occupancy_model`.
To execute:
`python merge_GT.py Day1.csv Day2.csv`

## gt_vs_occupancy_model
This script compares the output of any two model csv outputs, `model_A.csv` and `model_B.csv` (either Ground truth or AI models). `model_A` is used as the reference model. The script will identify the common and unique `occurred_at` timestamps between both models and then estimate:
- the difference between the two models in %
- the Matthews Correlation Coefficient by occupancy states and overall
- the precision, recall and f1-score per occupancy states

To execute:
`python gt_vs_occupancy_model.py model_A.csv model_B.csv`
