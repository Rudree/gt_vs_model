"""
compare GT/MODEl with MODEl
"""
#pylint: disable=line-too-long, deprecated-lambda
import sys
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report

print "------------------------------------------------"
print "gt_vs_occupancy_MODEl.py"
print "--------------"
print "Reading the two datafiles..."

INPUT_STR = sys.argv[1] + " "+sys.argv[2] + " " + sys.argv[3]
MODEL_A = pd.read_csv(INPUT_STR.split(" ")[0])
MODEL_B = pd.read_csv(INPUT_STR.split(" ")[1])

print "Initial length of ", INPUT_STR.split(" ")[0], ":", len(MODEL_A)
print "Initial length of ", INPUT_STR.split(" ")[1], ":", len(MODEL_B)

MODE = INPUT_STR.split(" ")[2]
if MODE != "per_event" and MODE != "continuous":
    print "Wrong MODE. Per-event set by default."
    MODE = "per_event"
print "--------------"

def reshape_gt(_model):
    """ reshape gt """
    _df = _model
    occurred_at = []
    space_name = []
    state = []

    #save occurred_at timestamps to list
    occurred_at_list = pd.to_datetime(_df['occurred_at']).tolist()

    #format occurred_at, UTC to AUT timezone
    occurred_at_list = [t.tz_localize('utc') for t in occurred_at_list]
    occurred_at_list = [t.tz_convert('Australia/Victoria') for t in occurred_at_list]
    occurred_at_list = [t.strftime('%Y-%m-%d %H:%M:%S') for t in occurred_at_list]

    #assuming last 8 columns are space names
    space_list = list((_df[_df.columns[-8:]]).columns.values)

    #add space_name for each timestamps
    for time in occurred_at_list:
        for space in space_list:
            occurred_at.append(time)
            space_name.append(space.lower().replace(" ", "_"))

    #add state at each timestamp for each space
    for index, row in _df.iterrows():#pylint: disable=unused-variable
        row_states = [row[s] for s in space_list]
        for _row in row_states:
            #print r
            state.append(_row.upper())

    #add occurred_at, space_name, state to df
    new_df = pd.DataFrame(list(zip(space_name, occurred_at, state)), columns=['space_name', 'occurred_at', 'state'])

    #change UNOCCUPIED -> UNOCCUPIED INACTIVE
    new_df['state'].replace('UNOCCUPIED', 'UNOCCUPIED INACTIVE', inplace=True)

    #renamne master_bedroom to bedroom
    new_df['space_name'].replace('master_bedroom', 'bedroom', inplace=True)

    #renamne medication_area to medication_room
    new_df['space_name'].replace('medication_area', 'medication_room', inplace=True)

    #renamne space_bedroom to space_room
    new_df['space_name'].replace('spare_bedroom', 'spare_room', inplace=True)

    return new_df

def check_for_gt(model):
    """ check model """
    if "space_name" not in model.columns.tolist():
        #reshape GT to match occupancy MODEl file
        print "Reshaping Ground Truth file..."
        model = reshape_gt(model)
    else:
        if "occurred_at" not in model.columns.tolist():
            #resolve column name typo and drop unwanted columns
            print "Remove UNKNOWN and correct typos..."
            model = model.rename(columns={'occured_at': 'occurred_at'})
            unwanted_columns = filter(lambda c: not c  in ["space_name", "occurred_at", "state"], model.columns.tolist())
            model.drop(unwanted_columns, axis=1, inplace=True)
            #drop rows with UNKNOWN state
            model['state'] = model['state'].apply(lambda x: x.strip(' \t\n\r'))
            model = model[model.state != 'UNKNOWN']
            # keep only AVA basic layout
            ava_basic = ["bedroom", "bathroom"]
            model = model[model.space_name.isin(ava_basic)]

    return model

#reshape if MODEl = GT
MODEL_A = check_for_gt(MODEL_A)
MODEL_B = check_for_gt(MODEL_B)

print "Length of ", INPUT_STR.split(" ")[0], ":", len(MODEL_A)
print "Length of ", INPUT_STR.split(" ")[1], ":", len(MODEL_B)

print "--------------"
if MODE == "per_event":
    print "Drop some duplicates due to pulse..."
    MODEL_A = MODEL_A.drop_duplicates(subset=["space_name", "occurred_at"], keep='last')
    if "m4" not in INPUT_STR.split(" ")[1]:
        MODEL_B = MODEL_B.drop_duplicates(subset=["space_name", "occurred_at"], keep='last')
    else:
        MODEL_B = MODEL_B.groupby(["space_name", "occurred_at"], as_index=False).apply(lambda x: x if len(x) == 1 else x.iloc[[-2]]).reset_index(level=0, drop=True)
    print "Length of ", INPUT_STR.split(" ")[0], " after dropping duplicates:", len(MODEL_A)
    print "Length of ", INPUT_STR.split(" ")[1], " after dropping duplicates:", len(MODEL_B)
elif MODE == "continuous":
    print "Keep all pulse events for comparison..."

print "Rooms in MODEl A: ", MODEL_A.space_name.unique()
print "Rooms in MODEl B: ", MODEL_B.space_name.unique()

def merge_dfs():
    """ merge dfs """
    MODEL_A.sort_values(by=['occurred_at'], ascending=True)
    MODEL_B.sort_values(by=['occurred_at'], ascending=True)
    return pd.merge(left=MODEL_A, right=MODEL_B, left_on=['space_name', 'occurred_at'], right_on=['space_name', 'occurred_at'], how='inner', suffixes=['_MODEL_A', '_MODEL_B'])


def check_for_m4(model_):
    """ check for m4 """
    # read the TLM MODEl
    model_c = pd.read_csv("occupancy_results_m3.csv")
    model_c = check_for_gt(model_c)
    model_c = model_c.drop_duplicates(subset=["space_name", "occurred_at"], keep='last')
    # select unique timestamp in TLM
    list_timestamps = model_c["occurred_at"].unique()
    # keep only these timestamps in the HMM
    model_ = model_[model_["occurred_at"].isin(list_timestamps)]
    return model_

print "--------------"
print "Merge MODEl A with MODEl B..."
if "m4" in INPUT_STR.split(" ")[0]:
    MODEL_A = check_for_m4(MODEL_A)
elif "m4" in INPUT_STR.split(" ")[1]:
    MODEL_B = check_for_m4(MODEL_B)
DFS = merge_dfs()
#df.to_csv("Merged_output.csv")
print "Length of the merged file: ", len(DFS)

print "--------------"
print "Get Accuracy results..."
Y_TRUE = DFS['state_MODEL_A'].tolist()
Y_PRED = DFS['state_MODEL_B'].tolist()

# diff in percentage
COUNT = 0
for i in range(0, len(Y_TRUE)):
    if Y_PRED[i] != Y_TRUE[i]:
        COUNT = COUNT+1
print "Difference in % between both MODEls is:", round(100.0*COUNT/len(Y_TRUE), 2)

print "------"
#create confusion matrix
CONFUSION_MATRIX = ConfusionMatrix(Y_TRUE, Y_PRED)

#MODEL_A state as reference
CLASSES = np.unique(Y_TRUE)
#print CLASSES

#save report for each class
def get_mcc(c_label):
    """ get Matthews correlation coefficient """
    dict = {c_label : (CONFUSION_MATRIX.stats()['class'][c_label]).to_dict()['MCC: Matthews correlation coefficient']} #pylint: disable=redefined-builtin
    return dict

#Matthews correlation coefficient, for each class and overall
CLASSES_MCC = [get_mcc(c)for c in CLASSES]

MCCS = 0
for (key_mcc, key_label) in zip(CLASSES_MCC, CLASSES):
    print key_label, "MCC = ", key_mcc[key_label]
    MCCS += key_mcc[key_label]

print "overall_MCC = ", MCCS/len(CLASSES)
print "------"

#precision,recall,f1-score for each class
print "\n\nclassification report\n", (classification_report(Y_TRUE, Y_PRED))
