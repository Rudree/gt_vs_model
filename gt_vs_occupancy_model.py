import sys
import numpy as np
import pandas as pd
import dateutil.parser
from datetime import datetime
from pytz import timezone
from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report

print("------------------------------------------------")
print("gt_vs_occupancy_model.py")
print("--------------")
print("Reading the two datafiles...")
input_str=sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
model_A = pd.read_csv(input_str.split(" ")[0])
model_B = pd.read_csv(input_str.split(" ")[1])

print("Initial length of ",input_str.split(" ")[0],":",len(model_A))
print("Initial length of ",input_str.split(" ")[1],":",len(model_B))

mode=input_str.split(" ")[2]
if(mode != "per_event" and mode != "continuous"):
    print("Wrong mode. Per-event set by default.")
    mode = "per_event"
print("--------------")

def reshape_GT(m):
    df = m
    occurred_at = []
    space_name = []
    state = []

    #save occurred_at timestamps to list
    occurred_at_list = pd.to_datetime(df['occurred_at']).tolist()

    #format occurred_at, UTC to AUT timezone
    occurred_at_list = [ t.tz_localize('utc') for t in occurred_at_list]
    occurred_at_list = [ t.tz_convert('Australia/Victoria') for t in occurred_at_list]
    occurred_at_list = [t.strftime('%Y-%m-%d %H:%M:%S') for t in occurred_at_list]

    #assuming last 8 columns are space names
    space_list = list((df[df.columns[-8:]]).columns.values)

    #add space_name for each timestamps
    for t in occurred_at_list:
        for s in space_list:
            occurred_at.append(t)
            space_name.append(s.lower().replace(" ", "_"))

    #add state at each timestamp for each space
    for index, row in df.iterrows():
        row_states = [row[s] for s in space_list]
        for r in row_states:
            #print r
            state.append(r.upper())

    #add occurred_at, space_name, state to df
    new_df = pd.DataFrame(list(zip(space_name, occurred_at, state)),columns=['space_name','occurred_at', 'state'])

    #change UNOCCUPIED -> UNOCCUPIED INACTIVE
    new_df['state'].replace('UNOCCUPIED','UNOCCUPIED INACTIVE',inplace=True)

    #renamne master_bedroom to bedroom
    new_df['space_name'].replace('master_bedroom','bedroom',inplace=True)

    #renamne medication_area to medication_room
    new_df['space_name'].replace('medication_area','medication_room',inplace=True)

    #renamne space_bedroom to space_room
    new_df['space_name'].replace('spare_bedroom','spare_room',inplace=True)

    return new_df

def check_for_GT(m):
    if ( "space_name" not in m.columns.tolist() ):
        #reshape GT to match occupancy model file
        print("Reshaping Ground Truth file...")
        m = reshape_GT(m)
    else:
        if ( "occurred_at" not in m.columns.tolist() ):
            #resolve column name typo and drop unwanted columns
            print("Remove UNKNOWN and correct typos...")
            m = m.rename(columns={'occured_at': 'occurred_at'})
            unwanted_columns = filter(lambda c: not c  in ["space_name", "occurred_at", "state"], m.columns.tolist())
            m.drop(unwanted_columns, axis=1, inplace=True)
            #drop rows with UNKNOWN state
            m['state'] = m['state'].apply(lambda x: x.strip(' \t\n\r') )
            m = m[m.state != 'UNKNOWN']
            # keep only AVA basic layout
            ava_basic=["bedroom", "bathroom"]
            m=m[m.space_name.isin(ava_basic)]

    return m

#reshape if model = GT
model_A = check_for_GT(model_A)
model_B = check_for_GT(model_B)

print("Length of ",input_str.split(" ")[0],":",len(model_A))
print("Length of ",input_str.split(" ")[1],":",len(model_B))

print("--------------")
if (mode == "per_event"):
    print("Drop some duplicates due to pulse...")
    model_A = model_A.drop_duplicates(subset=["space_name","occurred_at"], keep='last')
    if("m4" not in input_str.split(" ")[1]):
        model_B = model_B.drop_duplicates(subset=["space_name","occurred_at"], keep='last')
    else:
        model_B= model_B.groupby(["space_name","occurred_at"], as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[-2]]).reset_index(level=0, drop=True)
    print("Length of ",input_str.split(" ")[0]," after dropping duplicates:",len(model_A))
    print("Length of ",input_str.split(" ")[1]," after dropping duplicates:",len(model_B))
elif(mode == "continuous"):
    print("Keep all pulse events for comparison...")

print("Rooms in Model A: ", model_A.space_name.unique())
print("Rooms in Model B: ", model_B.space_name.unique())

def merge_dfs():
    model_A.sort_values( by= ['occurred_at'], ascending=True)
    model_B.sort_values( by= ['occurred_at'], ascending=True)
    return pd.merge(left = model_A, right = model_B, left_on=['space_name', 'occurred_at'], right_on = ['space_name', 'occurred_at'], how='inner',suffixes=['_Model_A', '_Model_B'] )


def check_for_m4(m):
    # read the TLM model
    model_C = pd.read_csv("occupancy_results_m3.csv")
    model_C = check_for_GT(model_C)
    model_C = model_C.drop_duplicates(subset=["space_name","occurred_at"], keep='last')
    # select unique timestamp in TLM
    list_timestamps=model_C["occurred_at"].unique()
    # keep only these timestamps in the HMM
    m=m[m["occurred_at"].isin(list_timestamps)]
    return m

print("--------------")
print("Merge model A with model B...")
if("m4" in input_str.split(" ")[0]):
    model_A=check_for_m4(model_A)
elif("m4" in input_str.split(" ")[1]):
    model_B=check_for_m4(model_B)
df = merge_dfs()
#df.to_csv("Merged_output.csv")
print("Length of the merged file: ", len(df))

print("--------------")
print("Get Accuracy results...")
y_true = df['state_Model_A'].tolist()
y_pred = df['state_Model_B'].tolist()

# diff in percentage
count=0
for i in range(0,len(y_true)):
    if(y_pred[i] != y_true[i]):
        count=count+1
print("Difference in % between both models is:", round(100.0*count/len(y_true),2))

print("------")
#create confusion matrix
confusion_matrix = ConfusionMatrix(y_true, y_pred)

#model_A state as reference
classes = np.unique(y_true)
#print classes

#save report for each class
def get_MCC(c_label):
    dict = { c_label : (confusion_matrix.stats()['class'][c_label]).to_dict()['MCC: Matthews correlation coefficient'] }
    return dict

#Matthews correlation coefficient, for each class and overall
classes_MCC = [get_MCC(c)for c in classes]

mccs = 0
for (key_mcc, key_label) in zip(classes_MCC, classes):
    print (key_label , "MCC = ",key_mcc[key_label])
    mccs += key_mcc[key_label]

print ("overall_MCC = ", mccs/len(classes))
print("------")

#precision,recall,f1-score for each class
print ("\n\nclassification report\n", (classification_report(y_true, y_pred)))
