""" merge 2 days' gt"""
import sys
import pandas as pd

# read both ground truth files
print "Reading ground truth file"
INPUT_STR = sys.argv[1] + " " + sys.argv[2]
GT1 = pd.read_csv(INPUT_STR.split(" ")[0])
GT2 = pd.read_csv(INPUT_STR.split(" ")[1])

# save the columns order of the first file
COLS = list(GT1)

# reorder columns of the second file
GT2 = GT2[COLS]

# append into one file
GT0 = GT1.append(GT2)

print "Initial GT1 length: ", len(GT1)
print "Initial GT2 length: ", len(GT2)
print "GT merged length: ", len(GT0)

# remove potential duplicates
GT0 = GT0.drop_duplicates().drop_duplicates(subset=["occurred_at"], keep='last')
print "GT merged lengh without duplicates: ", len(GT0)
GT0 = GT0.reset_index()

# save output
print "Saving output."
GT0.to_csv("GT.csv")
