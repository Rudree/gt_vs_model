import pandas as pd
import sys

# read both ground truth files
print("Reading ground truth file")
input_str=sys.argv[1]+" "+sys.argv[2]
GT1 = pd.read_csv(input_str.split(" ")[0])
GT2 = pd.read_csv(input_str.split(" ")[1])

# save the columns order of the first file
cols=list(GT1)
# reorder columns of the second file
GT2 = GT2[cols]

# append into one file
GT0 = GT1.append(GT2)

print("Initial GT1 length: ", len(GT1))
print("Initial GT2 length: ", len(GT2))
print("GT merged length: ", len(GT0))

# remove potential duplicates
GT0=GT0.drop_duplicates().drop_duplicates(subset=["occurred_at"], keep='first')
print("GT merged lengh without duplicates: ", len(GT0))
GT0=GT0.reset_index()

# save output
print("Saving output.")
GT0.to_csv("GT.csv")
