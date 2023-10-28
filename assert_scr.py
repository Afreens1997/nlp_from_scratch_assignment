import pandas as pd

df = pd.read_csv("private_test.csv")
fi = open("transformed.txt").readlines()

# for t1, t2 in zip(df['input'], fi):

# 	if t1 != t1:
# 		try:
# 			if t1 is None:
# 				pass
# 			assert t2 == "\n"
# 		except Exception as e:
			
# 			if t2.split("\t")[1] == 'I-MethodName' and t2.split("\t")[2] == "0.77982926\n":
# 				pass
# 			elif t2.split("\t")[1] == 'B-MethodName' and t2.split("\t")[2] == "0.62165\n":
# 				pass
# 			else:
# 				print(t1, t2)
# 				raise e
# 	else:
# 		assert t1 == t2.split("\t")[0]

print(len(fi), len(df['input']))
labels = []
ind = 1
for line in fi:
	if line == '\n':
		labels.append([ind, "X"])
	else:
		vals = line.split()
		if "-DOCSTART-" == vals[0]:
			labels.append([ind,"O"])
		else:
			labels.append([ind,vals[1]])
	ind += 1

pd.DataFrame(labels, columns=["id", "target"]).to_csv("submission.csv",index=None)