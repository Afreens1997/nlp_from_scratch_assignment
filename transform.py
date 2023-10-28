fi = open('transformed.txt', 'w')
fi1 = open('results.txt', 'r').readlines()

for line in fi1:
	tokens = line.split(" ")
	N = len(tokens)
	i = 0
	while i < N-1:
		# print(, tokens[i+1][1:-1])
		splits = tokens[i+1][1:-1].split(",")
		# print(tokens[i][:-1][1:-1]+"\t"+splits[0]+"\t"+str(splits[1]))
		fi.write(tokens[i][:-1][1:-1]+"\t"+splits[0]+"\t"+str(splits[1])+"\n")
		i += 2
	fi.write("\n")
fi.close()


