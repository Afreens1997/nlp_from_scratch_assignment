import pandas as pd
df = pd.read_csv('private_test.csv')

tokens = []
curr_sentence = []
for word in df['input']:
  if word != word:
    tokens.append(curr_sentence)
    curr_sentence = []
  else:
    curr_sentence.append(word)
if len(curr_sentence) > 0:
  tokens.append(curr_sentence)

from transformers import pipeline
model_path = "./scibert-nlp/checkpoint-1212"
classifier = pipeline("ner", model=model_path)
fi = open("results.txt", "w")

for text in tokens:
  try:
    # text = tokens[2]
    # print(text)

    output = classifier(' '.join(text))
    # print(output)

    data = text
    data_o = []
    start = 0
    for word in data:
      data_o.append((word, start, start+len(word)-1))
      start += len(word)+1
    # print(data_o)

    final = {}
    ind = -1
    for word, a, b in data_o:
      ind += 1
      for entity in output:
        if entity['start'] > b and (word,ind) not in final:
          final[(word,ind)] = ('O', 0)
        if entity['start'] >= a and entity['start'] <= b:
          if ((word,ind) not in final) or ((word,ind) in final and final[(word,ind)][1] < entity['score']):
            final[(word,ind)] = (entity['entity'], entity['score'])

        if entity['end']-1 >= a and entity['end']-1 <= b:
          if (word,ind) not in final or ((word,ind) in final and final[(word,ind)][1] < entity['score']):
            final[(word,ind)] = (entity['entity'], entity['score'])

    string = ""

    for i, tok in enumerate(text):
      word, ind = tok, i
      if (word,ind) in final:
        string += "("+word+"): (" + str(final[(word,ind)][0])+","+str(final[(word,ind)][1])+") "
      else:
        string += "("+word+"): (" + str('O')+","+str(0)+") "
    fi.write(string+"\n")
  except:
    string = ""
    for i, tok in enumerate(text):
      word, ind = tok, i
      string += "("+word+"): (" + str('O')+","+str(-1)+") "
    fi.write(string+"\n")
fi.close()