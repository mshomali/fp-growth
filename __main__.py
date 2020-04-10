import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import json
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

with open(THIS_FOLDER + '/dataset.json') as file:
    data = json.load(file)

hits = data['result']['hits']['hit']

dataset = []

for hit in hits:
    temp = []
    if "authors" not in hit['info']:
        continue
    authors = hit['info']['authors']['author']
    if type(authors) is dict:
        temp.append(authors['text'])
    else:
        for author in authors:
            temp.append(author['text'])
    dataset.append(temp)

te = TransactionEncoder()

te_ary = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_ary, columns=te.columns_)

result = fpgrowth(df, min_support=0.007, use_colnames=True)

type2 = type(result)

print(type2)

result = result.values.tolist()

for item in result:
    print(item)
