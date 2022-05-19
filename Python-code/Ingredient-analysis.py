import json
import numpy as np
import os
import sys

def counter_IngCui(ingredient,cuisine,data):
    count=0
    for el in data:
        if cuisine != el["cuisine"]:
            continue
        for ing in el["ingredients"]:
            if ingredient in ing:
                count+=1
                break
    return(count)


def counter(word,data):
    count=0
    for el in data:
        for ing in el["ingredients"]:
            if word in ing:
                count+=1
                break
    return(count)


np.set_printoptions(threshold=sys.maxsize)

os.chdir(r"C:\Users\ariel\Documents\GitHub\Cultural-Analysis\Datasets\whats-cooking\train.json")
file = "train.json"

with open(file,"r") as f:
    data = json.load(f)

cuisine=[]
ingr=[]
ingr_results=[]
Y = ['sugar', 'salt','butter','peanut','olive oil','chicken','veal','pork']
X = []

for i in range(len(data)):
    cuisine.append(data[i]['cuisine'])
    ingr.extend(data[i]['ingredients'])

cuisine_unique = np.unique(cuisine)
print(len(np.unique(ingr)))
#print(ingr[:100])
for ing in np.unique(ingr):
    ingr_results.append([counter(ing,data),ing])

ingr_results.sort(key=lambda row: row[0], reverse=True)
print(ingr_results)

cuisine_ordered=[]
for c in cuisine_unique:
    cuisine_ordered.append([cuisine.count(c), c])

cuisine_ordered.sort(key=lambda row: row[0], reverse=True)

'''
for c in cuisine_ordered:
    print(f"CUISINE {c[1]}:", cuisine.count(c[1]), end="   ")
    for p in predictions:
        print(counter_IngCui(p,c,data))
        print(p, ":", counter_IngCui(p,c[1],data), end="   ")
    print("\n")
'''