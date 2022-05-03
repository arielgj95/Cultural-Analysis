import pandas as pd
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

os.chdir(r"C:\Users\ariel\Documents\PHD\Datasets\movies")
#file=

np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_rows = 999
pd.options.display.max_columns= 999


file = "IMDb movies.csv"
file2 = "IMDb ratings.csv"
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_rows = 999
pd.options.display.max_columns= 999

mov = pd.read_csv(file, low_memory=False)
rat = pd.read_csv(file2)

print(mov.columns)
#print(mov['imdb_title_id'])
ind=[]

'''
for i in range(len(mov)):
    ind.append(rat['imdb_title_id'][rat['imdb_title_id'] == mov['imdb_title_id'][i]].index[0])
    if i%1000==0:
        print("iteration:",i)
'''
ind.extend(mov.index.values)
data = pd.concat([mov,rat.iloc[ind,:]], axis=1)

data = pd.concat([data['year'], data['date_published'], data['genre'], data["duration"],
                  data['country'], data['language'], data['avg_vote'], data['votes'], data.iloc[:,22:]], axis=1)
data = data.drop(['allgenders_0age_avg_vote','allgenders_0age_votes',
                  'males_0age_avg_vote','males_0age_votes',
                  'females_0age_avg_vote','females_0age_votes'], axis=1)
#print(data.isnull().sum(axis = 0))
data = data.dropna(axis=0)
data = data[data['language']!='None']


genres = []
countries = []
for el in data.iloc():
    if not pd.isna(el['genre']):
        genres.extend(el['genre'].split(','))
    if not pd.isna(el['country']):
        countries.extend(el['country'].split(','))

genres = list(set(genres))
countries = list(set(countries))
print(len(genres),len(countries))

res_country=[]
res_genre=[]
gen_for_country={}
for c in countries:
    if len(data[data.country.apply(lambda x: c in x)])==0:
        continue
    res_country.append([len(data[data.country.apply(lambda x: c in x)]), c])
    for g in genres:
        #print(len(data[data.genre.apply(lambda x: g in x) & data.country.apply(lambda x: c in x)]))
        gen_for_country[(c,g)]=len(data[data.genre.apply(lambda x: g in x) & data.country.apply(lambda x: c in x)])
        #print("GEN: ", g,"with:",res_country[-1], " Counrty:",c, "gen_for_country:",gen_for_country[(c,g)])
    #print("number of",c,":",len(data[data['country']==c]), "        ")

#print("\n \n")
for c in genres:
    if len(data[data.genre.apply(lambda x: c in x)])==0:
        continue
    res_genre.append([len(data[data.genre.apply(lambda x: c in x)]), c])
    #print("number of", c, ":", len(data[data['genre'] == c]), "        ")

res_country.sort(key=lambda row: row[0],reverse=True)
res_genre.sort(key=lambda row: row[0],reverse=True)
#res_genre.append( ["        "] * (len(res_country)-len(res_genre)))     ###only for printing purpose

'''
for i,c in enumerate(res_country):
    if i<len(res_genre):
        print("Country", c[1], ": ", c[0], "        ", "Genre", res_genre[i][1], ": ", res_genre[i][0])
    else:
        print("Country", c[1], ": ", c[0])
'''

for i,c in enumerate(res_country):
    print("Country", c[1], ":", c[0], end="   ")
    for l,g in enumerate(res_genre):
        print("Genre", res_genre[l][1], ":", gen_for_country[(c[1],g[1])], end="  ")
    print("\n")
'''
### HOTEL
data = pd.read_csv(file)
data = data.drop(['agent','company'],axis=1)
data = data.dropna(axis=0)
print(data.isnull().sum(axis = 0))

for c in data['country'].unique():
    print("number of",c,":",len(data[data['country']==c]), "        ",
          "number of",c,"with is_canceled=1:", len(data[(data['country']==c) & (data['is_canceled']==1)]), "        ",
          "number of",c,"with is_canceled=0:", len(data[(data['country']==c) & (data['is_canceled']==0)]))
            
            
### MOVIES          
file = "IMDb movies.csv"
file2 = "IMDb ratings.csv"
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_rows = 999
pd.options.display.max_columns= 999

mov = pd.read_csv(file, low_memory=False)
rat = pd.read_csv(file2)

print(mov.columns)
#print(mov['imdb_title_id'])
ind=[]


for i in range(len(mov)):
    ind.append(rat['imdb_title_id'][rat['imdb_title_id'] == mov['imdb_title_id'][i]].index[0])
    if i%1000==0:
        print("iteration:",i)

ind.extend(mov.index.values)
data = pd.concat([mov,rat.iloc[ind,:]], axis=1)

data = pd.concat([data['year'], data['date_published'], data['genre'], data["duration"],
                  data['country'], data['language'], data['avg_vote'], data['votes'], data.iloc[:,22:]], axis=1)
data = data.drop(['allgenders_0age_avg_vote','allgenders_0age_votes',
                  'males_0age_avg_vote','males_0age_votes',
                  'females_0age_avg_vote','females_0age_votes'], axis=1)
#print(data.isnull().sum(axis = 0))
data = data.dropna(axis=0)
data = data[data['language']!='None']


genres = []
countries = []
for el in data.iloc():
    if not pd.isna(el['genre']):
        genres.extend(el['genre'].split(','))
    if not pd.isna(el['country']):
        countries.extend(el['country'].split(','))

genres = set(genres)
genres = list(genres)
countries = set(countries)
countries  = list(countries )
print(len(genres),len(countries))

res_country=[]
res_genre=[]
for c in countries:
    if len(data[data.country.apply(lambda x: c in x)])==0:
        continue
    res_country.append([len(data[data.country.apply(lambda x: c in x)]), c])
    #print("number of",c,":",len(data[data['country']==c]), "        ")

#print("\n \n")
for c in genres:
    if len(data[data.genre.apply(lambda x: c in x)])==0:
        continue
    res_genre.append([len(data[data.genre.apply(lambda x: c in x)]), c])
    #print("number of", c, ":", len(data[data['genre'] == c]), "        ")
res_country.sort(key=lambda row: row[0],reverse=True)
res_genre.sort(key=lambda row: row[0],reverse=True)
#res_genre.append( ["        "] * (len(res_country)-len(res_genre)))     ###only for printing purpose

for i,c in enumerate(res_country):
    if i<len(res_genre):
        print("Country", c[1], ": ", c[0], "        ", "Genre", res_genre[i][1], ": ", res_genre[i][0])
    else:
        print("Country", c[1], ": ", c[0])
           

### LOANS       
data = pd.read_csv(file) 
data = data.drop(['id', 'country_code', 'region', 'posted_time', 'funded_time', 'tags'],axis=1)
data = data.dropna(axis=0)
data['funded'] = data["loan_amount"] == data["funded_amount"]
#data["funded"] = data["funded"].astype('bool')
print(len(data[data['funded']==True]))
print(len(data[data['funded']==False]))

res_country=[]
for c in data['country'].unique():
    res_country.append([len(data[data['country']==c]), len(data[(data['country']==c) & (data['funded']==True)]),
                        len(data[(data['country']==c) & (data['funded']==False)]), c])

res_country.sort(key=lambda row: row[0], reverse=True)

for i, c in enumerate(res_country):
    print("Country",c[3],":",c[0], "        ",
          "Country",c[3],"with funded=1:", c[1], "        ",
          "Country",c[3],"with funded=0:", c[2])

### BIG 5 PERSONALITY
data = pd.read_csv(file, sep='\t')
data = data[data['IPC']==1]
#print(len(data[data['IPC']==1]))
data = data.drop(['dateload', 'screenw', 'screenh','lat_appx_lots_of_err','long_appx_lots_of_err'], axis=1)
data = data.dropna(axis=0)
print(data.isnull().sum(axis = 0))
print(len(data))
print(data.columns)

res_country=[]
for c in data['country'].unique():
    res_country.append([len(data[data['country']==c]), c])

res_country.sort(key=lambda row: row[0],reverse=True)

for i,c in enumerate(res_country):
    if i<len(data.columns[0:50]):
        print("Country", c[1], ": ", c[0], "        ", data.columns[i],
              " mean: ", round(np.mean(data.iloc[:,i]),2), " std: ", round(np.std(data.iloc[:,i]),2))
    else:
        print("Country", c[1], ": ", c[0])          
          
### CUISINE   
with open(file,"r") as f:
    data = json.load(f)       
cuisine=[]
#ingr=[]
predictions=['sugar', 'salt','butter','peanut','olive oil','chicken','veal','pork']
for i in range(len(data)):
    cuisine.append(data[i]['cuisine'])
    #ingr.append(data[i]['ingredients'])

cuisine_unique=np.unique(cuisine)

cuisine_ordered=[]
for c in cuisine_unique:
    cuisine_ordered.append([cuisine.count(c), c])

cuisine_ordered.sort(key=lambda row: row[0], reverse=True)

for c in cuisine_ordered:
    print(f"CUISINE {c[1]}:", cuisine.count(c[1]), end="   ")
    for p in predictions:
        #print(counter_IngCui(p,c,data))
        print(p, ":", counter_IngCui(p,c[1],data), end="   ")
    print("\n")
'''