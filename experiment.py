import json


f = open('data/Stanford/data_final.json', 'r')
data = json.load(f)
f.close()

class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

new_data = list()

for person in data:
    if person['label'] in class_list:
        new_data.append(person)

f = open('data/Stanford/data_subset.json', 'w')
json.dump(new_data, f)
f.close()
