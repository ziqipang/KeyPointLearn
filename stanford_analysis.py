import json
import os
import numpy as np
import matplotlib.pyplot as plt


def class_wise_occurrence():
    src_dir = 'data/Stanford/'

    class_name_file = src_dir + 'class_names.json'
    f = open(class_name_file, 'r')
    class_names = json.load(f)
    f.close()

    data_final_file = src_dir + 'data_final.json'
    f = open(data_final_file, 'r')
    data = json.load(f)
    f.close()

    occurrences = np.zeros(40)

    for person in data:
        occurrences[person['label']] += 1

    plt.bar(np.arange(40), occurrences, tick_label=class_names, color='green')
    plt.xticks(rotation=270)

    plt.show()


if __name__ == '__main__':
    class_wise_occurrence()
