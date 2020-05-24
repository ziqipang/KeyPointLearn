import argparse
import json
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='../Stanford_Raw/', help='where you store the raw open pose data')
parser.add_argument('--dst_dir', default='data/Stanford/', help='where you put the preprocessed file')

args = parser.parse_args()


def process_json_file(file_name, threshold=0.6):
    """
    read the json in
    filter people out
    and the class name
    :param file_name: input json file name
    :param threshold: filter the people with confidence
    :return: (key_point_json, class_name)
    """
    # first get the class name
    last_location = file_name.rfind('_') - 4
    class_name = file_name[:last_location]

    # then process the file
    f = open(file_name, 'r')
    data = json.load(f)
    f.close()
    people_data = data['people']

    if len(people_data) == 0:
        return [None, class_name]

    selected_index = 0    # which person to select if multiple available
    max_confident = 0     # compare for the most
    # select the person with maximum confident joints
    for i, person in enumerate(people_data):
        pose_data = person['pose_keypoints_2d']
        pose_data = np.array(pose_data).reshape((-1, 3))
        confidence = pose_data[:-1]
        confident_joints = np.sum(np.where(confidence >= threshold, 1, 0))

        if confident_joints > max_confident:
            selected_index = i
            max_confident = confident_joints

    # get the pose data
    person_data = np.array(people_data[selected_index]['pose_keypoints_2d']).reshape((-1, 3))
    person_data = person_data[:, :-1].tolist()

    return [person_data, class_name]


def main(args):
    json_list = os.listdir(args.src_dir)
    print('In Total {} files'.format(len(json_list)))
    class_names = list()
    target_json = list()

    for i, file_name in enumerate(json_list):
        person_data, class_name = process_json_file(args.src_dir + file_name)
        if person_data is None:
            continue
        if class_name not in class_names:
            class_names.append(class_name)

        target_json.append({'key_point': person_data, 'label': class_names.index(class_name)})

        if i % 100 == 0:
            print('{}'.format(i))

    print(class_names)
    f = open(os.path.join(args.dst_dir, 'data.json'), 'w')
    json.dump(target_json, f)
    f.close()


if __name__ == '__main__':
    main(args)