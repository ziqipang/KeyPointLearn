import argparse
import json
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='../KTH_Raw/', help='where you store the raw open pose data')
parser.add_argument('--dst_dir', default='data/KTH/', help='where you put the preprocessed file')
parser.add_argument('--threshold', default=0.6)
parser.add_argument('--critical_joints', default='data/KTH/critical_joints.json')
args = parser.parse_args()

fp = open(args.critical_joints, 'r')
args.critical_joints = json.load(fp)
fp.close()


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
    first_location = file_name.find('/')
    class_name = file_name[first_location + 1:]
    first_location = class_name.find('/')
    class_name = class_name[first_location + 1:]
    first_location = class_name.find('_')
    class_name = class_name[first_location + 1:]
    first_location = class_name.find('_')
    class_name = class_name[:first_location]

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


def filter_func(args, critical_joints):
    """
    if the person does not have a critical joint, then we abandon it
    :param args
    :param critical_joints
    :return: write new json file to data_final.json
    """
    f = open(args.dst_dir + 'data.json', 'r')
    people_data = json.load(f)
    f.close()

    def judge(x, y):
        return x == 0 and y == 0

    new_people_data = []

    for person in people_data:
        pose = person['key_point']
        flag = True
        for joint_index in critical_joints:
            if judge(pose[joint_index][0], pose[joint_index][1]):
                flag = False
                break
        if flag:
            new_people_data.append(person)
    print(len(new_people_data))
    f = open(args.dst_dir + 'data_final.json', 'w')
    json.dump(new_people_data, f)
    f.close()


def main(args):
    json_list = os.listdir(args.src_dir)
    print('In Total {} files'.format(len(json_list)))
    class_names = list()
    target_json = list()

    for i, file_name in enumerate(json_list):
        person_data, class_name = process_json_file(args.src_dir + file_name, args.threshold)
        if person_data is None:
            continue
        if class_name not in class_names:
            class_names.append(class_name)

        label = class_names.index(class_name)
        if label >= 3:
            label = 3
        target_json.append({'key_point': person_data, 'label': label})

        if i % 100 == 0:
            print('{}'.format(i))

    print(class_names)
    f = open(os.path.join(args.dst_dir, 'class_names.json'), 'w')
    json.dump(class_names, f)
    f.close()
    f = open(os.path.join(args.dst_dir, 'data.json'), 'w')
    json.dump(target_json, f)
    f.close()


if __name__ == '__main__':
    main(args)
    filter_func(args, args.critical_joints)
