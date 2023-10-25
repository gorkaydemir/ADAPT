import logging
import time
import datetime
import sys
import argparse
import math
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import random
import zlib
import pickle


import multiprocessing
from multiprocessing import Process

parser = argparse.ArgumentParser("Argoverse Motion Forecasting Preprocessig")

# === Data Related Parameters ===
parser.add_argument('--data_dir', type=str,
                    help="Path of original ex-list file")
parser.add_argument('--output_dir', type=str, help="Path of output ex-file")
parser.add_argument('--core_num', type=int, default=8)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


class Arguments:
    def __init__(self):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.core_num = args.core_num


def swap_columns(array_list, id1, id2=0):
    for array in array_list:
        array[:, [id1, id2]] = array[:, [id2, id1]]


def get_angle(pos_matrix_i):
    span = pos_matrix_i[15:21]
    intervals = [2]
    angles = []
    for interval in intervals:
        for j in range(len(span)):
            if j + interval < len(span):
                der_x, der_y = span[j + interval, 0] - \
                    span[j, 0], span[j + interval, 1] - span[j, 1]
                angles.append([der_x, der_y])

    angles = np.array(angles)
    der_x, der_y = np.mean(angles, axis=0)
    angle = -math.atan2(der_y, der_x) + math.radians(90)
    return angle


def normalize_vectors(matrix, span, ref_coord, start, end, rot_matrix, agent=True):
    vectors = matrix[span]
    if agent:
        vectors[:, start:end] -= ref_coord
        vectors[:, start:end] = np.matmul(vectors[:, start:end], rot_matrix)
    else:
        ref_coord_flip = ref_coord[[-1, 0]]
        if end is None:
            vectors[:, start:] -= ref_coord_flip
            vectors[:, start:] = np.matmul(
                vectors[:, start:], np.transpose(rot_matrix))
        else:
            vectors[:, start:end] -= ref_coord_flip
            vectors[:, start:end] = np.matmul(
                vectors[:, start:end], np.transpose(rot_matrix))


def normalize_scene(mapping, id, angle):
    matrix = mapping["matrix"]
    polyline_spans = mapping["polyline_spans"]
    map_start_polyline_idx = mapping["map_start_polyline_idx"]
    labels = mapping["labels"]
    pos_matrix = mapping["pos_matrix"]
    information_matrix = mapping["information_matrix"]
    new_origin_labels = mapping["origin_labels"]

    vector_available = np.where(
        (information_matrix[:20] == 1).sum(axis=0) > 1)[0]

    labels = labels[:, vector_available]
    pos_matrix = pos_matrix[:, vector_available]
    information_matrix = information_matrix[:, vector_available]

    assert map_start_polyline_idx == pos_matrix.shape[1]
    assert map_start_polyline_idx == information_matrix.shape[1]
    assert map_start_polyline_idx == labels.shape[1]

    new_matrix = matrix.copy()
    new_pos_matrix = pos_matrix.copy()
    new_information_matrix = information_matrix.copy()
    new_labels = labels.copy()

    # === swap pose_matrix, information_matrix and label columns ===
    swap_columns([new_pos_matrix,
                 new_information_matrix, new_labels], id)
    # === === ===

    # === swap matrix columns ===
    span_0 = polyline_spans[0]
    span_id = polyline_spans[id]
    poly_0 = new_matrix[span_0].copy()
    poly_id = new_matrix[span_id].copy()

    new_matrix[span_0] = poly_id
    new_matrix[span_id] = poly_0
    # === === ===

    # === swap agent roles ===
    new_matrix[span_0][:, 6] = 1
    new_matrix[span_0][:, 7] = 0
    new_matrix[span_0][:, 8] = 0

    new_matrix[span_id][:, 6] = 0
    new_matrix[span_id][:, 7] = 1
    new_matrix[span_id][:, 8] = id
    # === === ===

    # === get new reference ===
    ref_point = new_pos_matrix[19, 0].copy()
    rot_matrix = np.array([[math.cos(angle), math.sin(angle)],
                           [-math.sin(angle), math.cos(angle)]])
    # === === ===

    # === normalize new_pos_matrix ===
    new_pos_matrix -= ref_point
    new_pos_matrix = np.matmul(new_pos_matrix, rot_matrix)

    assert abs(new_pos_matrix[19, 0, 0]) < 1e-5
    assert abs(new_pos_matrix[19, 0, 1]) < 1e-5
    # === === ===

    # === normalize new_labels ===
    new_labels -= ref_point
    new_labels = np.matmul(new_labels, rot_matrix)

    assert (new_labels[:, 0] == new_pos_matrix[20:, 0]).all()
    # === === ===

    # === normalize new_origin_labels ===
    new_origin_labels -= ref_point
    new_origin_labels = np.matmul(new_origin_labels, rot_matrix)
    # === === ===

    for j, polyline_span in enumerate(polyline_spans):
        is_agent_polyline = j < map_start_polyline_idx

        if is_agent_polyline:
            # prepoints
            normalize_vectors(new_matrix, polyline_span,
                              ref_point, 0, 2, rot_matrix, agent=True)
            # points
            normalize_vectors(new_matrix, polyline_span,
                              ref_point, 2, 4, rot_matrix, agent=True)

        else:
            # prepoints
            normalize_vectors(new_matrix, polyline_span,
                              ref_point, -2, None, rot_matrix, agent=False)
            # points
            normalize_vectors(new_matrix, polyline_span,
                              ref_point, -4, -2, rot_matrix, agent=False)
            # pre-pre points
            normalize_vectors(new_matrix, polyline_span,
                              ref_point, -18, -16, rot_matrix, agent=False)

    assert abs(new_matrix[18, 2]) < 1e-5
    assert abs(new_matrix[18, 3]) < 1e-5

    new_mapping = {"matrix": new_matrix,
                   "polyline_spans": polyline_spans,
                   "map_start_polyline_idx": map_start_polyline_idx,
                   "labels": new_labels,
                   "pos_matrix": new_pos_matrix,
                   "information_matrix": new_information_matrix,
                   "city_name": mapping["city_name"],
                   "file_name": mapping["file_name"],
                   "origin_labels": new_origin_labels,
                   "cent_x": mapping["cent_x"] + ref_point[0],
                   "cent_y": mapping["cent_y"] + ref_point[1],
                   "angle": mapping["angle"] + angle}

    return new_mapping


def get_new_agents(mapping):
    # pos_matrix.shape = (50, #agents, 2)
    # information_matrix.shape = (50, #agents)

    pos_matrix = mapping["pos_matrix"]
    information_matrix = mapping["information_matrix"]

    vector_available = np.where(
        (information_matrix[:20] == 1).sum(axis=0) > 1)[0]
    information_matrix = information_matrix[:, vector_available]
    pos_matrix = pos_matrix[:, vector_available]

    displacement = np.linalg.norm(
        pos_matrix[19] - pos_matrix[0], axis=-1) > 6.0
    full_traj = np.mean(information_matrix, axis=0) == 1.0
    candidates = np.where(np.logical_and(displacement, full_traj))[0]

    candidates_list = []

    for i in candidates:
        angle = get_angle(pos_matrix[:, i])
        candidates_list.append((i, angle))

    return candidates_list


def get_more_agents_scene(mapping):
    mapping = pickle.loads(zlib.decompress(mapping))
    new_mapping_list = [mapping]
    candidates_list = get_new_agents(mapping)

    ids = [candidate[0] for candidate in candidates_list]

    if 0 in ids:
        candidates_list = candidates_list[1:]

    for id, angle in candidates_list:
        new_mapping = normalize_scene(mapping, id, angle)
        new_mapping_list.append(new_mapping)

    random.shuffle(new_mapping_list)
    return new_mapping_list


def extend_data(args):
    pickle_file = open(args.data_dir, 'rb')
    ex_list = pickle.load(pickle_file)
    pickle_file.close()

    print(f"{len(ex_list)} scenes in ex-list")

    scene_num = len(ex_list)

    pbar = tqdm(total=scene_num)
    queue = multiprocessing.Queue(args.core_num)
    queue_res = multiprocessing.Queue()
    counter = multiprocessing.Queue()

    def calc_ex_list(queue, queue_res, counter):
        # res = []
        while True:
            mapping = queue.get()
            if mapping is None:
                break
            else:
                new_mappings = get_more_agents_scene(mapping)
                counter.put(len(new_mappings))
                for new_mapping in new_mappings:
                    data_compress = zlib.compress(
                        pickle.dumps(new_mapping))
                    queue_res.put(data_compress)

    processes = [Process(target=calc_ex_list, args=(
        queue, queue_res, counter)) for _ in range(args.core_num)]

    for each in processes:
        each.start()

    for mapping in ex_list:
        assert mapping is not None
        queue.put(mapping)
        pbar.update(1)
    pbar.close()

    while not queue.empty():
        pass

    cnt = 0
    for i in range(scene_num):
        cnt += counter.get()

    ex_list = []
    pbar = tqdm(total=cnt)
    for _ in range(cnt):
        t = queue_res.get()
        ex_list.append(t)
        pbar.update(1)
    pbar.close()

    for i in range(args.core_num):
        queue.put(None)
    for each in processes:
        each.join()

    pickle_file = open(os.path.join(
        args.output_dir, "extended_ex_list"), 'wb')
    pickle.dump(ex_list, pickle_file)
    pickle_file.close()
    assert len(ex_list) > 0
    print("Valid data size is", len(ex_list))


if __name__ == '__main__':
    args = Arguments()
    extend_data(args)
