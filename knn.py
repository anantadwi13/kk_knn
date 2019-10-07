import csv
from pprint import pprint
import random
import math


def load_data_set(filename, k_fold=5):
    groups = []
    with open(filename, 'rt') as csvfile:
        dataset = csv.reader((line.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                              for line in csvfile))
        dataset_list = list(dataset)
        # print(dataset_list)
        random.shuffle(dataset_list)
        # print(dataset_list)
        i = 0
        max_item = math.ceil(len(dataset_list)/k_fold)
        for line in dataset_list:
            if i//max_item > len(groups)-1:
                groups.insert(i//max_item, [])
            instance = []
            for attrib in line:
                try:
                    instance.append(float(attrib))
                except:
                    instance.append(attrib)
            groups[i//max_item].append(instance)
            i += 1
        return groups


def euclidean_distance(data1, data2):
    distance = 0
    length = len(data1) if len(data1) < len(data2) else len(data2)
    for x in range(length):
        distance += pow(data1[x] - data2[x], 2)
    return math.sqrt(distance)


def manhattan_distance(data1, data2):
    distance = 0
    length = len(data1) if len(data1) < len(data2) else len(data2)
    for x in range(length):
        distance += abs(data1[x] - data2[x])
    return distance


def cosine_similarity_distance(data1, data2):
    length = len(data1) if len(data1) < len(data2) else len(data2)
    atas = 0
    bawah_kiri = 0
    bawah_kanan = 0
    for x in range(length):
        atas += data1[x] * data2[x]
        bawah_kiri += pow(data1[x], 2)
        bawah_kanan += pow(data2[x], 2)
    similarity = atas / (math.sqrt(bawah_kiri) * math.sqrt(bawah_kanan))
    return similarity


def get_neighbors(data_trains, data_test, k, distance_algo=1):
    d_test = data_test.copy()
    d_test.pop(0)
    d_test.pop(-1)
    #print([type(y) for y in d_test])

    distances = []
    neighbors = []
    for x in range(len(data_trains)):
        d_train = data_trains[x].copy()
        d_train.pop(0)
        d_train.pop(-1)
        #print([type(y) for y in d_train])

        if distance_algo == 2:
            distance = manhattan_distance(d_train, d_test)
        elif distance_algo == 3:
            distance = cosine_similarity_distance(d_train, d_test)
        else:
            distance = euclidean_distance(d_train, d_test)
        distances.append((data_trains[x], distance))

    if distance_algo == 3:
        distances.sort(key=lambda tup: tup[1], reverse=True)
    else:
        distances.sort(key=lambda tup: tup[1], reverse=False)

    for x in range(k):
        neighbors.append(distances[x])

    return neighbors


def get_majority_vote(data_neighbors):
    votes = {}
    for item in data_neighbors:
        class_vote = item[0][-1]
        if class_vote in votes:
            votes[class_vote] += 1
        else:
            votes[class_vote] = 1

    sorted_votes = sorted(votes.items(), key=lambda tup: tup[1], reverse=True)
    return sorted_votes[0][0]


if __name__ == '__main__':
    k_start = 1
    k_end = 25
    k_fold = 10
    #distance_algo = 2
    # 1 euclidean, 2 manhattan, 3 cosine similarity, default euclidean

    dataset = load_data_set('yeast.data', k_fold)

    for distance_algo in range(1, 4):
        for k in range(k_start, k_end+1):
            total_accuracy = 0
            for id_group in range(k_fold):
                temp_data_trains = dataset.copy()
                data_tests = temp_data_trains.pop(id_group)
                data_trains = []
                for group in temp_data_trains:
                    for item in group:
                        data_trains.append(item)

                if k == k_start and id_group == 0:
                    print('Total data trains : ' + str(len(data_trains)))
                    print('Total data tests : ' + str(len(data_tests)))
                    print('Rasio data tests & data trains : 1:' + str(math.ceil(len(data_trains)/len(data_tests))))

                correct = 0
                for data_test in data_tests:
                    neighbors = get_neighbors(data_trains, data_test, k, distance_algo)
                    class_test = data_test[-1]
                    class_predict = get_majority_vote(neighbors)
                    # print(neighbors)
                    # print('Test : ' + class_test)
                    # print('Predict : ' + class_predict)
                    correct += 1 if class_test == class_predict else 0
                accuracy = correct/len(data_tests)
                total_accuracy += accuracy
                # print('Akurasi ke-{0} : {1:.5f} %'.format(id_group+1, accuracy*100))
            # print('k = {0}\tRata-rata akurasi : {1:.5f} %'.format(k, total_accuracy/k_fold*100))
            print('{1:.5f}'.format(k, total_accuracy/k_fold*100))
