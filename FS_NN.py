import numpy as np
import concurrent.futures
import copy


def nearest_neighbor_classifier(data, point, feature_subset, num_instances):
    """使用 NumPy 快速计算给定点的最近邻居。"""
    point_data = data[point, feature_subset]
    distances = np.sqrt(((data[:, feature_subset] - point_data) ** 2).sum(axis=1))
    distances[point] = np.inf  # 在距离计算中排除自身
    nearest_neighbor = np.argmin(distances)
    return nearest_neighbor


def one_out_validator(data, feature_subset, num_instances):
    """使用并行处理加速执行留一交叉验证。"""
    correct = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(nearest_neighbor_classifier, data, i, feature_subset, num_instances) for i in
                   range(num_instances)]
        results = [future.result() for future in futures]

    correct = sum(1 for i in range(num_instances) if data[results[i], 0] == data[i, 0])
    accuracy = (correct / num_instances) * 100
    return accuracy


def forward_selection(data, num_instances, num_features):
    """使用 NumPy 和并行处理优化特征子集的前向选择。"""
    feature_subset = []
    final_set = []
    top_accuracy = 0.0

    for i in range(num_features):
        add_this_feature = -1
        for j in range(1, num_features + 1):
            if j not in feature_subset:
                temp_subset = copy.deepcopy(feature_subset)
                temp_subset.append(j)
                accuracy = one_out_validator(data, temp_subset, num_instances)
                print(f'\tUsing feature(s) {temp_subset}, accuracy is {accuracy:.2f}%')
                if accuracy > top_accuracy:
                    top_accuracy = accuracy
                    add_this_feature = j

        if add_this_feature >= 0:
            feature_subset.append(add_this_feature)
            final_set.append(add_this_feature)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {top_accuracy:.2f}%\n\n')

    print(f'Finished search!! The best feature subset is {final_set}, which has an accuracy of {top_accuracy:.2f}%')


def backward_elimination(data, num_instances, num_features):
    """使用 NumPy 和并行处理优化特征子集的后向消除。"""
    feature_subset = list(range(1, num_features + 1))
    final_set = copy.deepcopy(feature_subset)
    top_accuracy = one_out_validator(data, feature_subset, num_instances)
    print(f'Initial set {feature_subset} with accuracy: {top_accuracy:.2f}%')

    for _ in range(num_features):
        remove_this_feature = -1
        local_accuracy = 0.0
        for j in feature_subset:
            temp_subset = copy.deepcopy(feature_subset)
            temp_subset.remove(j)
            accuracy = one_out_validator(data, temp_subset, num_instances)
            print(f'\tUsing feature(s) {temp_subset}, accuracy is {accuracy:.2f}%')
            if accuracy > top_accuracy:
                top_accuracy = accuracy
                remove_this_feature = j

        if remove_this_feature >= 0:
            feature_subset.remove(remove_this_feature)
            final_set.remove(remove_this_feature)
            print(f'\n\nFeature set {feature_subset} was best, accuracy is {top_accuracy:.2f}%\n\n')
        else:
            break

    print(f'Finished search!! The best feature subset is {final_set}, which has an accuracy of {top_accuracy:.2f}%')


def normalize(data, num_features, num_instances):
    """使用 NumPy 高效计算数据集的标准化处理。"""
    data = np.array(data)
    mean = data[:, 1:num_features + 1].mean(axis=0)
    std = data[:, 1:num_features + 1].std(axis=0)
    data[:, 1:num_features + 1] = (data[:, 1:num_features + 1] - mean) / std
    return data


def main():
    print("Welcome to the Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test: ")

    try:
        data = np.loadtxt(file_name)
    except IOError:
        print(f"The file {file_name} does not exist. Exiting program.")
        return

    num_features = data.shape[1] - 1
    num_instances = data.shape[0]

    print("Type the number of the algorithm you want to run:")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    choice = int(input())

    while choice not in [1, 2]:
        print("Invalid choice, please try again.")
        choice = int(input())

    print(
        f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")

    print("Please wait while I normalize the data... Done!")
    normalized_instances = normalize(data, num_features, num_instances)

    all_features = list(range(1, num_features + 1))
    accuracy = one_out_validator(normalized_instances, all_features, num_instances)
    print(
        f"Running nearest neighbor with all {num_features} features, using 'leaving-one-out' evaluation, I get an accuracy of {accuracy:.2f}%.")

    if choice == 1:
        forward_selection(normalized_instances, num_instances, num_features)
    elif choice == 2:
        backward_elimination(normalized_instances, num_instances, num_features)


if __name__ == '__main__':
    main()
