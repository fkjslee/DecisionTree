"""
    write Decision Tree by hand
    only focused on basic principle
    (no continuous or miss value)
    (no pre-prune or post-prune)
    ...
"""

import numpy as np
import pandas as pd
import queue


class DecisionTree:
    def __init__(self, data):
        self.root = {}
        self.buildTree(data, self.root)

    @staticmethod
    def getEntropy(data: pd.DataFrame):
        entropy = 0.
        for k in data[data.columns[-1]].unique():
            pk = np.sum(data[data.columns[-1]] == k) * 1.0 / len(data)
            entropy += -pk * np.log2(pk)
        return entropy

    @staticmethod
    def getGain(data: pd.DataFrame, feature):
        assert feature in data.columns
        sum_entropy = 0.
        for k in data[feature].unique():
            data_son = data[data[feature] == k]
            sum_entropy += 1.0 * len(data_son) / len(data) * DecisionTree.getEntropy(data_son)
        return DecisionTree.getEntropy(data) - sum_entropy

    def buildTree(self, data: pd.DataFrame, node: map):
        max_gain = 0.
        columns = data.columns
        if len(data[data.columns[-1]].unique()) == 1:
            node['name'] = 'None'
            node['label'] = data[data.columns[-1]].unique()[0]
            return
        for feature in columns[0: len(columns) - 1]:
            gain = DecisionTree.getGain(data, feature)
            if gain > max_gain:
                max_gain = gain
                selected_feature = feature

        node['name'] = selected_feature
        for key in data[selected_feature].unique():
            node[key] = {}
            self.buildTree(data[data[selected_feature] == key], node[key])

    def classify(self, features: pd.Series):
        return self.classify_core(features, self.root)

    def classify_core(self, features: pd.Series, node: map):
        if 'label' in node.keys():
            return node['label']
        assert node['name'] in features
        key = features[node['name']]
        return self.classify_core(features, node[key])


def print_tree(root):
    que = queue.Queue()
    que.put(root)
    while not que.empty():
        node = que.get()
        for key, val in node.items():
            if key != 'name':
                if key != 'label':
                    print(node['name'], 'has son :', key)
                else:
                    print(node['name'], 'has son :', node['label'])
        for key, val in node.items():
            if key != 'name' and key != 'label':
                que.put(val)


if __name__ == "__main__":
    content = pd.read_csv("data/train.csv")
    test_watermelon = content.iloc[2]
    content.drop(content.columns[0], axis=1, inplace=True)
    tree = DecisionTree(content)
    # for i in range(len(content)):
    #     print(content.iloc[i]['好瓜'], tree.classify(content.iloc[i]))
    print_tree(tree.root)
