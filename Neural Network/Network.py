"""
A rudimentary neural network that supports JSON encoding and decoding features.
"""


import collections

import random

import json

import math

import numpy as np

from enum import Enum

from collections import deque

from abc import ABC, abstractmethod


class DataMismatchError(Exception):
    pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {self.Side.UPSTREAM: 0,
                                 self.Side.DOWNSTREAM: 0}
        self._reference_value = {self.Side.UPSTREAM: 0,
                                 self.Side.DOWNSTREAM: 0}
        self._neighbors = {self.Side.UPSTREAM: [],
                           self.Side.DOWNSTREAM: []}

    def __str__(self):
        stringify = ""
        stringify += "{:<24} {:<24} {:<24}".format("Current -",
                                                   "Downstream -",
                                                   "Upstream -")
        stringify += "\n{:<25}".format(str(id(self)))
        if self._neighbors[self.Side.UPSTREAM] or \
                self._neighbors[self.Side.DOWNSTREAM]:
            for i in range(0,
                           len(self._neighbors[self.Side.DOWNSTREAM]) if
                           len(self._neighbors[self.Side.DOWNSTREAM]) >
                           len(self._neighbors[self.Side.UPSTREAM]) else
                           len(self._neighbors[self.Side.UPSTREAM])
                           ):
                if i < len(self._neighbors[self.Side.DOWNSTREAM]):
                    stringify += "{:<25}".format(str(id(
                        self._neighbors[self.Side.DOWNSTREAM][i])))
                else:
                    stringify += "{:<25}".format("")
                if i < len(self._neighbors[self.Side.UPSTREAM]):
                    stringify += "{:<25}".format(str(id(
                        self._neighbors[self.Side.UPSTREAM][i])))
                stringify += "\n{:<25}".format("")
        return stringify

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        pass

    def reset_neighbors(self, nodes, side):
        """ sets _neighbors, processes nodes, and sets _reference_value """
        self._neighbors[side] = nodes.copy()
        for neighbor in self._neighbors[side]:
            self._process_new_neighbor(neighbor, side)
        self._reference_value[side] = pow(2, len(nodes)) - 1


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def _process_new_neighbor(self, node, side):
        """ sets weights for node if upstream """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.uniform(0, 1)

    def _check_in(self, node, side):
        """ checks if a node in _neighbors via binary encoding """
        power = 1 << self._neighbors[side].index(node)
        self._reporting_nodes[side] = (self._reporting_nodes[side] | power)
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        return False

    def get_weight(self, node):
        return self._weights[node]


class FFNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """ Calculates the weighted sum of upstream nodes """
        self._value = self._sigmoid(sum(node.value * self.get_weight(node) for
                                        node in
                                        self._neighbors[self.Side.UPSTREAM]))

    def _fire_downstream(self):
        """ Readies the data of the downstream neurodes """
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Upstream neurodes call this method when they have data ready """
        if self._check_in(node, self.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Sets the input value and readies neurodes """
        self._value = input_value
        self._fire_downstream()


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1 - value)

    def _calculate_delta(self, expected_value=None):
        """ Calculate delta from expected value """
        if self._node_type == LayerType.OUTPUT:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            self._delta = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += neurode.get_weight(self) * neurode.delta
            self._delta *= self._sigmoid_derivative(self.value)

    def _data_ready_downstream(self, node):
        """ Downstream neurodes call this method when they are ready """
        if self._check_in(node, self.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        """ Adjusts the current neurode's weight """
        self._weights[node] += adjustment

    def _update_weights(self):
        """ Calculates updates for weights downstream to self """
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            adjustment = self.value * node.delta * node.learning_rate
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """ Tells downstream nodes that this neurode is ready """
        for node in self._neighbors[self.Side.UPSTREAM]:
            node._data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.previous = None


class DoublyLinkedList:
    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = None

    class EmptyListError(Exception):
        pass

    def move_forward(self):
        """ Moves current forward """
        if self._curr is None:
            return self.EmptyListError
        elif self._curr == self._tail:
            raise IndexError
        else:
            self._curr = self._curr.next
        return self._curr.data

    def move_back(self):
        """ Moves current backwards """
        if self._curr is None:
            raise self.EmptyListError
        elif self._curr == self._head:
            raise IndexError
        else:
            self._curr = self._curr.previous
        return self._curr.data

    def reset_to_head(self):
        """ Resets current to head """
        self._curr = self._head
        if self._curr is None:
            raise self.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        """ Resets current to tail """
        self._curr = self._tail
        if self._curr is None:
            raise self.EmptyListError
        else:
            return self._curr.data

    def add_to_head(self, data):
        """ Adds a node as the new head """
        new_node = Node(data)
        new_node.next = self._head
        if self._tail is None:
            self._tail = new_node
        if self._head is not None:
            self._head.previous = new_node
        self._head = new_node
        self.reset_to_head()

    def add_after_curr(self, data):
        """ Adds a node after current node """
        if self._curr is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        new_node.next = self._curr.next
        new_node.previous = self._curr
        if self._curr == self._tail:
            self._tail = new_node
        else:
            self._curr.next.previous = new_node
        self._curr.next = new_node

    def remove_from_head(self):
        """ Removes the current head and replaces it """
        if self._head is None:
            raise self.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        self._head.previous = None
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        """ Removes the node after current """
        if self._curr is None:
            raise self.EmptyListError
        elif self._curr == self._tail:
            raise IndexError
        ret_val = self._curr.next.data
        if self._curr.next == self._tail:
            self._tail = self._curr
        else:
            self._curr.next.next.previous = self._curr
        self._curr.next = self._curr.next.next
        return ret_val

    def get_current_data(self):
        """ Gets current data """
        if self._curr is None:
            raise self.EmptyListError
        return self._curr.data

    def __iter__(self):
        self._curr_iter = self._head
        return self

    def __next__(self):
        if self._curr_iter is None:
            raise StopIteration
        ret_val = self._curr_iter.data
        self._curr_iter = self._curr_iter.next
        return ret_val


class LayerList(DoublyLinkedList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self._input_nodes = [FFBPNeurode(LayerType.INPUT) for
                             _ in range(inputs)]
        self._output_nodes = [FFBPNeurode(LayerType.OUTPUT) for
                              _ in range(outputs)]
        for node in self._input_nodes:
            node.reset_neighbors(self._output_nodes, node.Side.DOWNSTREAM)
        for node in self._output_nodes:
            node.reset_neighbors(self.input_nodes, node.Side.UPSTREAM)
        self.add_to_head(self.input_nodes)
        self.add_after_curr(self.output_nodes)

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    def add_layer(self, num_nodes: int):
        """ Adds a layer and readjusts neurode links """
        if self._curr == self._tail:
            raise IndexError
        self.add_after_curr([FFBPNeurode(LayerType.HIDDEN) for
                             _ in range(num_nodes)])
        self.link_layer_add()

    def remove_layer(self):
        """ Removes a layer and readjusts neurode links """
        if self._curr.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self.link_layer_remove()

    def link_layer_add(self):
        """ Helper function that fixes neighbors when adding a node """
        temp_curr = self._curr.next
        for node in temp_curr.data:
            node.reset_neighbors(temp_curr.previous.data,
                                 node.Side.UPSTREAM)
            node.reset_neighbors(temp_curr.next.data,
                                 node.Side.DOWNSTREAM)
        for node in temp_curr.previous.data:
            node.reset_neighbors(temp_curr.data,
                                 node.Side.DOWNSTREAM)
        for node in temp_curr.next.data:
            node.reset_neighbors(temp_curr.data,
                                 node.Side.UPSTREAM)

    def link_layer_remove(self):
        """ Helper function that fixes neighbors when removing a node """
        for node in self._curr.data:
            node.reset_neighbors(self._curr.next.data,
                                 node.Side.DOWNSTREAM)
        if self._curr.next is not None:
            for node in self._curr.next.data:
                node.reset_neighbors(self._curr.data,
                                     node.Side.UPSTREAM)


def load_XOR():
    """ Provides the default exclusive-or truth values as an object """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    return NNData(features, labels, 1)


class NNData:
    def __init__(self, features: [[]] = None, labels: [[]] = None,
                 train_factor: float = .9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self._features = None
        self._labels = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        try:
            self.load_data(features, labels)
        except DataMismatchError:
            pass
        except ValueError:
            pass
        self.split_set(train_factor)

    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """ Confines range of percentage to 0 <= percentage <= 1 """
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        return percentage

    def load_data(self, features=None, labels=None):
        """ Provides input validation, ensuring proper size and data type """
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError

    def split_set(self, new_train_factor=None):
        """ Randomly fills train and test indices """
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)
        loaded_examples = len(self._features)
        example_pool = int(loaded_examples * new_train_factor)
        self._train_indices = random.sample(range(loaded_examples),
                                            example_pool)
        self._test_indices = [_ for _ in range(0, loaded_examples)
                              if _ not in self._train_indices]
        self._train_indices.sort()

    def prime_data(self, target_set=None, order=None):
        """ Primes the deques based on indices """
        if target_set == NNData.Set.TRAIN:
            self._train_pool = deque(self._train_indices)
        elif target_set == NNData.Set.TEST:
            self._test_pool = deque(self._test_indices)
        else:
            self._train_pool = deque(self._train_indices)
            self._test_pool = deque(self._test_indices)
        if order == NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        """ Pops a single item from the deque and returns target indices """
        if self.pool_is_empty(target_set):
            return None
        elif target_set != NNData.Set.TEST:
            index = self._train_pool.popleft()
            return self._features[index], self._labels[index]
        index = self._test_pool.popleft()
        return self._features[index], self._labels[index]

    def number_of_samples(self, target_set=None):
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0


class FFBPNetwork:
    def __init__(self, num_inputs: int, num_outputs: int):
        self._input_nodes = num_inputs
        self._output_nodes = num_outputs
        self._layer_list = LayerList(num_inputs, num_outputs)

    class EmptySetException(Exception):
        pass

    def add_hidden_layer(self, num_nodes: int, position=0):
        self._layer_list.reset_to_head()
        if position != 0:
            for _ in range(position):
                self._layer_list.move_forward()
        self._layer_list.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise self.EmptySetException
        sum_error = 0
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self._layer_list.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self._layer_list.output_nodes):
                    node.set_expected(y[j])
                    sum_error += (node.value - y[j]) ** 2 / self._output_nodes
                    produced.append(node.value)
                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "produced", produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(
                    sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise self.EmptySetException
        sum_error = 0
        data_set.prime_data(order=order)
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for i, node in enumerate(self._layer_list.input_nodes):
                node.set_input(x[i])
            produced = []
            for i, node in enumerate(self._layer_list.output_nodes):
                sum_error += (node.value - y[i]) ** 2 / self._output_nodes
                produced.append(node.value)
            print("Sample", x, "expected", y, "produced", produced)
        print("Final RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TEST)))


class NNEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__NDarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        else:
            return json.JSONEncoder.default(self, o)


def nn_decoder(o):
    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    elif "__NNData__" in o:
        dec_obj = o["__NNData__"]
        ret_obj = NNData()

        labels = dec_obj["_labels"]
        features = dec_obj["_features"]
        train_indices = dec_obj["_train_indices"]
        test_indices = dec_obj["_test_indices"]
        train_factor = dec_obj["_train_factor"]
        train_pool = dec_obj["_train_pool"]
        test_pool = dec_obj["_test_pool"]

        ret_obj._labels = labels
        ret_obj._features = features
        ret_obj._train_indices = train_indices
        ret_obj._test_indices = test_indices
        ret_obj._train_factor = train_factor
        ret_obj._train_pool = train_pool
        ret_obj._test_pool = test_pool

        return ret_obj
    else:
        return o


def main():
    print("\n XOR --- \n")
    xor = load_XOR()
    xor_data_encoded = json.dumps(xor, cls=NNEncoder)
    xor_data_decoded = json.loads(xor_data_encoded, object_hook=nn_decoder)
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    network.train(xor_data_decoded, 10001, order=NNData.Order.RANDOM)
    print("\n SIN --- \n")
    with open("sin.txt", "r") as f:
        sin_decoded = json.load(f, object_hook=nn_decoder)
        network = FFBPNetwork(1, 1)
        network.add_hidden_layer(3)
        network.train(sin_decoded, 10001, order=NNData.Order.RANDOM)
        network.test(sin_decoded)


if __name__ == "__main__":
    main()


"""

 XOR --- 

Sample [1. 1.] expected [0.] produced [0.7231647718639505]
Sample [1. 0.] expected [1.] produced [0.6915190443149403]
Sample [0. 1.] expected [1.] produced [0.6926908560964862]
Sample [0. 0.] expected [0.] produced [0.6583263676967556]
Epoch 0 RMSE =  0.535247677159117
Epoch 100 RMSE =  0.5038129509251188
Epoch 200 RMSE =  0.5015281428957015
Epoch 300 RMSE =  0.5013849146901951
Epoch 400 RMSE =  0.5013399664943408
Epoch 500 RMSE =  0.5012972297713691
Epoch 600 RMSE =  0.5012649091841576
Epoch 700 RMSE =  0.5012274073642894
Epoch 800 RMSE =  0.5011961682470207
Epoch 900 RMSE =  0.5011645438191701
Sample [1. 1.] expected [0.] produced [0.5029404186625414]
Sample [0. 1.] expected [1.] produced [0.5012081504848643]
Sample [1. 0.] expected [1.] produced [0.5031462608151023]
Sample [0. 0.] expected [0.] produced [0.5058925646311658]
Epoch 1000 RMSE =  0.501132043217727
Epoch 1100 RMSE =  0.501101378462584
Epoch 1200 RMSE =  0.5010709694704587
Epoch 1300 RMSE =  0.5010418162388639
Epoch 1400 RMSE =  0.5010117643097268
Epoch 1500 RMSE =  0.5009822802632414
Epoch 1600 RMSE =  0.5009535884991616
Epoch 1700 RMSE =  0.5009234305602034
Epoch 1800 RMSE =  0.5008904176736846
Epoch 1900 RMSE =  0.5008634685004376
Sample [1. 1.] expected [0.] produced [0.5047491675450306]
Sample [0. 1.] expected [1.] produced [0.501308826515176]
Sample [0. 0.] expected [0.] produced [0.50146635771638]
Sample [1. 0.] expected [1.] produced [0.5016072291812557]
Epoch 2000 RMSE =  0.5008314263738823
Epoch 2100 RMSE =  0.5007993149234359
Epoch 2200 RMSE =  0.5007657844030177
Epoch 2300 RMSE =  0.5007311372297795
Epoch 2400 RMSE =  0.500691483802556
Epoch 2500 RMSE =  0.5006533473637297
Epoch 2600 RMSE =  0.5006119458204024
Epoch 2700 RMSE =  0.500568229194733
Epoch 2800 RMSE =  0.5005241755904147
Epoch 2900 RMSE =  0.5004736970176868
Sample [0. 1.] expected [1.] produced [0.5039787810968277]
Sample [1. 1.] expected [0.] produced [0.5093550011593739]
Sample [0. 0.] expected [0.] produced [0.49807277602865646]
Sample [1. 0.] expected [1.] produced [0.5018813942532283]
Epoch 3000 RMSE =  0.5004193747298995
Epoch 3100 RMSE =  0.5003561980797724
Epoch 3200 RMSE =  0.5002932296921674
Epoch 3300 RMSE =  0.5002171915889413
Epoch 3400 RMSE =  0.5001409873252377
Epoch 3500 RMSE =  0.5000531159325721
Epoch 3600 RMSE =  0.4999542270840177
Epoch 3700 RMSE =  0.4998394351699886
Epoch 3800 RMSE =  0.4997198884242039
Epoch 3900 RMSE =  0.49958024926346095
Sample [1. 1.] expected [0.] produced [0.5126634335561664]
Sample [0. 0.] expected [0.] produced [0.4918812767892323]
Sample [1. 0.] expected [1.] produced [0.5024649314173174]
Sample [0. 1.] expected [1.] produced [0.5046565758558479]
Epoch 4000 RMSE =  0.49941897212506636
Epoch 4100 RMSE =  0.49924305801947083
Epoch 4200 RMSE =  0.4990407256594883
Epoch 4300 RMSE =  0.4988095083595033
Epoch 4400 RMSE =  0.4985465250757462
Epoch 4500 RMSE =  0.4982412754211069
Epoch 4600 RMSE =  0.49790199662965917
Epoch 4700 RMSE =  0.49750469045864343
Epoch 4800 RMSE =  0.4970599209554863
Epoch 4900 RMSE =  0.49654220566873486
Sample [0. 0.] expected [0.] produced [0.4829524018422257]
Sample [0. 1.] expected [1.] produced [0.5119846469682976]
Sample [1. 0.] expected [1.] produced [0.513797641232583]
Sample [1. 1.] expected [0.] produced [0.5254650803375204]
Epoch 5000 RMSE =  0.49596075741054063
Epoch 5100 RMSE =  0.49529052629507253
Epoch 5200 RMSE =  0.49453740931118206
Epoch 5300 RMSE =  0.49368021376685844
Epoch 5400 RMSE =  0.4927149110249576
Epoch 5500 RMSE =  0.4916290931527545
Epoch 5600 RMSE =  0.49042299679096985
Epoch 5700 RMSE =  0.48909126290868243
Epoch 5800 RMSE =  0.48762009980266086
Epoch 5900 RMSE =  0.4860099136792944
Sample [1. 1.] expected [0.] produced [0.5338851683925017]
Sample [1. 0.] expected [1.] produced [0.5291520370746834]
Sample [0. 0.] expected [0.] produced [0.45824078560226517]
Sample [0. 1.] expected [1.] produced [0.5295392087776264]
Epoch 6000 RMSE =  0.4842646876676466
Epoch 6100 RMSE =  0.4823772640904589
Epoch 6200 RMSE =  0.4803687315336814
Epoch 6300 RMSE =  0.47822851743252714
Epoch 6400 RMSE =  0.4759680541785336
Epoch 6500 RMSE =  0.47359016118402214
Epoch 6600 RMSE =  0.47112308736734654
Epoch 6700 RMSE =  0.4685512205428562
Epoch 6800 RMSE =  0.4659065751062666
Epoch 6900 RMSE =  0.46319191105448
Sample [1. 1.] expected [0.] produced [0.5290635086315053]
Sample [0. 1.] expected [1.] produced [0.5564717392643493]
Sample [1. 0.] expected [1.] produced [0.5595187942652798]
Sample [0. 0.] expected [0.] produced [0.42102570698158137]
Epoch 7000 RMSE =  0.4604106462663313
Epoch 7100 RMSE =  0.457574592060705
Epoch 7200 RMSE =  0.4547019929512126
Epoch 7300 RMSE =  0.45179283445433227
Epoch 7400 RMSE =  0.4488598998337446
Epoch 7500 RMSE =  0.44590040569048117
Epoch 7600 RMSE =  0.4429414923464533
Epoch 7700 RMSE =  0.439967178097468
Epoch 7800 RMSE =  0.43699234866431186
Epoch 7900 RMSE =  0.4340232662188793
Sample [1. 1.] expected [0.] produced [0.5061993431393369]
Sample [1. 0.] expected [1.] produced [0.5856781665244152]
Sample [0. 0.] expected [0.] produced [0.3794927143390496]
Sample [0. 1.] expected [1.] produced [0.5860312419936727]
Epoch 8000 RMSE =  0.431069950684047
Epoch 8100 RMSE =  0.42811460194223894
Epoch 8200 RMSE =  0.42518508942702754
Epoch 8300 RMSE =  0.42226008405174464
Epoch 8400 RMSE =  0.4193665233719151
Epoch 8500 RMSE =  0.41648239317767716
Epoch 8600 RMSE =  0.41362097697840694
Epoch 8700 RMSE =  0.4107687166876482
Epoch 8800 RMSE =  0.4079492273733708
Epoch 8900 RMSE =  0.4051357785229971
Sample [1. 1.] expected [0.] produced [0.4784443151436633]
Sample [0. 0.] expected [0.] produced [0.3416274628553654]
Sample [1. 0.] expected [1.] produced [0.6101276451956847]
Sample [0. 1.] expected [1.] produced [0.612803711349401]
Epoch 9000 RMSE =  0.4023492590572719
Epoch 9100 RMSE =  0.39957371915674383
Epoch 9200 RMSE =  0.3968127179920418
Epoch 9300 RMSE =  0.39405714452752477
Epoch 9400 RMSE =  0.3912998049879263
Epoch 9500 RMSE =  0.38853329113949403
Epoch 9600 RMSE =  0.38576958973225306
Epoch 9700 RMSE =  0.38297126604684967
Epoch 9800 RMSE =  0.3801307898848981
Epoch 9900 RMSE =  0.37725541227412895
Sample [0. 0.] expected [0.] produced [0.3106812842008141]
Sample [1. 1.] expected [0.] produced [0.4472785438418864]
Sample [0. 1.] expected [1.] produced [0.6370828187151352]
Sample [1. 0.] expected [1.] produced [0.6365650409568865]
Epoch 10000 RMSE =  0.374290931649568
Final Epoch RMSE =  0.374290931649568

 SIN --- 

Sample [0.61] expected [0.57286746] produced [0.6878040331100207]
Sample [0.87] expected [0.76432894] produced [0.694824783020753]
Sample [0.82] expected [0.73114583] produced [0.6935987880055458]
Sample [1.1] expected [0.89120736] produced [0.7013092179951811]
Sample [0.24] expected [0.23770263] produced [0.6775306550742507]
Sample [1.2] expected [0.93203909] produced [0.7034901899814707]
Sample [0.48] expected [0.46177918] produced [0.6841303145050025]
Sample [0.54] expected [0.51413599] produced [0.685394114328993]
Sample [0.34] expected [0.33348709] produced [0.6792924070919215]
Sample [0.44] expected [0.42593947] produced [0.681486897870226]
Sample [0.66] expected [0.61311685] produced [0.687200778070756]
Sample [0.21] expected [0.2084599] produced [0.6741820621330618]
Sample [1.36] expected [0.9778646] produced [0.7048507883811328]
Sample [0.9] expected [0.78332691] produced [0.6933684683591919]
Sample [1.45] expected [0.99271299] produced [0.7080405774569962]
Sample [0.83] expected [0.73793137] produced [0.6923293962027796]
Sample [0.08] expected [0.07991469] produced [0.6709468406454786]
Sample [0.56] expected [0.5311862] produced [0.6836990283905875]
Sample [0.17] expected [0.16918235] produced [0.6721692172442154]
Sample [0.53] expected [0.50553334] produced [0.681529173894055]
Sample [0.41] expected [0.39860933] produced [0.6777570882391195]
Sample [0.39] expected [0.38018842] produced [0.6766205082006166]
Sample [1.48] expected [0.99588084] produced [0.7052623838264219]
Sample [0.8] expected [0.71735609] produced [0.688161127313562]
Sample [0.69] expected [0.63653718] produced [0.6851885198447468]
Sample [0.97] expected [0.82488571] produced [0.6927376577719235]
Sample [0.49] expected [0.47062589] produced [0.6797638251254923]
Sample [0.47] expected [0.45288629] produced [0.678764300859923]
Sample [0.01] expected [0.00999983] produced [0.6650755409720908]
Sample [0.15] expected [0.14943813] produced [0.6678985570623096]
Sample [1.46] expected [0.99386836] produced [0.7023056524666736]
Epoch 0 RMSE =  0.29828345205064183
Epoch 100 RMSE =  0.2656207163148338
Epoch 200 RMSE =  0.23183725893066287
Epoch 300 RMSE =  0.18713082720295302
Epoch 400 RMSE =  0.15226330540929106
Epoch 500 RMSE =  0.12762272332084496
Epoch 600 RMSE =  0.109812519630166
Epoch 700 RMSE =  0.09644451327605755
Epoch 800 RMSE =  0.08608546179642258
Epoch 900 RMSE =  0.07784964099774908
Sample [1.36] expected [0.9778646] produced [0.8481675885075809]
Sample [0.49] expected [0.47062589] produced [0.5022591680226259]
Sample [0.97] expected [0.82488571] produced [0.7681700199432867]
Sample [0.53] expected [0.50553334] produced [0.5336297545293116]
Sample [1.46] expected [0.99386836] produced [0.8593516685897438]
Sample [0.47] expected [0.45288629] produced [0.4863971967582806]
Sample [0.69] expected [0.63653718] produced [0.6430876362927792]
Sample [0.01] expected [0.00999983] produced [0.1448309929946721]
Sample [0.61] expected [0.57286746] produced [0.5916339067735698]
Sample [0.44] expected [0.42593947] produced [0.4613515772250734]
Sample [0.17] expected [0.16918235] produced [0.24214571252505146]
Sample [0.48] expected [0.46177918] produced [0.49373630450116784]
Sample [1.2] expected [0.93203909] produced [0.8238842605721298]
Sample [0.8] expected [0.71735609] produced [0.7013764223811796]
Sample [1.48] expected [0.99588084] produced [0.8611780303980928]
Sample [0.15] expected [0.14943813] produced [0.22824858949571641]
Sample [0.87] expected [0.76432894] produced [0.7322421578803389]
Sample [0.39] expected [0.38018842] produced [0.41961223432154693]
Sample [0.66] expected [0.61311685] produced [0.6244573735560761]
Sample [0.41] expected [0.39860933] produced [0.4362666001920366]
Sample [0.34] expected [0.33348709] produced [0.3770468673549129]
Sample [1.45] expected [0.99271299] produced [0.8581500163067812]
Sample [1.1] expected [0.89120736] produced [0.8036399617554075]
Sample [0.08] expected [0.07991469] produced [0.18306098654101355]
Sample [0.21] expected [0.2084599] produced [0.27178325238020684]
Sample [0.24] expected [0.23770263] produced [0.2949298403958134]
Sample [0.9] expected [0.78332691] produced [0.743688791088119]
Sample [0.82] expected [0.73114583] produced [0.7106959981347405]
Sample [0.83] expected [0.73793137] produced [0.7152516967247519]
Sample [0.54] expected [0.51413599] produced [0.541039188633402]
Sample [0.56] expected [0.5311862] produced [0.5558694787307916]
Epoch 1000 RMSE =  0.07118624215677938
Epoch 1100 RMSE =  0.0657041789629037
Epoch 1200 RMSE =  0.06114482060117639
Epoch 1300 RMSE =  0.057315040925002385
Epoch 1400 RMSE =  0.0540730916236828
Epoch 1500 RMSE =  0.05130782533186158
Epoch 1600 RMSE =  0.04894051123487567
Epoch 1700 RMSE =  0.0468994385766572
Epoch 1800 RMSE =  0.0451301727670504
Epoch 1900 RMSE =  0.04359161091494291
Sample [0.54] expected [0.51413599] produced [0.5304380361384631]
Sample [0.15] expected [0.14943813] produced [0.1744960819799828]
Sample [0.87] expected [0.76432894] produced [0.7626297077445827]
Sample [0.48] expected [0.46177918] produced [0.4727826274776908]
Sample [0.83] expected [0.73793137] produced [0.7427395631263557]
Sample [0.17] expected [0.16918235] produced [0.1884072136233196]
Sample [1.1] expected [0.89120736] produced [0.8429757797732167]
Sample [0.01] expected [0.00999983] produced [0.09769248877216972]
Sample [0.82] expected [0.73114583] produced [0.7374553674788074]
Sample [0.34] expected [0.33348709] produced [0.3340497532904578]
Sample [1.48] expected [0.99588084] produced [0.9023636176632844]
Sample [0.21] expected [0.2084599] produced [0.2186410866472106]
Sample [0.61] expected [0.57286746] produced [0.5927108408502996]
Sample [0.53] expected [0.50553334] produced [0.5209052120089711]
Sample [0.41] expected [0.39860933] produced [0.40309260651398066]
Sample [0.47] expected [0.45288629] produced [0.46280849147724845]
Sample [1.36] expected [0.9778646] produced [0.8895297112081159]
Sample [0.44] expected [0.42593947] produced [0.4331710408419707]
Sample [0.9] expected [0.78332691] produced [0.776197501076066]
Sample [0.8] expected [0.71735609] produced [0.7265157953740793]
Sample [0.66] expected [0.61311685] produced [0.6330746613964863]
Sample [0.08] expected [0.07991469] produced [0.13153525289927376]
Sample [0.39] expected [0.38018842] produced [0.38306278700907426]
Sample [1.2] expected [0.93203909] produced [0.8647998690655474]
Sample [0.97] expected [0.82488571] produced [0.80387283693643]
Sample [1.45] expected [0.99271299] produced [0.8995830404060495]
Sample [0.49] expected [0.47062589] produced [0.4827824582792992]
Sample [0.69] expected [0.63653718] produced [0.655777719348917]
Sample [1.46] expected [0.99386836] produced [0.9005701777933265]
Sample [0.24] expected [0.23770263] produced [0.24320769015141613]
Sample [0.56] expected [0.5311862] produced [0.5489612867730352]
Epoch 2000 RMSE =  0.04224752307543063
Epoch 2100 RMSE =  0.04106782581540853
Epoch 2200 RMSE =  0.04002888842373464
Epoch 2300 RMSE =  0.039109931836531345
Epoch 2400 RMSE =  0.03829386814928877
Epoch 2500 RMSE =  0.03756673560208412
Epoch 2600 RMSE =  0.03691620226846188
Epoch 2700 RMSE =  0.03633203329071797
Epoch 2800 RMSE =  0.03580562594303117
Epoch 2900 RMSE =  0.03532913251601306
Sample [0.82] expected [0.73114583] produced [0.746834096814397]
Sample [0.87] expected [0.76432894] produced [0.7735101193556926]
Sample [0.54] expected [0.51413599] produced [0.5253030348942187]
Sample [0.56] expected [0.5311862] produced [0.5448274309757537]
Sample [1.46] expected [0.99386836] produced [0.9165714383296671]
Sample [0.9] expected [0.78332691] produced [0.7878653240369092]
Sample [0.61] expected [0.57286746] produced [0.5917351608784125]
Sample [0.47] expected [0.45288629] produced [0.45391743259504536]
Sample [0.21] expected [0.2084599] produced [0.20387217750341666]
Sample [0.83] expected [0.73793137] produced [0.7522654950173667]
Sample [0.01] expected [0.00999983] produced [0.08735236155483242]
Sample [1.45] expected [0.99271299] produced [0.915562419519186]
Sample [0.48] expected [0.46177918] produced [0.4642966791863265]
Sample [0.24] expected [0.23770263] produced [0.22814930360439759]
Sample [0.08] expected [0.07991469] produced [0.1193861625083675]
Sample [0.44] expected [0.42593947] produced [0.42264376028504386]
Sample [0.66] expected [0.61311685] produced [0.634932377332669]
Sample [0.53] expected [0.50553334] produced [0.5150934589450583]
Sample [0.15] expected [0.14943813] produced [0.16057780225876933]
Sample [0.34] expected [0.33348709] produced [0.3199521972103053]
Sample [1.2] expected [0.93203909] produced [0.8803929161725014]
Sample [1.36] expected [0.9778646] produced [0.905600390691024]
Sample [0.49] expected [0.47062589] produced [0.4747440165348317]
Sample [0.97] expected [0.82488571] produced [0.817087998208084]
Sample [0.41] expected [0.39860933] produced [0.39153491455008266]
Sample [1.1] expected [0.89120736] produced [0.8580298345372502]
Sample [0.8] expected [0.71735609] produced [0.7351633848103853]
Sample [1.48] expected [0.99588084] produced [0.9185254980824189]
Sample [0.69] expected [0.63653718] produced [0.6593901654173605]
Sample [0.17] expected [0.16918235] produced [0.17432963312320215]
Sample [0.39] expected [0.38018842] produced [0.3708761794116575]
Epoch 3000 RMSE =  0.0348967551693889
Epoch 3100 RMSE =  0.034503284698895444
Epoch 3200 RMSE =  0.03414265982035907
Epoch 3300 RMSE =  0.03381347899344244
Epoch 3400 RMSE =  0.03350993131120021
Epoch 3500 RMSE =  0.033229821846106504
Epoch 3600 RMSE =  0.0329706710704445
Epoch 3700 RMSE =  0.03272980754677001
Epoch 3800 RMSE =  0.032506482943686986
Epoch 3900 RMSE =  0.032296774951032156
Sample [0.9] expected [0.78332691] produced [0.7930322126724075]
Sample [0.24] expected [0.23770263] produced [0.22317010342154392]
Sample [0.41] expected [0.39860933] produced [0.38642279426822207]
Sample [0.54] expected [0.51413599] produced [0.522403425675914]
Sample [0.66] expected [0.61311685] produced [0.6350800741456256]
Sample [1.48] expected [0.99588084] produced [0.9269208387707543]
Sample [0.8] expected [0.71735609] produced [0.7383484244071893]
Sample [0.39] expected [0.38018842] produced [0.3654703930312318]
Sample [0.69] expected [0.63653718] produced [0.6598976219804805]
Sample [0.56] expected [0.5311862] produced [0.5422069831760353]
Sample [0.97] expected [0.82488571] produced [0.8230421733199118]
Sample [0.87] expected [0.76432894] produced [0.777961347946386]
Sample [1.45] expected [0.99271299] produced [0.9239968094889647]
Sample [1.1] expected [0.89120736] produced [0.8652270687974756]
Sample [0.01] expected [0.00999983] produced [0.08482660746274202]
Sample [0.47] expected [0.45288629] produced [0.44957113348868655]
Sample [0.08] expected [0.07991469] produced [0.11608944554652123]
Sample [0.34] expected [0.33348709] produced [0.3146333178304978]
Sample [1.46] expected [0.99386836] produced [0.9250451478857772]
Sample [0.44] expected [0.42593947] produced [0.41805132090436664]
Sample [0.82] expected [0.73114583] produced [0.7505526419879578]
Sample [1.2] expected [0.93203909] produced [0.8883951748840773]
Sample [0.48] expected [0.46177918] produced [0.4603388179076033]
Sample [0.49] expected [0.47062589] produced [0.4708444055157125]
Sample [0.53] expected [0.50553334] produced [0.5122752794686889]
Sample [0.83] expected [0.73793137] produced [0.7563753530991388]
Sample [0.61] expected [0.57286746] produced [0.5904452476146967]
Sample [0.21] expected [0.2084599] produced [0.19911287370958183]
Sample [0.15] expected [0.14943813] produced [0.1565520388883216]
Sample [0.17] expected [0.16918235] produced [0.16991848495657183]
Sample [1.36] expected [0.9778646] produced [0.9139305792309972]
Epoch 4000 RMSE =  0.032102612280119305
Epoch 4100 RMSE =  0.03191982492277305
Epoch 4200 RMSE =  0.031748126114885644
Epoch 4300 RMSE =  0.03158650726505113
Epoch 4400 RMSE =  0.03143322306055615
Epoch 4500 RMSE =  0.03129037053847987
Epoch 4600 RMSE =  0.03115415904042469
Epoch 4700 RMSE =  0.031024956491315307
Epoch 4800 RMSE =  0.030902139282599772
Epoch 4900 RMSE =  0.03078610395730202
Sample [1.36] expected [0.9778646] produced [0.9190058496994055]
Sample [0.8] expected [0.71735609] produced [0.7398534561442108]
Sample [0.47] expected [0.45288629] produced [0.4472938933416324]
Sample [0.39] expected [0.38018842] produced [0.3630731228025224]
Sample [0.97] expected [0.82488571] produced [0.826570571256681]
Sample [0.44] expected [0.42593947] produced [0.41561827028540416]
Sample [1.45] expected [0.99271299] produced [0.9293005392216076]
Sample [0.9] expected [0.78332691] produced [0.7958554761271542]
Sample [1.1] expected [0.89120736] produced [0.8696340045704206]
Sample [0.34] expected [0.33348709] produced [0.3126383157335765]
Sample [0.82] expected [0.73114583] produced [0.7524842566374332]
Sample [0.66] expected [0.61311685] produced [0.6348023773932704]
Sample [1.46] expected [0.99386836] produced [0.930283844422622]
Sample [0.61] expected [0.57286746] produced [0.5895401167201898]
Sample [0.21] expected [0.2084599] produced [0.19777466074837244]
Sample [0.69] expected [0.63653718] produced [0.659981170470113]
Sample [0.49] expected [0.47062589] produced [0.4684553634486889]
Sample [0.54] expected [0.51413599] produced [0.5204282245777132]
Sample [0.17] expected [0.16918235] produced [0.16880691102963635]
Sample [1.48] expected [0.99588084] produced [0.9321502493181935]
Sample [0.08] expected [0.07991469] produced [0.11557122954932618]
Sample [0.01] expected [0.00999983] produced [0.08458468826096936]
Sample [0.53] expected [0.50553334] produced [0.5101375296622276]
Sample [0.41] expected [0.39860933] produced [0.3838749326252838]
Sample [0.56] expected [0.5311862] produced [0.540683606194621]
Sample [0.83] expected [0.73793137] produced [0.7581045208381969]
Sample [0.48] expected [0.46177918] produced [0.45776780005509915]
Sample [0.15] expected [0.14943813] produced [0.15553293474979252]
Sample [0.87] expected [0.76432894] produced [0.7803456116674728]
Sample [1.2] expected [0.93203909] produced [0.8929554821100347]
Sample [0.24] expected [0.23770263] produced [0.22145056311814063]
Epoch 5000 RMSE =  0.03067509569272187
Epoch 5100 RMSE =  0.030569410595768887
Epoch 5200 RMSE =  0.030468594571929675
Epoch 5300 RMSE =  0.030372093758905278
Epoch 5400 RMSE =  0.03027986610012435
Epoch 5500 RMSE =  0.03019165507141999
Epoch 5600 RMSE =  0.030106793504981404
Epoch 5700 RMSE =  0.030025753424829802
Epoch 5800 RMSE =  0.0299476290127261
Epoch 5900 RMSE =  0.02987249708017891
Sample [0.83] expected [0.73793137] produced [0.7590295959113811]
Sample [0.69] expected [0.63653718] produced [0.6593304027728051]
Sample [0.08] expected [0.07991469] produced [0.11571141274824431]
Sample [0.34] expected [0.33348709] produced [0.31105446027432365]
Sample [1.2] expected [0.93203909] produced [0.8960022453199091]
Sample [0.15] expected [0.14943813] produced [0.1554832803424586]
Sample [0.44] expected [0.42593947] produced [0.41391830421459386]
Sample [0.47] expected [0.45288629] produced [0.44574270459828963]
Sample [1.1] expected [0.89120736] produced [0.8722190385382057]
Sample [0.21] expected [0.2084599] produced [0.19736325410118963]
Sample [0.82] expected [0.73114583] produced [0.753062238701142]
Sample [0.17] expected [0.16918235] produced [0.16864203049909163]
Sample [1.36] expected [0.9778646] produced [0.9223963873768835]
Sample [0.97] expected [0.82488571] produced [0.8285710038189582]
Sample [0.87] expected [0.76432894] produced [0.7817558425304171]
Sample [0.01] expected [0.00999983] produced [0.08495877394155033]
Sample [0.53] expected [0.50553334] produced [0.508641690575687]
Sample [0.39] expected [0.38018842] produced [0.361600094941926]
Sample [0.48] expected [0.46177918] produced [0.4563660694769537]
Sample [0.9] expected [0.78332691] produced [0.7971737505527615]
Sample [1.48] expected [0.99588084] produced [0.9356628189169096]
Sample [0.56] expected [0.5311862] produced [0.5394122220307962]
Sample [0.8] expected [0.71735609] produced [0.7404220360658034]
Sample [1.46] expected [0.99386836] produced [0.9337183780088194]
Sample [0.49] expected [0.47062589] produced [0.46692134122830026]
Sample [0.41] expected [0.39860933] produced [0.3824796058511962]
Sample [1.45] expected [0.99271299] produced [0.932785144539744]
Sample [0.61] expected [0.57286746] produced [0.5885146974820595]
Sample [0.54] expected [0.51413599] produced [0.5191323849914157]
Sample [0.66] expected [0.61311685] produced [0.6339973402182812]
Sample [0.24] expected [0.23770263] produced [0.22091523991657472]
Epoch 6000 RMSE =  0.029800518530533178
Epoch 6100 RMSE =  0.029730296009329374
Epoch 6200 RMSE =  0.029663139851516928
Epoch 6300 RMSE =  0.029599141543908778
Epoch 6400 RMSE =  0.029536276220954626
Epoch 6500 RMSE =  0.029476806165757985
Epoch 6600 RMSE =  0.02941853959312283
Epoch 6700 RMSE =  0.029362055161967642
Epoch 6800 RMSE =  0.02930759586795489
Epoch 6900 RMSE =  0.029255003851238234
Sample [1.36] expected [0.9778646] produced [0.9249006348845956]
Sample [0.41] expected [0.39860933] produced [0.3816711492451208]
Sample [0.87] expected [0.76432894] produced [0.7827080320835941]
Sample [0.24] expected [0.23770263] produced [0.22088536396282488]
Sample [0.9] expected [0.78332691] produced [0.798297592696971]
Sample [1.1] expected [0.89120736] produced [0.8742211902507229]
Sample [0.69] expected [0.63653718] produced [0.6592564410704703]
Sample [0.61] expected [0.57286746] produced [0.5875070549095258]
Sample [0.49] expected [0.47062589] produced [0.4657964816729388]
Sample [0.34] expected [0.33348709] produced [0.31062344661586333]
Sample [0.21] expected [0.2084599] produced [0.1973876794780995]
Sample [0.53] expected [0.50553334] produced [0.5077838575660887]
Sample [0.01] expected [0.00999983] produced [0.0855128395172863]
Sample [0.83] expected [0.73793137] produced [0.7596012380515491]
Sample [0.15] expected [0.14943813] produced [0.15573943709764018]
Sample [1.45] expected [0.99271299] produced [0.9352825154505493]
Sample [0.8] expected [0.71735609] produced [0.7407485574591893]
Sample [1.48] expected [0.99588084] produced [0.9382185809967232]
Sample [0.08] expected [0.07991469] produced [0.11619946792204566]
Sample [1.2] expected [0.93203909] produced [0.8982770391891152]
Sample [0.17] expected [0.16918235] produced [0.16882843240564938]
Sample [0.48] expected [0.46177918] produced [0.45531835750362487]
Sample [0.47] expected [0.45288629] produced [0.444785734311353]
Sample [0.66] expected [0.61311685] produced [0.6334597302565079]
Sample [1.46] expected [0.99386836] produced [0.9363159309239203]
Sample [0.97] expected [0.82488571] produced [0.8299463768789473]
Sample [0.56] expected [0.5311862] produced [0.5384082100322893]
Sample [0.44] expected [0.42593947] produced [0.4130228330944059]
Sample [0.54] expected [0.51413599] produced [0.5180670029478847]
Sample [0.39] expected [0.38018842] produced [0.36087318841998633]
Sample [0.82] expected [0.73114583] produced [0.7536053919499478]
Epoch 7000 RMSE =  0.02920400808797414
Epoch 7100 RMSE =  0.029154358590443432
Epoch 7200 RMSE =  0.029106210293868513
Epoch 7300 RMSE =  0.029059130015156157
Epoch 7400 RMSE =  0.029014490798796925
Epoch 7500 RMSE =  0.028970259652753245
Epoch 7600 RMSE =  0.028927750051910145
Epoch 7700 RMSE =  0.02888481469588242
Epoch 7800 RMSE =  0.028845631639897402
Epoch 7900 RMSE =  0.028805688775352205
Sample [0.8] expected [0.71735609] produced [0.7409053092951066]
Sample [0.48] expected [0.46177918] produced [0.4543050888748116]
Sample [0.15] expected [0.14943813] produced [0.15604834819182778]
Sample [0.82] expected [0.73114583] produced [0.7536255203864006]
Sample [0.97] expected [0.82488571] produced [0.8307059666827847]
Sample [0.01] expected [0.00999983] produced [0.08598227623383893]
Sample [0.24] expected [0.23770263] produced [0.22063077140303242]
Sample [0.39] expected [0.38018842] produced [0.35995627615454623]
Sample [0.87] expected [0.76432894] produced [0.7828846429546188]
Sample [0.54] expected [0.51413599] produced [0.5168334554011565]
Sample [1.2] expected [0.93203909] produced [0.8998601068202918]
Sample [0.41] expected [0.39860933] produced [0.3806632339434485]
Sample [1.36] expected [0.9778646] produced [0.9267598293035162]
Sample [0.17] expected [0.16918235] produced [0.1690829500158286]
Sample [0.08] expected [0.07991469] produced [0.11666512632180134]
Sample [0.47] expected [0.45288629] produced [0.4438569025707871]
Sample [1.48] expected [0.99588084] produced [0.9402549181805181]
Sample [1.45] expected [0.99271299] produced [0.9373269754730673]
Sample [0.49] expected [0.47062589] produced [0.46521858950938666]
Sample [1.1] expected [0.89120736] produced [0.8757542572503435]
Sample [0.53] expected [0.50553334] produced [0.507137804669321]
Sample [0.66] expected [0.61311685] produced [0.6331356783265351]
Sample [0.56] expected [0.5311862] produced [0.5376964190038365]
Sample [0.69] expected [0.63653718] produced [0.6587953255644817]
Sample [0.61] expected [0.57286746] produced [0.5867260895603316]
Sample [0.34] expected [0.33348709] produced [0.3101115014411262]
Sample [1.46] expected [0.99386836] produced [0.9383014686735186]
Sample [0.21] expected [0.2084599] produced [0.19749569633508232]
Sample [0.44] expected [0.42593947] produced [0.4123481963434479]
Sample [0.83] expected [0.73793137] produced [0.7600883826497034]
Sample [0.9] expected [0.78332691] produced [0.7988825643927583]
Epoch 8000 RMSE =  0.02876795925996827
Epoch 8100 RMSE =  0.028730501863593518
Epoch 8200 RMSE =  0.02869410219594172
Epoch 8300 RMSE =  0.028658997198932085
Epoch 8400 RMSE =  0.028624612634064777
Epoch 8500 RMSE =  0.028590674818771315
Epoch 8600 RMSE =  0.02855801740046927
Epoch 8700 RMSE =  0.028525651974263353
Epoch 8800 RMSE =  0.028494778377890467
Epoch 8900 RMSE =  0.028463769417657995
Sample [0.66] expected [0.61311685] produced [0.6323993155212579]
Sample [0.8] expected [0.71735609] produced [0.7409340325632222]
Sample [0.53] expected [0.50553334] produced [0.5058731857742469]
Sample [0.17] expected [0.16918235] produced [0.16928158328396783]
Sample [0.48] expected [0.46177918] produced [0.4535517487628216]
Sample [1.46] expected [0.99386836] produced [0.9398502981532731]
Sample [0.41] expected [0.39860933] produced [0.3802409364032335]
Sample [0.87] expected [0.76432894] produced [0.7834564790180163]
Sample [1.45] expected [0.99271299] produced [0.9388607132089172]
Sample [1.2] expected [0.93203909] produced [0.9013660740682942]
Sample [1.36] expected [0.9778646] produced [0.9283752561578292]
Sample [0.44] expected [0.42593947] produced [0.41183972550053727]
Sample [0.56] expected [0.5311862] produced [0.5371644642940414]
Sample [1.48] expected [0.99588084] produced [0.941932633480794]
Sample [0.97] expected [0.82488571] produced [0.8319390745058189]
Sample [0.34] expected [0.33348709] produced [0.31014523123740884]
Sample [0.49] expected [0.47062589] produced [0.4647542608549782]
Sample [0.61] expected [0.57286746] produced [0.5866661883957816]
Sample [0.01] expected [0.00999983] produced [0.08656069133948702]
Sample [1.1] expected [0.89120736] produced [0.8769511692794097]
Sample [0.15] expected [0.14943813] produced [0.15651772414394685]
Sample [0.39] expected [0.38018842] produced [0.36001464072569056]
Sample [0.47] expected [0.45288629] produced [0.44368409859300756]
Sample [0.83] expected [0.73793137] produced [0.7606135607232188]
Sample [0.69] expected [0.63653718] produced [0.658716125268319]
Sample [0.54] expected [0.51413599] produced [0.5166792295267327]
Sample [0.08] expected [0.07991469] produced [0.11715410028769649]
Sample [0.24] expected [0.23770263] produced [0.22095237376639354]
Sample [0.9] expected [0.78332691] produced [0.7995332468562747]
Sample [0.82] expected [0.73114583] produced [0.7540526786864112]
Sample [0.21] expected [0.2084599] produced [0.1976278770404514]
Epoch 9000 RMSE =  0.028433720375773873
Epoch 9100 RMSE =  0.028405181497763773
Epoch 9200 RMSE =  0.02837600108473255
Epoch 9300 RMSE =  0.02834727671785864
Epoch 9400 RMSE =  0.0283215122688686
Epoch 9500 RMSE =  0.028294558808209375
Epoch 9600 RMSE =  0.028268486413849003
Epoch 9700 RMSE =  0.028242907252626924
Epoch 9800 RMSE =  0.028218002083129752
Epoch 9900 RMSE =  0.028193653030193312
Sample [0.21] expected [0.2084599] produced [0.1978493993594587]
Sample [0.69] expected [0.63653718] produced [0.6581233257031388]
Sample [0.82] expected [0.73114583] produced [0.754038651717714]
Sample [1.2] expected [0.93203909] produced [0.9024096367175702]
Sample [0.9] expected [0.78332691] produced [0.7997104239266868]
Sample [1.48] expected [0.99588084] produced [0.9431504378760515]
Sample [0.41] expected [0.39860933] produced [0.3798324007832194]
Sample [1.46] expected [0.99386836] produced [0.9412467887889989]
Sample [0.39] expected [0.38018842] produced [0.3594742426747563]
Sample [0.01] expected [0.00999983] produced [0.08699146514142592]
Sample [0.15] expected [0.14943813] produced [0.15677616756735052]
Sample [0.8] expected [0.71735609] produced [0.7412511516198051]
Sample [0.08] expected [0.07991469] produced [0.11752230677336449]
Sample [1.45] expected [0.99271299] produced [0.9402081042788043]
Sample [0.49] expected [0.47062589] produced [0.4638211014203123]
Sample [0.56] expected [0.5311862] produced [0.5363911712081236]
Sample [0.66] expected [0.61311685] produced [0.6320418483666338]
Sample [0.87] expected [0.76432894] produced [0.7837214924029302]
Sample [1.1] expected [0.89120736] produced [0.8776461873491674]
Sample [0.53] expected [0.50553334] produced [0.5054155244777995]
Sample [0.48] expected [0.46177918] produced [0.4531178967395886]
Sample [1.36] expected [0.9778646] produced [0.929604132550147]
Sample [0.61] expected [0.57286746] produced [0.5857047615761318]
Sample [0.24] expected [0.23770263] produced [0.2209457905658881]
Sample [0.44] expected [0.42593947] produced [0.41115767988408086]
Sample [0.47] expected [0.45288629] produced [0.4427985538071961]
Sample [0.17] expected [0.16918235] produced [0.1696910378200736]
Sample [0.83] expected [0.73793137] produced [0.7604741848386821]
Sample [0.97] expected [0.82488571] produced [0.8323515051748954]
Sample [0.54] expected [0.51413599] produced [0.5158495057564231]
Sample [0.34] expected [0.33348709] produced [0.3096760729218724]
Epoch 10000 RMSE =  0.028169598071233357
Final Epoch RMSE =  0.028169598071233357
Sample [0.] expected [0.] produced [0.08323238564923671]
Sample [0.02] expected [0.01999867] produced [0.09085101335729655]
Sample [0.03] expected [0.0299955] produced [0.09489016901570455]
Sample [0.04] expected [0.03998933] produced [0.09908794953120581]
Sample [0.05] expected [0.04997917] produced [0.10344829693144784]
Sample [0.06] expected [0.05996401] produced [0.1079750447799706]
Sample [0.07] expected [0.06994285] produced [0.11267189705001643]
Sample [0.09] expected [0.08987855] produced [0.12258994933395143]
Sample [0.1] expected [0.09983342] produced [0.1278177061397167]
Sample [0.11] expected [0.1097783] produced [0.13322863273183425]
Sample [0.12] expected [0.11971221] produced [0.13882543752462498]
Sample [0.13] expected [0.12963414] produced [0.1446105556958364]
Sample [0.14] expected [0.13954311] produced [0.15058612361191367]
Sample [0.16] expected [0.15931821] produced [0.16311550672130512]
Sample [0.18] expected [0.17902957] produced [0.17642373561200564]
Sample [0.19] expected [0.18885889] produced [0.18337136495031336]
Sample [0.2] expected [0.19866933] produced [0.19051458034496355]
Sample [0.22] expected [0.21822962] produced [0.2053847057470899]
Sample [0.23] expected [0.22797752] produced [0.2131088551015695]
Sample [0.25] expected [0.24740396] produced [0.22912457527211477]
Sample [0.26] expected [0.25708055] produced [0.23741025371142918]
Sample [0.27] expected [0.26673144] produced [0.2458763119512792]
Sample [0.28] expected [0.27635565] produced [0.2545184408775383]
Sample [0.29] expected [0.28595223] produced [0.26333178289526693]
Sample [0.3] expected [0.29552021] produced [0.27231093548682855]
Sample [0.31] expected [0.30505864] produced [0.28144995842005405]
Sample [0.32] expected [0.31456656] produced [0.29074238468123603]
Sample [0.33] expected [0.32404303] produced [0.30018123516538536]
Sample [0.35] expected [0.34289781] produced [0.3194678462237291]
Sample [0.36] expected [0.35227423] produced [0.32929927237367307]
Sample [0.37] expected [0.36161543] produced [0.339244508728794]
Sample [0.38] expected [0.37092047] produced [0.3492943641097361]
Sample [0.4] expected [0.38941834] produced [0.36966946024864555]
Sample [0.42] expected [0.40776045] produced [0.3903447523078056]
Sample [0.43] expected [0.4168708] produced [0.40076899886278883]
Sample [0.45] expected [0.43496553] produced [0.42173737672026473]
Sample [0.46] expected [0.44394811] produced [0.4322599319357742]
Sample [0.5] expected [0.47942554] produced [0.47435464708837366]
Sample [0.51] expected [0.48817725] produced [0.48482651245887304]
Sample [0.52] expected [0.49688014] produced [0.49525706771158534]
Sample [0.55] expected [0.52268723] produced [0.526203512085019]
Sample [0.57] expected [0.53963205] produced [0.5464550782794659]
Sample [0.58] expected [0.54802394] produced [0.5564414626203138]
Sample [0.59] expected [0.55636102] produced [0.5663244937394778]
Sample [0.6] expected [0.56464247] produced [0.5760969015204802]
Sample [0.62] expected [0.58103516] produced [0.5952830409302547]
Sample [0.63] expected [0.58914476] produced [0.6046845333468149]
Sample [0.64] expected [0.59719544] produced [0.6139509344849503]
Sample [0.65] expected [0.60518641] produced [0.6230773069591131]
Sample [0.67] expected [0.62098599] produced [0.6408925724312583]
Sample [0.68] expected [0.62879302] produced [0.6495739306700055]
Sample [0.7] expected [0.64421769] produced [0.6664686593879047]
Sample [0.71] expected [0.65183377] produced [0.6746771673595597]
Sample [0.72] expected [0.65938467] produced [0.6827238945039609]
Sample [0.73] expected [0.66686964] produced [0.690607434434949]
Sample [0.74] expected [0.67428791] produced [0.6983267617582833]
Sample [0.75] expected [0.68163876] produced [0.7058812144700495]
Sample [0.76] expected [0.68892145] produced [0.7132704756645787]
Sample [0.77] expected [0.69613524] produced [0.7204945547368021]
Sample [0.78] expected [0.70327942] produced [0.7275537682521587]
Sample [0.79] expected [0.71035327] produced [0.7344487206445104]
Sample [0.81] expected [0.72428717] produced [0.7477495832848836]
Sample [0.84] expected [0.74464312] produced [0.7664984498641632]
Sample [0.85] expected [0.75128041] produced [0.7724342373402517]
Sample [0.86] expected [0.75784256] produced [0.7782164592434316]
Sample [0.88] expected [0.77073888] produced [0.7893292691718758]
Sample [0.89] expected [0.77707175] produced [0.7946646890780735]
Sample [0.91] expected [0.78950374] produced [0.8049063861998579]
Sample [0.92] expected [0.79560162] produced [0.8098180362091415]
Sample [0.93] expected [0.80161994] produced [0.8145939072819022]
Sample [0.94] expected [0.8075581] produced [0.8192368257657481]
Sample [0.95] expected [0.8134155] produced [0.8237496541627315]
Sample [0.96] expected [0.81919157] produced [0.8281352812728395]
Sample [0.98] expected [0.83049737] produced [0.836536564211958]
Sample [0.99] expected [0.83602598] produced [0.8405580503943044]
Sample [1.] expected [0.84147098] produced [0.8444639811336084]
Sample [1.01] expected [0.84683184] produced [0.8482572533419356]
Sample [1.02] expected [0.85210802] produced [0.8519407454306358]
Sample [1.03] expected [0.85729899] produced [0.8555173119920697]
Sample [1.04] expected [0.86240423] produced [0.858989779018437]
Sample [1.05] expected [0.86742323] produced [0.8623609396278474]
Sample [1.06] expected [0.87235548] produced [0.8656335502678699]
Sample [1.07] expected [0.8772005] produced [0.8688103273671338]
Sample [1.08] expected [0.88195781] produced [0.8718939444060757]
Sample [1.09] expected [0.88662691] produced [0.874887029378616]
Sample [1.11] expected [0.89569869] produced [0.8806118749558799]
Sample [1.12] expected [0.90010044] produced [0.8833486462025255]
Sample [1.13] expected [0.90441219] produced [0.8860049039015098]
Sample [1.14] expected [0.9086335] produced [0.8885830223578497]
Sample [1.15] expected [0.91276394] produced [0.891085321904104]
Sample [1.16] expected [0.91680311] produced [0.8935140683879002]
Sample [1.17] expected [0.9207506] produced [0.895871472860427]
Sample [1.18] expected [0.92460601] produced [0.8981596914472202]
Sample [1.19] expected [0.92836897] produced [0.9003808253836922]
Sample [1.21] expected [0.935616] produced [0.9046299710327222]
Sample [1.22] expected [0.93909936] produced [0.9066619130705391]
Sample [1.23] expected [0.9424888] produced [0.9086346320847644]
Sample [1.24] expected [0.945784] produced [0.9105499600682273]
Sample [1.25] expected [0.94898462] produced [0.9124096769495826]
Sample [1.26] expected [0.95209034] produced [0.9142155113796658]
Sample [1.27] expected [0.95510086] produced [0.9159691415791669]
Sample [1.28] expected [0.95801586] produced [0.9176721962386959]
Sample [1.29] expected [0.96083506] produced [0.9193262554630647]
Sample [1.3] expected [0.96355819] produced [0.9209328517523047]
Sample [1.31] expected [0.96618495] produced [0.9224934710126025]
Sample [1.32] expected [0.9687151] produced [0.9240095535909453]
Sample [1.33] expected [0.97114838] produced [0.9254824953278467]
Sample [1.34] expected [0.97348454] produced [0.9269136486230555]
Sample [1.35] expected [0.97572336] produced [0.9283043235096544]
Sample [1.37] expected [0.97990806] produced [0.93096927282669]
Sample [1.38] expected [0.98185353] produced [0.9322459651945721]
Sample [1.39] expected [0.98370081] produced [0.9334870171753192]
Sample [1.4] expected [0.98544973] produced [0.9346935431074946]
Sample [1.41] expected [0.9871001] produced [0.9358666213804957]
Sample [1.42] expected [0.98865176] produced [0.9370072954734687]
Sample [1.43] expected [0.99010456] produced [0.9381165749798613]
Sample [1.44] expected [0.99145835] produced [0.9391954366160947]
Sample [1.47] expected [0.99492435] produced [0.9422588090054066]
Sample [1.49] expected [0.99673775] produced [0.9441654837908912]
Sample [1.5] expected [0.99749499] produced [0.9450806306936116]
Sample [1.51] expected [0.99815247] produced [0.9459713570662662]
Sample [1.52] expected [0.99871014] produced [0.9468384106722865]
Sample [1.53] expected [0.99916795] produced [0.9476825146055758]
Sample [1.54] expected [0.99952583] produced [0.9485043680964668]
Sample [1.55] expected [0.99978376] produced [0.9493046472958642]
Sample [1.56] expected [0.99994172] produced [0.9500840060376564]
Sample [1.57] expected [0.99999968] produced [0.9508430765795347]
Final RMSE =  0.03143962822264998

Process finished with exit code 0
"""
