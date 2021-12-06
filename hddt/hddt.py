import pandas as pd
import numpy as np


class Split:
    def __init__(self, feature, criterion, splitpoint):
        self.feature = feature
        self.criterion = criterion
        self.splitpoint = splitpoint

    def left(self, data):
        return data[data[self.feature] <= self.splitpoint]

    def right(self, data):
        return data[data[self.feature] > self.splitpoint]

    def __repr__(self):
        return f"Split({self.feature} > {self.splitpoint} ({self.criterion}))"


class HellingerCriterion:
    def __init__(self, min_samples_leaf):
        self.min_samples_leaf = min_samples_leaf

    def __call__(self, node):
        print(f"splitting node id = {node.id}; depth={node.depth}; total={node.total}; prediction={node.prediction}")
        xy = node.data
        yname = xy.columns[-1]
        best_feature = None
        best_criterion = -99999999
        best_splitpoint = None
        for feature in xy.columns[:-1]:
            s = xy.sort_values(feature)
            pos = 0
            neg = 0
            curval = None
            for t in s.itertuples():
                yval = getattr(t, yname)
                xval = getattr(t, feature)
                if (xval != curval
                        and (pos + neg) >= self.min_samples_leaf
                        and (node.total - pos - neg) >= self.min_samples_leaf):
                    criterion = self.value([pos, node.pos - pos],
                                           [neg, node.neg - neg],
                                           node.total)
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_feature = feature
                        best_splitpoint = curval
                curval = xval
                pos += yval
                neg += (1 - yval)

        if best_splitpoint is not None:
            node.split = Split(best_feature, best_criterion, best_splitpoint)
            left_data = xy[xy[best_feature] <= best_splitpoint]
            right_data = xy[xy[best_feature] > best_splitpoint]
            node.left = Node(depth=node.depth + 1, data=left_data, parent=node, node_id=2 * node.id + 1,
                             name=f"{node.name} -> ({best_feature} <= {best_splitpoint})")
            node.right = Node(depth=node.depth + 1, data=right_data, parent=node, node_id=2 * node.id + 2,
                              name=f"{node.name} -> ({best_feature} > {best_splitpoint})")

            return True
        return False

    @staticmethod
    def value(pos, neg, total):
        # sum across branches of (sqrt(pneg) - sqrt(ppos)) ** 2
        # don't need to take the final square root because it's just an order
        pneg = np.array(neg) / total
        ppos = np.array(pos) / total
        return np.sum(np.power(np.sqrt(pneg) - np.sqrt(ppos), 2))


class Node:
    def __init__(self, depth, node_id=0, data=None, left=None, right=None, split=None, parent=None, name="",
                 prediction=None):
        self.data = data
        self.split = split
        self.left = left
        self.right = right
        self.depth = depth
        self.parent = parent
        self.id = node_id
        self.name = name
        self._prediction = prediction
        if data is not None:
            self.pos = np.sum(data.values[:, -1])
            self.neg = len(data) - self.pos
            self.total = len(data)

    def empty_copy(self):
        # a copy without the data takes much less memory
        ret = Node(depth=self.depth, node_id=self.id, name=self.name, prediction=self.prediction, split=self.split,
                   parent=self.parent)
        ret.pos = self.pos
        ret.neg = self.neg
        ret.total = self.total
        if self.right is not None:
            ret.right = self.right.empty_copy()
        if self.left is not None:
            ret.left = self.left.empty_copy()
        return ret

    @property
    def prediction(self):
        return self._prediction or self.pos / self.total

    def __repr__(self):
        return f"{self.name} -> {self.prediction} ({self.pos} / {self.total})"

    def predict(self, df):
        if self.left is None:
            return pd.Series(np.ones(len(df)) * self.prediction, index=df.index)
        else:
            left = self.left.predict(self.split.left(df))
            right = self.right.predict(self.split.right(df))
            return pd.concat([left, right])


class SimpleTree:
    def __init__(self, criterion, max_depth=99999):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def print_tree(self):
        if self.root is None:
            print("<empty tree>")
        else:
            self._print_node(self.root)

    def _print_node(self, node):
        if node.left is None:
            return print(node)
        else:
            self._print_node(node.left)
            self._print_node(node.right)

    def empty_copy(self):
        # a copy without the data takes much less memory
        ret = SimpleTree(self.criterion, self.max_depth)
        if self.root is not None:
            ret.root = self.root.empty_copy()
        return ret

    def _add_nodes_edges(self, node, dot=None):
        from graphviz import Digraph
        if dot is None:
            dot = Digraph()
            dot.node(name=str(node.id), label=SimpleTree._node_label(self, node))

        if node.left:
            dot.node(name=str(node.left.id), label=SimpleTree._node_label(self, node.left))
            dot.edge(str(node.id), str(node.left.id))
            dot = SimpleTree._add_nodes_edges(self, node.left, dot=dot)

        if node.right:
            dot.node(name=str(node.right.id), label=SimpleTree._node_label(self, node.right))
            dot.edge(str(node.id), str(node.right.id))
            dot = SimpleTree._add_nodes_edges(self, node.right, dot=dot)

        return dot

    def _node_label(self, node):
        split_label = ""
        if node.id == 0:
            split_label = "root"
        else:
            split_label = node.name.split("->")[-1]
        split_prediction = ""
        try:
            split_prediction = f"{node.prediction:.2%} = {node.pos} / {node.total}"
        except AttributeError:
            pass
        return f"{split_label}\n{split_prediction}"

    def draw(self):
        return SimpleTree._add_nodes_edges(self, self.root)

    def fit(self, X, y, sample_weight=None):
        self.root = Node(data=pd.concat([X, y], axis='columns'), depth=0, name="root")
        self.split(self.root)

    def split(self, node):
        if node.depth == self.max_depth:
            return False
        new_split = self.criterion(node)
        if new_split:
            self.split(node.left)
            self.split(node.right)

    def predict(self, df):
        return self.root.predict(df)

    def predict_proba(self, df):
        prob1 = self.predict(df)
        prob0 = 1 - prob1
        scores = pd.concat([prob0, prob1], axis='columns')
        return scores.loc
