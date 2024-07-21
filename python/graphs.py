# Ryan McShane
# 03-08-2024

# Prints out graphs to visually represent log/packet statistics

from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()

# Export Tree as PDF
def graphTree(clfd, filename):
    dot_data = tree.export_graphviz(clfd, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(filename)