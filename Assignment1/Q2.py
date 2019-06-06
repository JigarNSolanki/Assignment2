import numpy as np

class Node:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
 
def BST(A):
    if not A:
        return None
 
    mid = int(len(A) / 2)
    root = Node(A[mid])

    root.left = BST(A[:mid])
    root.right = BST(A[mid+1:])
    return root
 
def InOrder(node):
    if not node:
        return
    InOrder(node.left)
    print(node.data)
    InOrder(node.right)

A = list(np.random.randint(low=1,high=100,size=10))
A.sort()

root = BST(A)
InOrder(root) 