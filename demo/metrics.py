import torchvision.transforms as transforms
import random
import re
import torch
#transform = transforms.ToPILImage()

from zss import Node, simple_distance

# Define the structure of the trees
def create_tree(text):
    if text == "":
        return None
    nodes = [Node(char) for char in text]
    for i in range(len(nodes) - 1):
        nodes[i].addkid(nodes[i + 1])
    return nodes[0]

# Calculate the TED using the zss library
def calculate_ted(predicted_text, ground_truth_text):
    predicted_tree = create_tree(predicted_text)
    ground_truth_tree = create_tree(ground_truth_text)
    distance = simple_distance(predicted_tree, ground_truth_tree)
    return distance

def TED_similarity_score(predicted_text,ground_truth_text):
    if predicted_text=="" or ground_truth_text=="":
        return 0

    ted = calculate_ted(predicted_text, ground_truth_text)
    similarity_score = 1 - ted / max(len(predicted_text), len(ground_truth_text))

    return similarity_score

