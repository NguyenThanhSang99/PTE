import numpy as np
import json
from sklearn import svm

train_documents = []
train_labels = []
test_documents = []
test_actual_labels = []
test_predicted_labels = []
all_words = json.load(open('word_mapping.json'))
word_embedding = np.load('pte.npy')

D = 40


def load_data(files):
    dataset = []
    document_no = 1

    class_labels = ['pos', 'neg']
    index = 0
    for file_name in files:
        fp = open(file_name)
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            words = line.split(" ")
            document = (words, class_labels[index], document_no)
            dataset.append(document)
            document_no += 1
        index += 1

    return dataset


def train(train_set):
    for index in range(len(train_set)):
        document_sum = np.zeros(D)
        for word_index in range(len(train_set[index][0])):
            word = train_set[index][0][word_index]
            i = all_words[word]
            embedding = word_embedding[i-1]
            document_sum = np.add(document_sum, embedding)
        document_average = np.divide(
            document_sum, len(train_set[index][0]))
        train_documents.append(document_average)
        train_labels.append(train_set[index][1])


def test(train_set, test_set):
    for index in range(len(test_set)):
        document_sum = np.zeros(D)
        for word_index in range(len(test_set[index][0])):
            word = test_set[index][0][word_index]
            i = all_words[word]
            embedding = word_embedding[i-1]
            document_sum = np.add(document_sum, embedding)
        document_average = np.divide(document_sum, len(test_set[index][0]))
        test_documents.append(document_average)
        test_actual_labels.append(train_set[index][1])
    clf = svm.SVC()
    clf.fit(train_documents, train_labels)
    test_predicted_labels = clf.predict(test_documents)
    correct = 0
    for i in range(len(test_predicted_labels)):
        if test_predicted_labels[i] == test_actual_labels[i]:
            correct = correct + 1
    accuracy = (correct)/float(len(test_predicted_labels)) * 100.0
    print('Accuracy : ', accuracy)


def main():
    train_files = ['./data/train-pos.txt', './data/train-neg.txt']
    test_files = ['./data/test-pos.txt', './data/test-neg.txt']
    train_set = load_data(train_files)
    test_set = load_data(test_files)

    train(train_set)

    test(train_set, test_set)


if __name__ == "__main__":
    main()
