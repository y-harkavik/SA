import matplotlib.pyplot as plt
from numpy import array
from scipy.cluster.vq import kmeans


def print_results(vectors, centroids):
    x_axis = [vector[0] for vector in vectors]
    y_axis = [vector[1] for vector in vectors]
    z_axis = [vector[2] for vector in vectors]
    colors = ['r', 'g', 'b']
    n = int(len(vectors) / 5)

    fig = plt.figure()
    graph = fig.add_subplot(111, projection='3d')

    for i in range(n):
        bound = i * 5
        graph.scatter(x_axis[bound: bound + 5],
                      y_axis[bound: bound + 5],
                      z_axis[bound: bound + 5],
                      c=colors[i], marker='o')
        graph.scatter(centroids[i][0],
                      centroids[i][1],
                      centroids[i][2],
                      c='k', marker='x', s=36)

    graph.set_xlabel('Economics')
    graph.set_ylabel('Psychology')
    graph.set_zlabel('Probability theory')

    plt.show()


def get_thesauruses():
    thesauruses = []

    for i in range(3):
        thesaurus = []
        file = open('articles\\t' + str(i) + '.txt', 'r', encoding='utf-16')

        for word in file:
            thesaurus.append(word[:-1])

        file.close()
        thesauruses.append(thesaurus)

    print(thesauruses)
    return thesauruses


def get_vectors(thesauruses):
    vectors = []

    for n in range(15):
        vector = [.0, .0, .0]
        file = open('articles\\' + str(n) + '.txt', 'r', encoding='utf-16')

        for line in file:
            words = line.split(' ')

            for word in words:
                for i in range(3):
                    for meaning in thesauruses[i]:
                        if meaning in word:
                            vector[i] += 1

        file.close()
        vectors.append(vector)

        print(n, vector)

    return vectors


def main():
    thesauruses = get_thesauruses()
    vectors = get_vectors(thesauruses)

    centroids, _ = kmeans(array(vectors), 3)
    print('Coordinates of centroids', centroids)

    print_results(vectors, centroids)


if __name__ == '__main__':
    main()
