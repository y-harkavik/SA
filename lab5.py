from minisom import MiniSom
from random import uniform as u
from random import randint as r


data = [[2500 / 60000, 0.3 / 60, 0.4 / 15, 0.04 / 1400000, 0.02],
        [4000 / 60000, 0.8 / 60, 0.9 / 15, 0.4 / 1400000, 0.02],
        [5500 / 60000, 1.1 / 60, 1.1 / 15, 1.2 / 1400000, 0.02],
        [7000 / 60000, 1.7 / 60, 1.3 / 15, 6 / 1400000, 0.02],
        [8000 / 60000, 3.1 / 60, 2.1 / 15, 80 / 1400000, 0.02],
        [20000 / 60000, 18 / 60, 7 / 15, 20000 / 1400000, 0.02],
        [50000 / 60000, 60 / 60, 15 / 15, 1400000 / 1400000, 0.02]]
        # [2500 / 60000, 0.2 / 60, 0.4 / 15, 0.04 / 1400000, 0.02],]


# Parameters: temperature, mass, radius, brightness, helium


def get_selection(count):
    selection = []

    for i in range(count):
        vector = [r(1000, 60000) / 60000,
                  u(0.1, 60) / 60,
                  u(0.1, 15) / 15,
                  u(0.01, 1400000) / 1400000,
                  u(0.01, 1)]
        selection.append(vector)

    return selection


def main():
    som = MiniSom(7, 1, 5, sigma=0.01, learning_rate=0.7)        # initialization of 7x1 SOM
    som.train_random(data, 200000)                               # trains the SOM with 100 iterations

    size = 300
    selection = get_selection(size)

    for i in range(size):
        print('Coordinates:', som.winner(selection[i]),
              'T:', selection[i][0] * 60000,
              '\tM:', selection[i][1] * 60,
              '\tR:', selection[i][2] * 15,
              '\tB:', selection[i][3] * 140000,
              '\tH:', selection[i][4])


if __name__ == '__main__':
    main()
