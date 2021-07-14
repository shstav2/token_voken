import random


ONE_PERCENT = 0.01
TEN_PERCENT = 0.1


def one_percent_chance():
    return random.random() < ONE_PERCENT


def ten_percent_chance():
    return random.random() < TEN_PERCENT
