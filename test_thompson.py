import thompson

def test_initialize_alpha():
    test_list = [1, 2, 3]
    a = thompson.initialize(test_list)[0]
    assert len(a) == len(test_list)

def test_initialize_beta():
    test_list = [1, 2, 3]
    b = thompson.initialize(test_list)[1]
    assert max(b) == 1

def test_initialize_regret():
    test_list = [1, 2, 3]
    regret = thompson.initialize(test_list)[2]
    assert regret == 0

def test_evaluate():
    reward = thompson.evaluate(1)
    assert reward == 1

def test_thompson_sampl():
    chosen_arm = thompson.thompson_sampl([0.2, 0.3], [3, 5], [2, 8], 0)[0]
    assert chosen_arm >= 0