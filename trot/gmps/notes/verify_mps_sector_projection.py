"""Check the PDF's projection examples without third-party dependencies.

Tensors are dictionaries {(old_left, physical, old_right): amplitude}.
Bond-state triples and the charge convention match the notebook. This
checks the construction and arithmetic, not the NumPy/JAX runtime or SVD.
"""
from itertools import product
from math import isclose, sqrt
from random import Random

CHARGES = ((0, 0), (1, 0), (0, 1), (1, 1))


def product_mps(local_vectors):
    return [
        {(0, p, 0): value for p, value in enumerate(vector) if value != 0}
        for vector in local_vectors
    ]


def coefficient(tensors, configuration):
    environment = {0: 1}
    for tensor, physical in zip(tensors, configuration):
        next_environment = {}
        for (left, p, right), value in tensor.items():
            if p == physical:
                next_environment[right] = (
                    next_environment.get(right, 0) + environment.get(left, 0) * value
                )
        environment = next_environment
    return environment.get(0, 0)


def dense(tensors):
    return {
        configuration: coefficient(tensors, configuration)
        for configuration in product(range(4), repeat=len(tensors))
    }


def norm2(tensors):
    return sum(value * value for value in dense(tensors).values())


def project(tensors, target):
    # Step 3: existence of paths, ignoring amplitude signs.
    reachable = [{(0, 0, 0)}]
    for tensor in tensors:
        next_reachable = set()
        for old_left, up, down in reachable[-1]:
            for (a, p, b), value in tensor.items():
                if a != old_left or value == 0:
                    continue
                next_up = up + CHARGES[p][0]
                next_down = down + CHARGES[p][1]
                if next_up <= target[0] and next_down <= target[1]:
                    next_reachable.add((b, next_up, next_down))
        reachable.append(next_reachable)

    # Step 4: intersect reachability with the ability to finish.
    alive = [set() for _ in reachable]
    endpoint = (0, *target)
    if endpoint in reachable[-1]:
        alive[-1].add(endpoint)
    for site in range(len(tensors) - 1, -1, -1):
        for old_left, up, down in reachable[site]:
            for (a, p, b), value in tensors[site].items():
                destination = (b, up + CHARGES[p][0], down + CHARGES[p][1])
                if a == old_left and value != 0 and destination in alive[site + 1]:
                    alive[site].add((old_left, up, down))
                    break

    # Step 5: assign integer indices in the notebook's charge ordering.
    states = [sorted(bond, key=lambda row: (row[1], row[2], row[0])) for bond in alive]

    # Step 6: copy every signed amplitude with consistent count labels.
    projected = []
    for site, tensor in enumerate(tensors):
        right_lookup = {state: index for index, state in enumerate(states[site + 1])}
        B = {}
        for new_left, (old_left, up, down) in enumerate(states[site]):
            for (a, p, b), value in tensor.items():
                if a != old_left or value == 0:
                    continue
                destination = (b, up + CHARGES[p][0], down + CHARGES[p][1])
                new_right = right_lookup.get(destination)
                if new_right is not None:
                    B[new_left, p, new_right] = value
        projected.append(B)
    return projected, states, reachable, alive


def assert_close(actual, expected):
    assert isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), (actual, expected)


def check_dense_projection(original, projected, target):
    for configuration in product(range(4), repeat=len(original)):
        total = tuple(sum(CHARGES[p][spin] for p in configuration) for spin in (0, 1))
        expected = coefficient(original, configuration) if total == target else 0
        assert_close(coefficient(projected, configuration), expected)


def main():
    r = 1 / sqrt(2)
    main_trial = product_mps([(0, r, r, 0), (0, r, -r, 0)] * 2)
    projected, states, reachable, alive = project(main_trial, (2, 2))
    assert [len(bond) for bond in states] == [1, 2, 3, 2, 1]
    assert reachable == alive
    assert states == [
        [(0, 0, 0)],
        [(0, 0, 1), (0, 1, 0)],
        [(0, 0, 2), (0, 1, 1), (0, 2, 0)],
        [(0, 1, 2), (0, 2, 1)],
        [(0, 2, 2)],
    ]
    expected_matrices = [
        { (0, 1, 1): r, (0, 2, 0): r },
        { (0, 1, 1): r, (1, 1, 2): r, (0, 2, 0): -r, (1, 2, 1): -r },
        { (0, 1, 0): r, (1, 1, 1): r, (1, 2, 0): r, (2, 2, 1): r },
        { (0, 1, 0): r, (1, 2, 0): -r },
    ]
    assert projected == expected_matrices
    check_dense_projection(main_trial, projected, (2, 2))
    expected = {
        'uudd': -0.25, 'udud': 0.25, 'uddu': -0.25,
        'duud': -0.25, 'dudu': 0.25, 'dduu': -0.25,
    }
    for name, value in expected.items():
        configuration = tuple(1 if spin == 'u' else 2 for spin in name)
        assert_close(coefficient(projected, configuration), value)
    assert_close(norm2(main_trial), 1)
    assert_close(norm2(projected), 3 / 8)
    normalized = [dict(A) for A in projected]
    normalized[0] = {key: value / sqrt(3 / 8) for key, value in projected[0].items()}
    assert_close(norm2(normalized), 1)
    print('PASS: all 256 configurations agree with direct (2,2) projection.')
    print('PASS: all explicit matrices, six signed amplitudes, and bond labels agree.')
    print('PASS: projected norm^2 = 0.375; normalized norm^2 = 1.')

    weights = []
    for target_up, expected_weight in enumerate((1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)):
        target = (target_up, 4 - target_up)
        sector, *_ = project(main_trial, target)
        check_dense_projection(main_trial, sector, target)
        weights.append(norm2(sector))
        assert_close(weights[-1], expected_weight)
    assert_close(sum(weights), 1)
    print('PASS: all five S_z-sector weights = 1/16, 4/16, 6/16, 4/16, 1/16.')

    alternative = product_mps([
        (0, r, r, 0), (0, r, -r, 0), (0, 0, 0, -1), (1, 0, 0, 0),
    ])
    alternative_projection, _, forward, backward = project(alternative, (2, 2))
    assert forward[2] == {(0, 0, 2), (0, 1, 1), (0, 2, 0)}
    assert backward[2] == {(0, 1, 1)}
    check_dense_projection(alternative, alternative_projection, (2, 2))
    assert_close(coefficient(alternative_projection, (1, 2, 3, 0)), 0.5)
    assert_close(coefficient(alternative_projection, (2, 1, 3, 0)), -0.5)
    assert_close(norm2(alternative_projection), 0.5)
    print('PASS: hole/double example prunes two dead ends and retains norm^2 = 0.5.')

    # A general signed MPS exercises sums over more than one original index.
    rng = Random(812)
    dimensions = (1, 2, 3, 2, 1)
    general = []
    for left_dim, right_dim in zip(dimensions[:-1], dimensions[1:]):
        tensor = {}
        for a, p, b in product(range(left_dim), range(4), range(right_dim)):
            value = rng.choice((-2, -1, 0, 0, 1, 2))
            if value != 0:
                tensor[a, p, b] = value
        general.append(tensor)
    for target in product(range(5), repeat=2):
        sector, *_ = project(general, target)
        check_dense_projection(general, sector, target)
    print('PASS: general signed MPS agrees with dense filtering for all 25 count sectors.')

    # Two paths cancel in (1,1), even though both are structurally present.
    cancelling = [
        {(0, 1, 0): 1, (0, 1, 1): 1},
        {(0, 2, 0): 1, (1, 2, 0): -1, (0, 1, 0): 1},
    ]
    cancelled_sector, _, forward, backward = project(cancelling, (1, 1))
    assert (0, 1, 1) in forward[-1]
    assert backward[0] == {(0, 0, 0)}
    assert norm2(cancelling) == 1
    assert norm2(cancelled_sector) == 0
    check_dense_projection(cancelling, cancelled_sector, (1, 1))
    print('PASS: reachable but cancelling sector has zero norm, as expected.')


if __name__ == '__main__':
    main()
