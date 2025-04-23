from collections import Counter
from dataclasses import dataclass
from itertools import product, chain, combinations
import numpy as np
from typing import Callable, Optional


@dataclass
class Call:
    meaning: Optional[Callable[[np.ndarray], bool]] = None
    worlds: Optional[np.ndarray] = None
    probas: Optional[np.ndarray] = None
    chance: Optional[float] = None


def calculate_chance(call):
    call.chance = get_distance(call, call)
    return call


def get_universe(n_features, probabilities):
    if probabilities is None:
        probabilities = np.ones(n_features) * 0.5
    n_worlds = 2**n_features
    worlds = np.array(
        [list(np.binary_repr(i, width=n_features)) for i in range(n_worlds)], dtype=int
    )
    probas = np.prod(
        worlds * probabilities + (1 - worlds) * (1 - probabilities), axis=1
    )
    return Call(worlds=worlds, probas=probas)


def get_worlds(universe, meaning):
    mask = np.array([meaning(world) for world in universe.worlds])
    call_worlds = universe.worlds[mask]
    call_probas = universe.probas[mask]
    call_probas = normalize_probas(call_probas)
    return call_worlds, call_probas


def normalize_probas(probas):
    total = sum(probas)
    if total > 0:
        probas = probas / total
    return probas


def make_meaning_from_DNF(dnf):
    all_cases = [make_meaning_from_case(case) for case in dnf]
    disjunction = make_disjunction_meaning(all_cases)
    return disjunction


def make_meaning_from_case(dnf_case):
    def meaning(vec):
        return all(
            (vec[i] == 1) if polarity else (vec[i] == 0) for i, polarity in dnf_case
        )

    return meaning


def make_disjunction_meaning(meanings):
    def meaning(vec):
        return any(m(vec) for m in meanings)

    return meaning


def make_conjunction_meaning(meanings):
    def meaning(vec):
        return all(m(vec) for m in meanings)

    return meaning


def make_negation_meaning(meaning):
    def meaning(vec):
        return not (meaning(vec))

    return meaning


def make_conjunction_dnf(dnf1, dnf2):
    return [sorted(set(clause1) | set(clause2)) for clause1 in dnf1 for clause2 in dnf2]


def add_calls(call1, call2):
    n_features = call1.worlds.shape[1]

    # Generate all combinations of sums
    world_sums = call1.worlds[:, None, :] + call2.worlds[None, :, :]
    prob_products = call1.probas[:, None] * call2.probas[None, :]

    # Reshape to 2D
    world_sums = world_sums.reshape(-1, n_features)
    prob_products = prob_products.reshape(-1)

    # Collapse duplicates by summing probabilities
    unique_worlds, inverse = np.unique(world_sums, axis=0, return_inverse=True)
    summed_probas = np.zeros(len(unique_worlds))
    np.add.at(summed_probas, inverse, prob_products)

    return Call(worlds=unique_worlds, probas=summed_probas)


def get_distance(call1, call2):
    # Expand dims to broadcast subtraction across all pairs
    diff = (
        call1.worlds[:, None, :] - call2.worlds[None, :, :]
    )  # shape: (N1, N2, n_features)
    dists = np.linalg.norm(diff, axis=2)  # Euclidean distances, shape: (N1, N2)

    # Outer product of probabilities
    weights = call1.probas[:, None] * call2.probas[None, :]  # shape: (N1, N2)

    # Weighted sum of distances
    expected_distance = np.sum(dists * weights)

    return expected_distance


def make_feature_call(universe: Call, dnf: list[list[int]]) -> Call:

    meaning = make_meaning_from_DNF(dnf)
    worlds, probas = get_worlds(universe, meaning)
    result = Call(meaning=meaning, worlds=worlds, probas=probas)
    result.chance = get_distance(result, result)

    return result


def test_criterion_1(A, B):
    d_AB = get_distance(A, B)
    mean_chance = (A.chance + B.chance) / 2
    if d_AB > mean_chance:
        return True, f"✅ (i) d(A, B) ({d_AB:.2f}) > chance ({mean_chance:.2f})"
    else:
        return False, f"❌ (i) d(A, B) ({d_AB:.2f}) > chance ({mean_chance:.2f})"


def test_criterion_2(A, B, AB):
    d_AB_A = get_distance(AB, A)
    d_AB_B = get_distance(AB, A)
    if d_AB_A > A.chance and d_AB_B > B.chance:
        return (
            True,
            f"✅ (ii) d(AB, A) ({d_AB_A:.2f}) > chance A ({A.chance:.2f}) and d(AB, B) ({d_AB_B:.2f}) > chance B ({B.chance:.2f})",
        )

    else:
        return (
            False,
            f"❌ (ii) d(AB, A)={d_AB_A:.2f}, chance A={A.chance:.2f}; d(AB, B)={d_AB_B:.2f}, chance B={B.chance:.2f}",
        )


def test_criterion_4(A, B, AB, addition_vec):
    d_comp = get_distance(AB, addition_vec)
    mean_chance = (A.chance + B.chance) / 2
    if d_comp > mean_chance:
        return True, f"✅ (iv) d(AB, A+B) ({d_comp:.2f}) > chance ({mean_chance:.2f})"
    else:
        return False, f"❌ (iv) d(AB, A+B) ({d_comp:.2f}) > chance ({mean_chance:.2f})"


def test_full(universe, A_dnf, B_dnf):
    A = make_feature_call(universe, A_dnf)
    B = make_feature_call(universe, B_dnf)
    criterion_1, _ = test_criterion_1(A, B)
    AB_dnf = make_conjunction_dnf(A_dnf, B_dnf)
    AB = make_feature_call(universe, AB_dnf)
    addition_vec = add_calls(A, B)
    if len(AB.worlds) > 0:
        criterion_2, _ = test_criterion_2(A, B, AB)
        # criterion_3 = test_criterion_3(A, B, AB, addition_vec)
        criterion_4, _ = test_criterion_4(A, B, AB, addition_vec)

    else:
        criterion_2 = None
        criterion_4 = None

    return A, B, AB, addition_vec, criterion_1, criterion_2, criterion_4


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def world_to_clause(world):
    return [(i, bool(bit)) for i, bit in enumerate(world)]


def all_dnfs(n_features):
    """
    Yield all possible DNFs over n_features as lists of clauses.
    Each clause corresponds to a world (input) where the function is true.
    """
    worlds = list(product([0, 1], repeat=n_features))
    for true_set in powerset(worlds):
        if not true_set:
            yield []  # Always false
        else:
            yield [world_to_clause(w) for w in true_set]


if __name__ == "__main__":

    N_FEATURES = 3
    # probabilities = np.array([1/10, 1/5, 1/2])
    # universe = get_universe(N_FEATURES, probabilities)
    universe = get_universe(N_FEATURES, None)

    # Simple test:
    test_full(universe, [[(0, True)]], [[(1, True)]])

    # Test all combinations
    results = []

    for dnf1 in all_dnfs(N_FEATURES):
        for dnf2 in all_dnfs(N_FEATURES):
            A, B, AB, addition_vec, c1, c2, c4 = test_full(universe, dnf1, dnf2)
            results.append((c1, c2, c4))
    total_cases = len(results)
    result_counts = Counter(results)
    print(
        f"Summary of unique result combinations (criteria 1, 2, 4) out of {len(results)} possibilities:"
    )
    for result, count in result_counts.items():
        print(f"{result}: {count} occurrences ({100*count/total_cases:.2f}%)")

# For N_FEATURES = 3
# Summary of unique result combinations (criteria 1, 2, 4) out of 65536 possibilities:
# (False, None, None): 511 occurrences (0.78%)
# (False, False, False): 1 occurrences (0.00%)
# (True, None, None): 6050 occurrences (9.23%)
# (True, False, True): 43591 occurrences (66.51%)
# (False, False, True): 254 occurrences (0.39%)
# (True, True, True): 15129 occurrences (23.09%)
