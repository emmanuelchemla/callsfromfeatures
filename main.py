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

    def meaning(x):
        return True

    result = Call(worlds=worlds, probas=probas, meaning=meaning)
    result.chance = get_distance(result, result)

    return result


def filter_call(source_call, meaning):
    if len(source_call.worlds) < 1:
        return source_call
    mask = np.array([meaning(world) for world in source_call.worlds])
    call_worlds = source_call.worlds[mask]
    call_probas = source_call.probas[mask]
    call_probas = normalize_probas(call_probas)
    result = Call(worlds=call_worlds, probas=call_probas, meaning=meaning)
    result.chance = get_distance(result, result)

    return result


def normalize_probas(probas):
    total = sum(probas)
    if total > 0:
        probas = probas / total
    return probas


def get_meaning_from_DNF(dnf):
    all_cases = [get_meaning_from_case(case) for case in dnf]
    disjunction = get_disjunction(all_cases)
    return disjunction


def get_meaning_from_case(dnf_case):
    def meaning(vec):
        return all(
            (vec[i] == 1) if polarity else (vec[i] == 0) for i, polarity in dnf_case
        )

    return meaning


def get_disjunction(meanings):
    def meaning(vec):
        return any(m(vec) for m in meanings)

    return meaning


def get_conjunction(meanings):
    def meaning(vec):
        return all(m(vec) for m in meanings)

    return meaning


def get_negation(meaning):
    def neg_meaning(vec):
        return not (meaning(vec))

    return neg_meaning


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


def get_call_from_DNF(universe: Call, dnf: list[list[int]]) -> Call:
    meaning = get_meaning_from_DNF(dnf)
    result = filter_call(universe, meaning)
    return result


def exh_calls(call1, call2):
    call1_exh = filter_call(call1, get_negation(call2.meaning))
    call2_exh = filter_call(call2, get_negation(call1.meaning))
    return call1_exh, call2_exh


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


def calls_from_DNF(universe, A_dnf, B_dnf):
    A = get_call_from_DNF(universe, A_dnf)
    B = get_call_from_DNF(universe, B_dnf)
    AB = filter_call(A, B.meaning)
    return A, B, AB


def test_full(A, B, AB):
    criterion_1, _ = test_criterion_1(A, B)
    addition_vec = add_calls(A, B)
    if len(AB.worlds) > 0:
        criterion_2, _ = test_criterion_2(A, B, AB)
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
    A, B, AB = calls_from_DNF(universe, [[(0, True)]], [[(1, True)]])
    test_full(A, B, AB)
    A_exh, B_exh = exh_calls(A, B)
    test_full(A_exh, B_exh, AB)

    # Test all combinations
    results = []

    for dnf1 in all_dnfs(N_FEATURES):
        for dnf2 in all_dnfs(N_FEATURES):
            A, B, AB = calls_from_DNF(universe, dnf1, dnf2)
            A, B, AB, addition_vec, c1, c2, c4 = test_full(A, B, AB)
            results.append(("no exh", c1, c2, c4))

            if c1 and c2 and (not (c4)):
                print(f"no exh: A={A.worlds}, B={B.worlds}")

            A_exh, B_exh = exh_calls(A, B)
            A, B, AB, addition_vec, c1, c2, c4 = test_full(A_exh, B_exh, AB)
            results.append(("exh", c1, c2, c4))
            if c1 and c2 and (not (c4)):
                print(f"exh: A={A.worlds}, B={B.worlds}")

    total_cases = len(results)
    result_counts = Counter(results)
    sorted_results = sorted(
        result_counts.items(),
        key=lambda item: tuple((v if v is not None else False) for v in item[0]),
    )

    print(
        f"Summary of unique result combinations (criteria 1, 2, 4) out of {len(results)} possibilities:"
    )
    for result, count in sorted_results:
        print(f"{result}: {count} occurrences ({100 * count / total_cases:.2f}%)")

# For N_FEATURES = 3
# Summary of unique result combinations (criteria 1, 2, 4) out of 13,1072 possibilities:
# ('exh', False, None, None): 511 occurrences (0.39%)
# ('exh', False, False, False): 6305 occurrences (4.81%)
# ('exh', False, True, False): 6050 occurrences (4.62%)
# ('exh', True, None, None): 6050 occurrences (4.62%)
# ('exh', True, False, True): 96 occurrences (0.07%)
# ('exh', True, True, False): 32 occurrences (0.02%)
# ('exh', True, True, True): 46492 occurrences (35.47%)
# ('no exh', False, None, None): 511 occurrences (0.39%)
# ('no exh', False, False, False): 1 occurrences (0.00%)
# ('no exh', False, False, True): 229 occurrences (0.17%)
# ('no exh', False, True, True): 25 occurrences (0.02%)
# ('no exh', True, None, None): 6050 occurrences (4.62%)
# ('no exh', True, False, True): 43591 occurrences (33.26%)
# ('no exh', True, True, True): 15129 occurrences (11.54%)
