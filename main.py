from collections import Counter
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
import tqdm
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


def calls_from_meanings(universe, meaning1, meaning2):
    A = filter_call(universe, meaning1)
    B = filter_call(universe, meaning2)
    AB = filter_call(A, B.meaning)
    return A, B, AB


def get_all_meanings(N_FEATURES):
    universe = get_universe(N_FEATURES, None)
    worlds = [tuple(world) for world in universe.worlds]  # make hashable

    for truth_values in itertools.product([False, True], repeat=len(worlds)):
        world_to_truth = dict(zip(worlds, truth_values))

        def meaning(vec, _map=world_to_truth):
            return _map[tuple(vec)]

        yield meaning


def format_worlds(worlds):
    return ", ".join("".join(str(bit) for bit in row) for row in worlds)


def test_all_meanings(N_FEATURES, probabilities):
    universe = get_universe(N_FEATURES, probabilities)

    results = []

    all_meanings = list(get_all_meanings(N_FEATURES))

    for meaning1, meaning2 in tqdm.tqdm(
        itertools.product(all_meanings, repeat=2),
        total=len(all_meanings) ** 2,
        desc="Testing all meaning pairs",
    ):
        A, B, AB = calls_from_meanings(universe, meaning1, meaning2)
        A, B, AB, addition_vec, c1, c2, c4 = test_full(A, B, AB)
        results.append(
            {
                "exhaustification": "no exh",
                "criterion 1": c1,
                "criterion 2": c2,
                "criterion 4": c4,
                "A": format_worlds(A.worlds),
                "B": format_worlds(B.worlds),
                "A&B": format_worlds(AB.worlds),
                "A+B": format_worlds(addition_vec.worlds),
            }
        )

        A_exh, B_exh = exh_calls(A, B)
        A_exh, B_exh, AB, addition_vec, c1, c2, c4 = test_full(A_exh, B_exh, AB)
        results.append(
            {
                "exhaustification": "exh",
                "criterion 1": c1,
                "criterion 2": c2,
                "criterion 4": c4,
                "A": format_worlds(A.worlds),
                "B": format_worlds(B.worlds),
                "A&B": format_worlds(AB.worlds),
                "A+B": format_worlds(addition_vec.worlds),
            }
        )

    results = pd.DataFrame(results)
    total_cases = len(results) / 2
    summary = (
        results.groupby(
            ["exhaustification", "criterion 1", "criterion 2", "criterion 4"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    summary["percentage"] = summary["count"] / total_cases * 100

    return results, summary


if __name__ == "__main__":

    N_FEATURES = 3
    probabilities = None
    # probabilities = np.array([1/250, 1/50, 1/10, 1/2])
    universe = get_universe(N_FEATURES, probabilities)

    # Specific tests
    meaning1 = get_meaning_from_DNF([[(0, True)]])
    meaning2 = get_meaning_from_DNF([[(1, True)]])
    A, B, AB = calls_from_meanings(universe, meaning1, meaning2)
    test_full(A, B, AB)
    A_exh, B_exh = exh_calls(A, B)
    test_full(A_exh, B_exh, AB)

    # All meanings
    results, summary = test_all_meanings(N_FEATURES, None)
    trivially_compositional = results[
        (results["criterion 1"] == True)
        & (results["criterion 2"] == True)
        & (results["criterion 4"] == False)
    ][["exhaustification", "A", "B", "A&B", "A+B"]]

    summary
    trivially_compositional
