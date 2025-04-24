We present simulations in the framework from Berthet et al to discover cases of non-trivial compositionality.

# Framework

The framework was as follows (small adjustments).

A situation of emission for a call is described as a binary vector of 0s and 1s, indicating which of a set of ordered features of context are true at emission time.

We assume, classically, that meanings can be any boolean functions based on these features: a call may or may not apply in a situation as described by these features. We also assume for modelling purposes that features (dimensions in a vector representing a situation) are probabilistically independent, each with probabilities p1, p2, … (all set to 1/2 by default).

We do not consider Berthet et al.’s dimension reduction step: if anything, it leads to less interpretable feature vectors, and thus also to less interpretable algebraic operations. This does not have a bearing on the two issues we highlight: if this dimension reduction step is successful, it should align the features with those that are relevant for the calls, as we assume here.

The simulations rely on assumptions about how calls would be produced.

- In the "no exhaustification" condition, we assume that this would be such that the situations where a call would be found (or its conjunction with another call) is simply the set of situations in which it is true, in proportions corresponding to the proportions of these situations in the real world. This is the "no exhaustification" condition.
- In the "exhaustification" condition, we assume that all calls which are appropriate in a situation will be produced. It results in e.g., a call A not to be produced on its own, if B is also appropriate (instead, the combination AB will be produced then).

We do not report on criterion (iii), which depends on the rest of the lexicon, which can therefore vary arbitrarily independently of the compositional rule involved in any given combination of calls.

# Statistics

Berthet et al. rely on statistics that we do not simulate here. We are in a simple position here, such that we can calculate expected value. Appropriate tests are those able to detect differences in expected values in this artificial setting.

Also, we use a different calculation of what is called chance value in the original paper. Instead of looking at the distances between various data halves, we look at all pairwise distances.

# Result

Results for 3 features are probably sufficient to get a sense of what would happen in higher dimensions, since there are only two critical calls at stake in each comparison. These results can be obtained by running `main.py`, and one of the two critical tables are reproduced here.

| exhaustification | criterion (i) | criterion (ii) | criterion (iv) | count | percentage |
| ---------------- | ------------- | -------------- | -------------- | ----- | ---------- |
| yes              | False         | NaN            | NaN            | 511   | 0.779724   |
| yes              | False         | False          | False          | 6305  | 9.620667   |
| yes              | False         | True           | False          | 6050  | 9.231567   |
| yes              | True          | NaN            | NaN            | 6050  | 9.231567   |
| yes              | True          | False          | True           | 96    | 0.146484   |
| yes              | True          | True           | False          | 32    | 0.048828   |
| yes              | True          | True           | True           | 46492 | 70.941162  |
| no               | False         | NaN            | NaN            | 511   | 0.779724   |
| no               | False         | False          | False          | 1     | 0.001526   |
| no               | False         | False          | True           | 229   | 0.349426   |
| no               | False         | True           | True           | 25    | 0.038147   |
| no               | True          | NaN            | NaN            | 6050  | 9.231567   |
| no               | True          | False          | True           | 43591 | 66.514587  |
| no               | True          | True           | True           | 15129 | 23.085022  |
