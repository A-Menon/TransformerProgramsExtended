import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output\rasp\reverse\baseline\modules_none\k8_len8_L3_H8_M2\s0\reverse_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 3

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 7}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 7}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 1

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5, 6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 3

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6, 7}:
            return k_position == 1

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {4, 7}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 6

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5, 6, 7}:
            return k_position == 1

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 3, 4, 5}:
            return k_position == 7
        elif q_position in {6, 7}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        if key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("1", "</s>"),
            ("3", "1"),
            ("4", "1"),
            ("</s>", "1"),
            ("<s>", "1"),
        }:
            return 4
        elif key in {("1", "2"), ("2", "1")}:
            return 6
        return 7

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
        }:
            return 4
        elif key in {
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 6),
            (2, 7),
        }:
            return 3
        return 6

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 5, 7}:
            return k_position == 7
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {2, 6}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {6, 7}:
            return k_position == 1

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 3, 4}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 0

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 6
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4, 5, 6}:
            return k_position == 1

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 3, 7}:
            return position == 3
        elif num_mlp_0_0_output in {1}:
            return position == 5
        elif num_mlp_0_0_output in {2, 5}:
            return position == 6
        elif num_mlp_0_0_output in {4}:
            return position == 4
        elif num_mlp_0_0_output in {6}:
            return position == 1

    num_attn_1_0_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"3", "4", "0"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"1", "<s>"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"2", "</s>"}:
            return k_attn_0_3_output == "<s>"

    num_attn_1_1_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(token, position):
        if token in {"3", "0", "</s>"}:
            return position == 4
        elif token in {"1", "4"}:
            return position == 3
        elif token in {"2", "<s>"}:
            return position == 1

    num_attn_1_2_pattern = select(positions, tokens, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6, 7}:
            return k_position == 3

    num_attn_1_3_pattern = select(positions, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_0_2_output):
        key = (attn_1_2_output, attn_0_2_output)
        if key in {
            ("1", "1"),
            ("2", "1"),
            ("4", "1"),
            ("4", "2"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "4"),
        }:
            return 5
        elif key in {
            ("1", "</s>"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 1
        elif key in {
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("3", "0"),
            ("3", "2"),
            ("3", "4"),
            ("</s>", "3"),
        }:
            return 3
        elif key in {("0", "</s>"), ("3", "1"), ("3", "3")}:
            return 0
        elif key in {("2", "</s>")}:
            return 4
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        if key in {(0, 0)}:
            return 0
        return 4

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 1, 2}:
            return k_position == 1
        elif q_position in {3, 6}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5, 7}:
            return k_position == 7

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 7
        elif q_position in {1, 2}:
            return k_position == 3
        elif q_position in {3, 4, 5, 6}:
            return k_position == 1

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 4}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 4

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, num_mlp_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 7
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5, 6}:
            return k_position == 1

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_0_output, attn_0_1_output):
        if num_mlp_1_0_output in {0, 4, 7}:
            return attn_0_1_output == "<pad>"
        elif num_mlp_1_0_output in {1}:
            return attn_0_1_output == "4"
        elif num_mlp_1_0_output in {2, 6}:
            return attn_0_1_output == "2"
        elif num_mlp_1_0_output in {3, 5}:
            return attn_0_1_output == "1"

    num_attn_2_0_pattern = select(
        attn_0_1_outputs, num_mlp_1_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_0_0_output):
        if attn_1_1_output in {"1", "4", "0", "3"}:
            return attn_0_0_output == "</s>"
        elif attn_1_1_output in {"2"}:
            return attn_0_0_output == "<pad>"
        elif attn_1_1_output in {"<s>", "</s>"}:
            return attn_0_0_output == "<s>"

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_var0_embeddings)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, attn_1_3_output):
        if attn_1_1_output in {"2", "0"}:
            return attn_1_3_output == "</s>"
        elif attn_1_1_output in {"1", "<s>"}:
            return attn_1_3_output == "<pad>"
        elif attn_1_1_output in {"3", "4"}:
            return attn_1_3_output == "<s>"
        elif attn_1_1_output in {"</s>"}:
            return attn_1_3_output == "0"

    num_attn_2_2_pattern = select(attn_1_3_outputs, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_0_0_output, attn_0_3_output):
        if num_mlp_0_0_output in {0, 5, 7}:
            return attn_0_3_output == "0"
        elif num_mlp_0_0_output in {1, 4}:
            return attn_0_3_output == "1"
        elif num_mlp_0_0_output in {2}:
            return attn_0_3_output == "<s>"
        elif num_mlp_0_0_output in {3}:
            return attn_0_3_output == "<pad>"
        elif num_mlp_0_0_output in {6}:
            return attn_0_3_output == "3"

    num_attn_2_3_pattern = select(
        attn_0_3_outputs, num_mlp_0_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_2_output, attn_1_2_output):
        key = (attn_0_2_output, attn_1_2_output)
        if key in {
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "<s>"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("<s>", "1"),
        }:
            return 0
        elif key in {("</s>", "0"), ("</s>", "3"), ("<s>", "3")}:
            return 4
        elif key in {("0", "</s>"), ("1", "</s>"), ("3", "</s>")}:
            return 6
        elif key in {("</s>", "4")}:
            return 1
        elif key in {("2", "</s>")}:
            return 7
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_0_output, num_attn_1_3_output):
        key = (num_attn_0_0_output, num_attn_1_3_output)
        if key in {
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
        }:
            return 5
        elif key in {(0, 0)}:
            return 6
        return 7

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                num_mlp_2_0_output_scores,
                num_var0_embedding_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "0", "4", "1", "1", "4", "2", "</s>"]))
