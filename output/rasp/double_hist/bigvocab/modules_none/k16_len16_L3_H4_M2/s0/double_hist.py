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
        "output\rasp\double_hist\bigvocab\modules_none\k16_len16_L3_H4_M2\s0\double_hist_weights.csv",
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
        if q_position in {0, 11}:
            return k_position == 5
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {9, 4, 5, 7}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {12, 15}:
            return k_position == 2
        elif q_position in {13, 14}:
            return k_position == 1

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 2}:
            return k_position == 3
        elif q_position in {3, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9, 11}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 2
        elif q_position in {13, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 13

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0", "9"}:
            return k_token == "7"
        elif q_token in {"1"}:
            return k_token == "13"
        elif q_token in {"10", "11"}:
            return k_token == "0"
        elif q_token in {"4", "12"}:
            return k_token == "8"
        elif q_token in {"13"}:
            return k_token == "3"
        elif q_token in {"2", "8"}:
            return k_token == "10"
        elif q_token in {"3", "7"}:
            return k_token == "11"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"6"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "1"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {8, 2}:
            return k_position == 5
        elif q_position in {10, 3}:
            return k_position == 2
        elif q_position in {4, 13}:
            return k_position == 6
        elif q_position in {11, 12, 5, 14}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {9, 7}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 15

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 10),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (9, 10),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
        }:
            return 12
        elif key in {
            (0, 7),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 7),
            (3, 7),
            (4, 7),
            (5, 7),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
        }:
            return 7
        elif key in {
            (0, 9),
            (2, 9),
            (3, 9),
            (4, 9),
            (5, 9),
            (6, 9),
            (7, 9),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
        }:
            return 11
        elif key in {
            (0, 8),
            (2, 2),
            (2, 6),
            (2, 8),
            (3, 6),
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 8),
        }:
            return 6
        elif key in {
            (0, 0),
            (0, 4),
            (1, 0),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        }:
            return 4
        elif key in {
            (0, 5),
            (0, 6),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
        }:
            return 5
        elif key in {(0, 1), (0, 2), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)}:
            return 15
        return 14

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 11
        elif key in {1}:
            return 3
        return 13

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {0, 3, 4, 6, 7, 8}:
            return position == 6
        elif attn_0_0_output in {1, 13, 14}:
            return position == 13
        elif attn_0_0_output in {2, 11, 12, 5}:
            return position == 12
        elif attn_0_0_output in {9, 10}:
            return position == 11
        elif attn_0_0_output in {15}:
            return position == 14

    attn_1_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 12
        elif q_position in {1, 2, 3, 4, 5, 6, 7}:
            return k_position == 7
        elif q_position in {8, 9}:
            return k_position == 9
        elif q_position in {10, 11, 12, 13}:
            return k_position == 10
        elif q_position in {14, 15}:
            return k_position == 11

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {0}:
            return position == 1
        elif attn_0_0_output in {1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14}:
            return position == 7
        elif attn_0_0_output in {3}:
            return position == 2
        elif attn_0_0_output in {4}:
            return position == 3
        elif attn_0_0_output in {12, 15}:
            return position == 9

    num_attn_1_0_pattern = select(positions, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"10"}:
            return k_token == "10"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"12"}:
            return k_token == "12"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"6"}:
            return k_token == "6"
        elif q_token in {"7"}:
            return k_token == "7"
        elif q_token in {"8"}:
            return k_token == "8"
        elif q_token in {"9"}:
            return k_token == "9"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_0_output):
        key = (attn_1_1_output, attn_1_0_output)
        if key in {
            (0, 12),
            (1, 12),
            (2, 1),
            (2, 11),
            (2, 12),
            (2, 13),
            (3, 12),
            (8, 12),
            (9, 1),
            (9, 3),
            (9, 9),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 15),
            (10, 9),
            (10, 12),
            (10, 13),
            (13, 12),
            (14, 0),
            (14, 1),
            (14, 3),
            (14, 4),
            (14, 9),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 15),
        }:
            return 9
        elif key in {
            (1, 4),
            (1, 7),
            (1, 8),
            (2, 7),
            (2, 8),
            (5, 7),
            (6, 7),
            (6, 8),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 6),
            (8, 8),
            (9, 4),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 10),
            (11, 6),
            (11, 7),
            (11, 8),
            (13, 6),
        }:
            return 4
        elif key in {
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 6),
            (1, 0),
            (1, 6),
            (2, 0),
            (2, 4),
            (2, 6),
            (3, 0),
            (3, 3),
            (3, 4),
            (3, 6),
            (4, 0),
            (4, 3),
            (4, 4),
            (5, 6),
            (6, 6),
        }:
            return 15
        elif key in {
            (0, 5),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 15),
            (10, 5),
            (13, 5),
            (15, 5),
        }:
            return 14
        elif key in {(2, 9)}:
            return 5
        return 1

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        if key in {0}:
            return 15
        elif key in {1}:
            return 4
        return 6

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_0_output, position):
        if attn_0_0_output in {0, 3, 4, 5, 6, 8}:
            return position == 8
        elif attn_0_0_output in {1}:
            return position == 4
        elif attn_0_0_output in {9, 2, 7}:
            return position == 10
        elif attn_0_0_output in {10}:
            return position == 11
        elif attn_0_0_output in {11, 14}:
            return position == 12
        elif attn_0_0_output in {12}:
            return position == 13
        elif attn_0_0_output in {13}:
            return position == 14
        elif attn_0_0_output in {15}:
            return position == 3

    attn_2_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 3, 5}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {11, 12, 6}:
            return k_position == 13
        elif q_position in {9, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {13, 14, 15}:
            return k_position == 6

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {0, 1, 2, 3}:
            return k_attn_1_0_output == 4
        elif q_attn_1_0_output in {4, 5}:
            return k_attn_1_0_output == 3
        elif q_attn_1_0_output in {6}:
            return k_attn_1_0_output == 6
        elif q_attn_1_0_output in {9, 7}:
            return k_attn_1_0_output == 7
        elif q_attn_1_0_output in {8}:
            return k_attn_1_0_output == 8
        elif q_attn_1_0_output in {10}:
            return k_attn_1_0_output == 10
        elif q_attn_1_0_output in {11}:
            return k_attn_1_0_output == 5
        elif q_attn_1_0_output in {12, 14}:
            return k_attn_1_0_output == 11
        elif q_attn_1_0_output in {13}:
            return k_attn_1_0_output == 12
        elif q_attn_1_0_output in {15}:
            return k_attn_1_0_output == 14

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {0, 1, 2, 3}:
            return k_attn_1_0_output == 4
        elif q_attn_1_0_output in {4, 5}:
            return k_attn_1_0_output == 3
        elif q_attn_1_0_output in {6}:
            return k_attn_1_0_output == 6
        elif q_attn_1_0_output in {9, 7}:
            return k_attn_1_0_output == 7
        elif q_attn_1_0_output in {8}:
            return k_attn_1_0_output == 8
        elif q_attn_1_0_output in {10}:
            return k_attn_1_0_output == 10
        elif q_attn_1_0_output in {11}:
            return k_attn_1_0_output == 5
        elif q_attn_1_0_output in {12, 14}:
            return k_attn_1_0_output == 11
        elif q_attn_1_0_output in {13}:
            return k_attn_1_0_output == 12
        elif q_attn_1_0_output in {15}:
            return k_attn_1_0_output == 14

    num_attn_2_1_pattern = select(attn_1_0_outputs, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_1_output, attn_2_0_output):
        key = (attn_1_1_output, attn_2_0_output)
        if key in {
            (2, 5),
            (2, 14),
            (2, 15),
            (3, 5),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 14),
            (4, 15),
            (5, 0),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 13),
            (5, 14),
            (5, 15),
            (7, 14),
            (7, 15),
            (8, 9),
            (8, 14),
            (8, 15),
            (9, 2),
            (9, 5),
            (9, 7),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (10, 13),
            (10, 14),
            (10, 15),
            (11, 14),
            (11, 15),
            (12, 0),
            (12, 3),
            (12, 5),
            (12, 7),
            (12, 9),
            (12, 13),
            (12, 14),
            (12, 15),
            (13, 5),
            (13, 7),
            (13, 14),
            (13, 15),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 5),
            (14, 7),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (15, 5),
            (15, 15),
        }:
            return 3
        elif key in {
            (1, 9),
            (3, 9),
            (4, 9),
            (5, 7),
            (5, 9),
            (5, 12),
            (6, 9),
            (7, 9),
            (9, 4),
            (9, 6),
            (9, 10),
            (10, 4),
            (10, 6),
            (10, 9),
            (11, 0),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 9),
            (12, 1),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 10),
            (12, 11),
            (12, 12),
            (13, 4),
            (13, 6),
            (14, 4),
            (14, 6),
            (14, 10),
            (15, 9),
        }:
            return 11
        elif key in {
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 15),
            (1, 0),
            (1, 8),
            (2, 0),
            (2, 4),
            (2, 6),
            (2, 7),
            (2, 8),
            (3, 8),
            (6, 0),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (7, 0),
            (7, 8),
            (8, 0),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        }:
            return 1
        elif key in {
            (1, 5),
            (3, 4),
            (3, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (4, 7),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (10, 5),
            (10, 7),
            (15, 6),
            (15, 7),
        }:
            return 6
        elif key in {
            (2, 3),
            (3, 0),
            (3, 3),
            (4, 0),
            (4, 3),
            (10, 0),
            (10, 3),
            (13, 0),
            (13, 3),
            (15, 0),
            (15, 3),
        }:
            return 0
        elif key in {(1, 3), (1, 4), (1, 6), (1, 7)}:
            return 15
        elif key in {(4, 4), (15, 4), (15, 14)}:
            return 9
        elif key in {(9, 3), (9, 11)}:
            return 5
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
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
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (9, 0),
            (9, 1),
            (9, 2),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
        }:
            return 3
        return 14

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
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
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                mlp_2_0_output_scores,
                num_mlp_2_0_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
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


print(run(["<s>", "1", "8", "1", "2", "9", "0", "5", "8", "2", "11", "10", "9"]))
