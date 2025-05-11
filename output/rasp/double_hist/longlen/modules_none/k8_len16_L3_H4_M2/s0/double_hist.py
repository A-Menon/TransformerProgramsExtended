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
        "output\rasp\double_hist\longlen\modules_none\k8_len16_L3_H4_M2\s0\double_hist_weights.csv",
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
        if q_position in {0, 1, 6}:
            return k_position == 1
        elif q_position in {2, 5, 15}:
            return k_position == 4
        elif q_position in {11, 3, 13}:
            return k_position == 6
        elif q_position in {9, 4, 12}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {10, 14}:
            return k_position == 5

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"<s>", "4", "1"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 2, 3, 4}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 0
        elif q_position in {7, 8, 9, 10, 13, 14}:
            return k_position == 7
        elif q_position in {11, 12}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 12

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {("0", "3"), ("1", "3"), ("2", "3"), ("4", "3"), ("5", "3")}:
            return 15
        return 9

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
            return 13
        elif key in {1}:
            return 15
        elif key in {2}:
            return 8
        return 0

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0}:
            return k_num_mlp_0_0_output == 0
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 1
        elif q_num_mlp_0_0_output in {2}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {3}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 4
        elif q_num_mlp_0_0_output in {5}:
            return k_num_mlp_0_0_output == 5
        elif q_num_mlp_0_0_output in {6}:
            return k_num_mlp_0_0_output == 6
        elif q_num_mlp_0_0_output in {7}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {8}:
            return k_num_mlp_0_0_output == 7
        elif q_num_mlp_0_0_output in {9}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {10}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {11, 12, 13, 14, 15}:
            return k_num_mlp_0_0_output == 15

    attn_1_0_pattern = select_closest(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0}:
            return mlp_0_0_output == 4
        elif num_mlp_0_0_output in {1, 2, 6, 7, 9, 10, 13}:
            return mlp_0_0_output == 15
        elif num_mlp_0_0_output in {8, 3}:
            return mlp_0_0_output == 0
        elif num_mlp_0_0_output in {4, 5}:
            return mlp_0_0_output == 11
        elif num_mlp_0_0_output in {11, 12, 14}:
            return mlp_0_0_output == 13
        elif num_mlp_0_0_output in {15}:
            return mlp_0_0_output == 7

    attn_1_1_pattern = select_closest(
        mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_1_1
    )
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 1, 2, 4, 5, 7, 11, 15}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {3, 6, 8, 9, 10, 12, 13, 14}:
            return k_num_mlp_0_0_output == 7

    num_attn_1_0_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0, 7}:
            return mlp_0_0_output == 11
        elif num_mlp_0_0_output in {1}:
            return mlp_0_0_output == 1
        elif num_mlp_0_0_output in {2}:
            return mlp_0_0_output == 14
        elif num_mlp_0_0_output in {3}:
            return mlp_0_0_output == 3
        elif num_mlp_0_0_output in {4}:
            return mlp_0_0_output == 4
        elif num_mlp_0_0_output in {5}:
            return mlp_0_0_output == 5
        elif num_mlp_0_0_output in {6}:
            return mlp_0_0_output == 6
        elif num_mlp_0_0_output in {8}:
            return mlp_0_0_output == 0
        elif num_mlp_0_0_output in {9}:
            return mlp_0_0_output == 13
        elif num_mlp_0_0_output in {10}:
            return mlp_0_0_output == 7
        elif num_mlp_0_0_output in {11, 12, 13, 14, 15}:
            return mlp_0_0_output == 9

    num_attn_1_1_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(token, attn_1_0_output):
        key = (token, attn_1_0_output)
        if key in {
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("1", "0"),
            ("1", "4"),
            ("1", "5"),
            ("2", "0"),
            ("2", "4"),
            ("2", "5"),
            ("3", "0"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "3"),
            ("4", "5"),
            ("5", "1"),
            ("5", "2"),
            ("5", "4"),
        }:
            return 7
        elif key in {
            ("0", "5"),
            ("1", "2"),
            ("2", "1"),
            ("3", "5"),
            ("5", "0"),
            ("5", "3"),
        }:
            return 4
        elif key in {("0", "2"), ("1", "3"), ("2", "3"), ("3", "1"), ("3", "2")}:
            return 2
        elif key in {("4", "2")}:
            return 1
        return 6

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(tokens, attn_1_0_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_1_0_output):
        key = (num_attn_0_0_output, num_attn_1_0_output)
        if key in {
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (2, 30),
            (2, 31),
        }:
            return 9
        elif key in {(0, 0), (1, 0)}:
            return 11
        elif key in {(1, 1), (1, 2)}:
            return 12
        elif key in {(1, 3), (1, 4)}:
            return 7
        elif key in {(2, 0), (2, 1)}:
            return 10
        elif key in {(2, 2), (2, 3)}:
            return 14
        elif key in {(0, 1)}:
            return 1
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0, 7}:
            return k_num_mlp_1_0_output == 15
        elif q_num_mlp_1_0_output in {1}:
            return k_num_mlp_1_0_output == 12
        elif q_num_mlp_1_0_output in {9, 2, 5}:
            return k_num_mlp_1_0_output == 14
        elif q_num_mlp_1_0_output in {3, 13}:
            return k_num_mlp_1_0_output == 11
        elif q_num_mlp_1_0_output in {4}:
            return k_num_mlp_1_0_output == 7
        elif q_num_mlp_1_0_output in {11, 6, 15}:
            return k_num_mlp_1_0_output == 13
        elif q_num_mlp_1_0_output in {8, 10, 12, 14}:
            return k_num_mlp_1_0_output == 8

    attn_2_0_pattern = select_closest(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0, 9, 12}:
            return k_num_mlp_1_0_output == 9
        elif q_num_mlp_1_0_output in {1, 5}:
            return k_num_mlp_1_0_output == 15
        elif q_num_mlp_1_0_output in {2}:
            return k_num_mlp_1_0_output == 13
        elif q_num_mlp_1_0_output in {3, 4, 7}:
            return k_num_mlp_1_0_output == 12
        elif q_num_mlp_1_0_output in {8, 6}:
            return k_num_mlp_1_0_output == 14
        elif q_num_mlp_1_0_output in {10}:
            return k_num_mlp_1_0_output == 7
        elif q_num_mlp_1_0_output in {11, 13, 14, 15}:
            return k_num_mlp_1_0_output == 10

    attn_2_1_pattern = select_closest(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, position):
        if attn_1_1_output in {0, 11, 6}:
            return position == 7
        elif attn_1_1_output in {1, 4, 9}:
            return position == 6
        elif attn_1_1_output in {8, 2, 15, 7}:
            return position == 8
        elif attn_1_1_output in {3, 12, 14}:
            return position == 10
        elif attn_1_1_output in {5}:
            return position == 2
        elif attn_1_1_output in {10}:
            return position == 9
        elif attn_1_1_output in {13}:
            return position == 11

    num_attn_2_0_pattern = select(positions, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 12}:
            return position == 2
        elif mlp_0_0_output in {1, 2, 9, 15}:
            return position == 4
        elif mlp_0_0_output in {11, 3}:
            return position == 6
        elif mlp_0_0_output in {4, 5, 6, 8, 10}:
            return position == 5
        elif mlp_0_0_output in {14, 7}:
            return position == 1
        elif mlp_0_0_output in {13}:
            return position == 8

    num_attn_2_1_pattern = select(positions, mlp_0_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, num_mlp_1_0_output):
        key = (attn_2_0_output, num_mlp_1_0_output)
        if key in {
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 10),
            (0, 15),
            (1, 15),
            (2, 1),
            (2, 3),
            (2, 10),
            (2, 12),
            (2, 15),
            (3, 3),
            (3, 15),
            (4, 4),
            (4, 15),
            (5, 10),
            (5, 15),
            (6, 15),
            (7, 15),
            (8, 0),
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 6),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 15),
            (9, 0),
            (9, 1),
            (9, 3),
            (9, 4),
            (9, 6),
            (9, 10),
            (9, 12),
            (9, 15),
            (10, 0),
            (10, 1),
            (10, 3),
            (10, 4),
            (10, 6),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 15),
            (11, 3),
            (11, 15),
        }:
            return 13
        elif key in {
            (0, 11),
            (1, 4),
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 5),
            (3, 10),
            (3, 11),
            (3, 13),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 11),
            (4, 13),
            (5, 1),
            (5, 4),
            (5, 11),
            (6, 3),
            (6, 4),
            (6, 11),
            (7, 1),
            (7, 3),
            (7, 4),
            (8, 5),
            (9, 13),
            (10, 13),
            (11, 1),
            (11, 5),
            (11, 6),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 13),
            (12, 3),
            (12, 4),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 13),
        }:
            return 4
        elif key in {
            (1, 11),
            (7, 0),
            (7, 11),
            (11, 0),
            (11, 2),
            (11, 4),
            (11, 11),
            (12, 11),
            (13, 0),
            (13, 11),
            (13, 12),
            (13, 15),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 11),
            (14, 13),
            (15, 11),
        }:
            return 3
        elif key in {
            (0, 7),
            (0, 12),
            (2, 7),
            (4, 0),
            (4, 7),
            (4, 12),
            (5, 2),
            (5, 7),
            (6, 0),
            (6, 7),
            (7, 7),
            (8, 7),
            (9, 2),
            (10, 2),
            (10, 7),
            (11, 7),
            (12, 0),
            (12, 7),
            (14, 7),
            (15, 7),
        }:
            return 7
        elif key in {
            (0, 6),
            (1, 1),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 13),
            (2, 6),
            (3, 8),
            (3, 9),
            (5, 3),
            (5, 5),
            (5, 8),
            (5, 9),
            (6, 6),
            (7, 6),
            (9, 5),
            (10, 5),
        }:
            return 9
        elif key in {
            (0, 14),
            (1, 14),
            (2, 14),
            (3, 14),
            (4, 14),
            (5, 14),
            (6, 14),
            (7, 14),
            (8, 14),
            (9, 14),
            (10, 14),
            (11, 14),
            (12, 14),
            (13, 7),
            (13, 14),
            (14, 14),
            (15, 14),
        }:
            return 10
        elif key in {
            (1, 12),
            (5, 12),
            (6, 12),
            (7, 12),
            (11, 12),
            (12, 12),
            (12, 15),
            (14, 12),
            (14, 15),
            (15, 0),
            (15, 12),
            (15, 15),
        }:
            return 12
        elif key in {(1, 3), (3, 0), (3, 7), (3, 12), (5, 0)}:
            return 0
        elif key in {(1, 2), (8, 9), (9, 9), (10, 9)}:
            return 11
        elif key in {(1, 0), (1, 7), (2, 0)}:
            return 5
        elif key in {(2, 4), (2, 11), (9, 11)}:
            return 14
        elif key in {(9, 7)}:
            return 8
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_0_output, num_attn_2_1_output):
        key = (num_attn_0_0_output, num_attn_2_1_output)
        if key in {(0, 0), (1, 0)}:
            return 1
        elif key in {(0, 1)}:
            return 14
        elif key in {(0, 2)}:
            return 3
        elif key in {(1, 1)}:
            return 10
        return 6

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_2_1_outputs)
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


print(run(["<s>", "0", "2", "5", "2", "4", "3", "5", "4", "5", "4", "5", "0", "5"]))
