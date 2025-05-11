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
        "output\rasp\most_freq\longlen\modules_none\k8_len16_L4_H8_M4\s0\most_freq_weights.csv",
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
        elif q_position in {1, 2, 6}:
            return k_position == 4
        elif q_position in {8, 3, 4, 5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {9, 14}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 3
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 11

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 6}:
            return token == "4"
        elif position in {3, 4, 5, 12, 14}:
            return token == "5"
        elif position in {7}:
            return token == ""
        elif position in {8, 9, 10, 11, 13, 15}:
            return token == "2"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 5, 14}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {8, 3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {9, 6}:
            return k_position == 3
        elif q_position in {12, 7}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 8

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 4}:
            return k_position == 2
        elif q_position in {11, 2, 10, 13}:
            return k_position == 5
        elif q_position in {3, 6}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {14, 15}:
            return k_position == 12

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "4"
        elif position in {2, 3}:
            return token == "3"
        elif position in {4, 5, 6}:
            return token == ""
        elif position in {8, 7}:
            return token == "5"
        elif position in {9, 10, 11, 12, 13, 14, 15}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 2, 10}:
            return k_position == 2
        elif q_position in {1, 9}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 12

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2, 3, 4, 5}:
            return token == "3"
        elif position in {6}:
            return token == ""
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 5
        elif q_position in {8, 1, 15}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {9, 3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 15
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {11, 13}:
            return k_position == 10
        elif q_position in {12, 14}:
            return k_position == 11

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 0),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("1", 0),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("2", 0),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("3", 0),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("4", 0),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("5", 3),
            ("<s>", 0),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 14
        elif key in {
            ("0", 6),
            ("1", 6),
            ("2", 6),
            ("3", 6),
            ("4", 6),
            ("5", 6),
            ("<s>", 6),
        }:
            return 9
        elif key in {
            ("0", 1),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("<s>", 1),
        }:
            return 3
        elif key in {
            ("0", 2),
            ("1", 2),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("5", 2),
            ("<s>", 2),
        }:
            return 15
        elif key in {("5", 0), ("5", 4), ("5", 5)}:
            return 12
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "<s>"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "<s>"),
            (13, "1"),
            (13, "2"),
            (13, "3"),
            (13, "4"),
            (13, "<s>"),
        }:
            return 2
        elif key in {
            (0, "0"),
            (1, "0"),
            (1, "3"),
            (2, "0"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (4, "0"),
            (4, "5"),
            (5, "0"),
            (6, "0"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (8, "0"),
            (9, "0"),
            (10, "0"),
            (11, "0"),
            (13, "0"),
            (14, "0"),
            (15, "0"),
        }:
            return 14
        elif key in {(6, "2"), (6, "<s>")}:
            return 6
        elif key in {
            (1, "5"),
            (2, "5"),
            (5, "1"),
            (5, "4"),
            (5, "5"),
            (6, "1"),
            (6, "4"),
            (6, "5"),
        }:
            return 7
        elif key in {(1, "1"), (1, "2"), (1, "4"), (1, "<s>")}:
            return 4
        return 12

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 4

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {
            (2, 1),
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
        }:
            return 2
        elif key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
        }:
            return 4
        return 12

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"4"}:
            return token == "4"
        elif attn_0_3_output in {"5"}:
            return token == "5"
        elif attn_0_3_output in {"<s>"}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, attn_0_3_output):
        if position in {0, 5, 6, 7, 9, 10, 11, 13, 15}:
            return attn_0_3_output == "5"
        elif position in {1, 2, 3, 4}:
            return attn_0_3_output == "2"
        elif position in {8}:
            return attn_0_3_output == "4"
        elif position in {12, 14}:
            return attn_0_3_output == "0"

    attn_1_1_pattern = select_closest(attn_0_3_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, mlp_0_0_output):
        if position in {0, 1, 2, 4, 13}:
            return mlp_0_0_output == 3
        elif position in {10, 3, 5}:
            return mlp_0_0_output == 15
        elif position in {9, 11, 6}:
            return mlp_0_0_output == 9
        elif position in {7}:
            return mlp_0_0_output == 11
        elif position in {8}:
            return mlp_0_0_output == 2
        elif position in {12}:
            return mlp_0_0_output == 5
        elif position in {14, 15}:
            return mlp_0_0_output == 14

    attn_1_2_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_0_output, num_mlp_0_0_output):
        if mlp_0_0_output in {0}:
            return num_mlp_0_0_output == 14
        elif mlp_0_0_output in {1, 14}:
            return num_mlp_0_0_output == 13
        elif mlp_0_0_output in {2}:
            return num_mlp_0_0_output == 2
        elif mlp_0_0_output in {3, 15}:
            return num_mlp_0_0_output == 1
        elif mlp_0_0_output in {10, 4, 12}:
            return num_mlp_0_0_output == 3
        elif mlp_0_0_output in {9, 5}:
            return num_mlp_0_0_output == 5
        elif mlp_0_0_output in {8, 11, 6, 7}:
            return num_mlp_0_0_output == 4
        elif mlp_0_0_output in {13}:
            return num_mlp_0_0_output == 9

    attn_1_3_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_0_outputs, predicate_1_3
    )
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 7, 8, 9, 10, 11, 12, 14, 15}:
            return token == "5"
        elif position in {1, 2}:
            return token == "2"
        elif position in {3, 4, 5, 6, 13}:
            return token == "0"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 2, 3, 4, 12}:
            return token == ""
        elif position in {1}:
            return token == "3"
        elif position in {5, 6, 7, 9, 11, 13}:
            return token == "2"
        elif position in {8, 10}:
            return token == "5"
        elif position in {14, 15}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0}:
            return token == "<pad>"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == ""
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return token == "1"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, attn_0_1_output):
        if num_mlp_0_1_output in {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15}:
            return attn_0_1_output == "1"
        elif num_mlp_0_1_output in {8}:
            return attn_0_1_output == "0"
        elif num_mlp_0_1_output in {9, 13}:
            return attn_0_1_output == "5"

    num_attn_1_3_pattern = select(
        attn_0_1_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("1", 3),
            ("1", 8),
            ("1", 10),
            ("1", 12),
            ("1", 13),
            ("2", 3),
            ("2", 8),
            ("2", 10),
            ("2", 12),
            ("2", 13),
            ("3", 3),
            ("3", 8),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 15),
            ("4", 3),
            ("4", 8),
            ("4", 10),
            ("4", 12),
            ("4", 13),
            ("5", 3),
            ("5", 8),
            ("5", 10),
            ("5", 12),
            ("<s>", 3),
            ("<s>", 8),
            ("<s>", 10),
            ("<s>", 12),
        }:
            return 4
        elif key in {
            ("0", 0),
            ("0", 2),
            ("1", 2),
            ("2", 0),
            ("2", 2),
            ("3", 0),
            ("3", 2),
            ("4", 0),
            ("4", 2),
            ("5", 0),
            ("5", 2),
            ("<s>", 0),
            ("<s>", 2),
        }:
            return 11
        elif key in {
            ("0", 4),
            ("1", 4),
            ("2", 4),
            ("3", 4),
            ("4", 4),
            ("5", 4),
            ("<s>", 4),
        }:
            return 12
        elif key in {
            ("0", 1),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("<s>", 1),
        }:
            return 14
        elif key in {("0", 3), ("0", 8), ("0", 12)}:
            return 0
        elif key in {("1", 5), ("3", 5), ("3", 14)}:
            return 10
        elif key in {("1", 0)}:
            return 7
        return 8

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_0_output, attn_1_0_output):
        key = (mlp_0_0_output, attn_1_0_output)
        if key in {
            (4, "0"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (5, "5"),
            (6, "5"),
            (12, "2"),
            (12, "4"),
            (14, "5"),
        }:
            return 1
        elif key in {
            (0, "0"),
            (0, "2"),
            (1, "0"),
            (1, "2"),
            (2, "0"),
            (2, "2"),
            (3, "0"),
            (3, "2"),
            (11, "0"),
            (11, "2"),
            (13, "2"),
            (15, "0"),
            (15, "2"),
            (15, "4"),
        }:
            return 12
        elif key in {
            (0, "1"),
            (0, "3"),
            (0, "4"),
            (0, "<s>"),
            (4, "1"),
            (4, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "3"),
            (7, "4"),
        }:
            return 10
        elif key in {(2, "5"), (10, "5")}:
            return 6
        elif key in {(1, "5"), (7, "2"), (7, "5"), (9, "5"), (13, "5")}:
            return 7
        elif key in {
            (0, "5"),
            (8, "5"),
            (11, "5"),
            (12, "0"),
            (12, "1"),
            (12, "3"),
            (12, "5"),
            (12, "<s>"),
        }:
            return 8
        elif key in {(3, "1"), (3, "<s>"), (15, "5")}:
            return 5
        elif key in {(1, "4"), (2, "4"), (3, "4"), (11, "4")}:
            return 9
        elif key in {(15, "1"), (15, "3"), (15, "<s>")}:
            return 14
        elif key in {(3, "3"), (3, "5")}:
            return 4
        return 0

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_0_output):
        key = (num_attn_1_3_output, num_attn_0_0_output)
        if key in {(0, 0), (1, 0)}:
            return 6
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_1_2_output):
        key = (num_attn_0_3_output, num_attn_1_2_output)
        if key in {(0, 0)}:
            return 5
        return 4

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, mlp_0_1_output):
        if position in {0}:
            return mlp_0_1_output == 10
        elif position in {1, 2, 10}:
            return mlp_0_1_output == 14
        elif position in {3}:
            return mlp_0_1_output == 8
        elif position in {4}:
            return mlp_0_1_output == 15
        elif position in {9, 5}:
            return mlp_0_1_output == 2
        elif position in {6}:
            return mlp_0_1_output == 5
        elif position in {7}:
            return mlp_0_1_output == 1
        elif position in {8}:
            return mlp_0_1_output == 3
        elif position in {11, 13, 14}:
            return mlp_0_1_output == 12
        elif position in {12}:
            return mlp_0_1_output == 6
        elif position in {15}:
            return mlp_0_1_output == 11

    attn_2_0_pattern = select_closest(mlp_0_1_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, mlp_1_0_output):
        if token in {"0"}:
            return mlp_1_0_output == 4
        elif token in {"1", "2", "3", "<s>"}:
            return mlp_1_0_output == 8
        elif token in {"4"}:
            return mlp_1_0_output == 12
        elif token in {"5"}:
            return mlp_1_0_output == 14

    attn_2_1_pattern = select_closest(mlp_1_0_outputs, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, mlp_1_0_output):
        if position in {0}:
            return mlp_1_0_output == 11
        elif position in {1, 4}:
            return mlp_1_0_output == 14
        elif position in {2, 3}:
            return mlp_1_0_output == 4
        elif position in {5, 6}:
            return mlp_1_0_output == 6
        elif position in {12, 14, 7}:
            return mlp_1_0_output == 8
        elif position in {8}:
            return mlp_1_0_output == 10
        elif position in {9, 13}:
            return mlp_1_0_output == 12
        elif position in {10}:
            return mlp_1_0_output == 0
        elif position in {11}:
            return mlp_1_0_output == 7
        elif position in {15}:
            return mlp_1_0_output == 2

    attn_2_2_pattern = select_closest(mlp_1_0_outputs, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, mlp_1_0_output):
        if token in {"0"}:
            return mlp_1_0_output == 14
        elif token in {"1", "2", "4", "<s>"}:
            return mlp_1_0_output == 7
        elif token in {"5", "3"}:
            return mlp_1_0_output == 8

    attn_2_3_pattern = select_closest(mlp_1_0_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_0_2_output):
        if position in {0, 13}:
            return attn_0_2_output == "4"
        elif position in {8, 1, 9}:
            return attn_0_2_output == "5"
        elif position in {2, 12}:
            return attn_0_2_output == ""
        elif position in {3}:
            return attn_0_2_output == "<pad>"
        elif position in {4}:
            return attn_0_2_output == "<s>"
        elif position in {5, 6}:
            return attn_0_2_output == "3"
        elif position in {7}:
            return attn_0_2_output == "0"
        elif position in {10, 11}:
            return attn_0_2_output == "2"
        elif position in {14, 15}:
            return attn_0_2_output == "1"

    num_attn_2_0_pattern = select(attn_0_2_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_1_0_output, token):
        if mlp_1_0_output in {0}:
            return token == "<s>"
        elif mlp_1_0_output in {1, 5}:
            return token == "<pad>"
        elif mlp_1_0_output in {2, 4, 7, 14, 15}:
            return token == "0"
        elif mlp_1_0_output in {3, 6, 8, 9, 10, 11, 12}:
            return token == "2"
        elif mlp_1_0_output in {13}:
            return token == "1"

    num_attn_2_1_pattern = select(tokens, mlp_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, token):
        if mlp_1_0_output in {0, 4, 13}:
            return token == "0"
        elif mlp_1_0_output in {1, 7}:
            return token == "<s>"
        elif mlp_1_0_output in {2, 5, 6, 14}:
            return token == "2"
        elif mlp_1_0_output in {3, 15}:
            return token == "5"
        elif mlp_1_0_output in {8, 10}:
            return token == ""
        elif mlp_1_0_output in {9, 12}:
            return token == "4"
        elif mlp_1_0_output in {11}:
            return token == "1"

    num_attn_2_2_pattern = select(tokens, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 11, 15}:
            return token == "3"
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 12}:
            return token == "5"
        elif position in {13, 14}:
            return token == ""

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_var0_embeddings)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("0", 0),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("2", 0),
            ("2", 2),
            ("2", 4),
            ("3", 0),
            ("4", 4),
            ("5", 0),
        }:
            return 5
        elif key in {("5", 6), ("5", 12), ("5", 13)}:
            return 13
        elif key in {
            ("0", 5),
            ("0", 6),
            ("0", 11),
            ("1", 5),
            ("1", 6),
            ("5", 2),
            ("<s>", 5),
        }:
            return 6
        elif key in {
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 10),
            ("2", 3),
            ("4", 0),
            ("4", 3),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 11),
        }:
            return 14
        elif key in {
            ("2", 5),
            ("3", 2),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("4", 2),
            ("4", 5),
            ("5", 1),
        }:
            return 7
        elif key in {
            ("1", 11),
            ("2", 11),
            ("4", 11),
            ("5", 3),
            ("5", 4),
            ("5", 11),
            ("5", 15),
        }:
            return 1
        elif key in {("2", 6), ("4", 6), ("<s>", 6)}:
            return 2
        elif key in {("5", 5)}:
            return 9
        elif key in {("1", 1), ("2", 1), ("4", 1)}:
            return 3
        elif key in {("0", 1)}:
            return 0
        elif key in {("3", 1)}:
            return 8
        return 4

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {("2", 0), ("2", 1)}:
            return 0
        elif key in {("2", 2), ("4", 4), ("4", 5), ("4", 6), ("4", 15), ("5", 2)}:
            return 2
        elif key in {
            ("1", 6),
            ("2", 3),
            ("2", 6),
            ("3", 3),
            ("3", 6),
            ("5", 3),
            ("5", 6),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
        }:
            return 13
        elif key in {("0", 0), ("1", 0), ("3", 2), ("<s>", 2)}:
            return 10
        elif key in {("0", 1), ("1", 1), ("3", 1), ("<s>", 1)}:
            return 11
        elif key in {("3", 0), ("4", 0), ("4", 3), ("5", 0), ("<s>", 0)}:
            return 14
        elif key in {("0", 6)}:
            return 5
        elif key in {("4", 1), ("4", 2)}:
            return 9
        elif key in {("5", 1)}:
            return 12
        return 3

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_1_3_output):
        key = (num_attn_1_2_output, num_attn_1_3_output)
        if key in {
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
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
        }:
            return 7
        return 0

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        return 11

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(q_token, k_token):
        if q_token in {"0", "2", "4", "3", "1", "5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_3_0_pattern = select_closest(tokens, tokens, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, attn_2_3_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(token, mlp_0_0_output):
        if token in {"0", "2", "4", "3", "5"}:
            return mlp_0_0_output == 9
        elif token in {"1"}:
            return mlp_0_0_output == 7
        elif token in {"<s>"}:
            return mlp_0_0_output == 6

    attn_3_1_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, num_mlp_2_0_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(token, position):
        if token in {"0", "2", "3", "1", "5"}:
            return position == 1
        elif token in {"4"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 3

    attn_3_2_pattern = select_closest(positions, tokens, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, mlp_2_0_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(token, mlp_0_0_output):
        if token in {"0", "2", "4", "3", "<s>", "1", "5"}:
            return mlp_0_0_output == 15

    attn_3_3_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_2_1_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 6, 8, 9, 10, 12, 13, 14}:
            return token == "4"
        elif mlp_0_0_output in {1}:
            return token == "<pad>"
        elif mlp_0_0_output in {2, 7}:
            return token == ""
        elif mlp_0_0_output in {3, 4, 5}:
            return token == "0"
        elif mlp_0_0_output in {11, 15}:
            return token == "<s>"

    num_attn_3_0_pattern = select(tokens, mlp_0_0_outputs, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, num_attn_1_0_outputs)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(position, token):
        if position in {0, 1, 15}:
            return token == "5"
        elif position in {2}:
            return token == ""
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {5, 6, 14}:
            return token == "0"
        elif position in {7, 8, 9, 10, 11, 12}:
            return token == "4"
        elif position in {13}:
            return token == "1"

    num_attn_3_1_pattern = select(tokens, positions, num_predicate_3_1)
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_attn_2_3_outputs)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(mlp_1_0_output, attn_1_3_output):
        if mlp_1_0_output in {0, 6}:
            return attn_1_3_output == "0"
        elif mlp_1_0_output in {1, 3, 13, 14}:
            return attn_1_3_output == "<s>"
        elif mlp_1_0_output in {2, 4, 5, 8, 9, 10, 12}:
            return attn_1_3_output == "2"
        elif mlp_1_0_output in {7}:
            return attn_1_3_output == "4"
        elif mlp_1_0_output in {11, 15}:
            return attn_1_3_output == "1"

    num_attn_3_2_pattern = select(attn_1_3_outputs, mlp_1_0_outputs, num_predicate_3_2)
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, num_attn_1_1_outputs)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(position, attn_0_1_output):
        if position in {0}:
            return attn_0_1_output == "5"
        elif position in {1}:
            return attn_0_1_output == "2"
        elif position in {2, 4, 8, 9, 10, 14}:
            return attn_0_1_output == "<s>"
        elif position in {3, 7, 12, 13, 15}:
            return attn_0_1_output == ""
        elif position in {5, 6}:
            return attn_0_1_output == "1"
        elif position in {11}:
            return attn_0_1_output == "<pad>"

    num_attn_3_3_pattern = select(attn_0_1_outputs, positions, num_predicate_3_3)
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, num_attn_1_1_outputs)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(attn_2_2_output, mlp_1_0_output):
        key = (attn_2_2_output, mlp_1_0_output)
        if key in {
            ("1", 0),
            ("1", 11),
            ("1", 14),
            ("2", 0),
            ("2", 1),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 11),
            ("2", 12),
            ("2", 15),
            ("3", 0),
            ("3", 11),
            ("3", 14),
            ("4", 0),
            ("4", 11),
            ("4", 14),
            ("5", 0),
            ("5", 11),
            ("<s>", 0),
            ("<s>", 14),
        }:
            return 3
        elif key in {
            ("0", 14),
            ("1", 2),
            ("1", 6),
            ("1", 12),
            ("1", 13),
            ("1", 15),
            ("3", 2),
            ("3", 6),
            ("3", 12),
            ("3", 15),
            ("4", 2),
            ("4", 6),
            ("4", 12),
            ("4", 15),
            ("5", 6),
            ("5", 12),
            ("5", 13),
            ("5", 15),
            ("<s>", 12),
            ("<s>", 15),
        }:
            return 2
        elif key in {
            ("0", 1),
            ("0", 5),
            ("0", 7),
            ("0", 11),
            ("1", 1),
            ("1", 5),
            ("1", 7),
            ("1", 9),
            ("3", 1),
            ("3", 5),
            ("3", 7),
            ("3", 9),
            ("3", 13),
            ("4", 1),
            ("4", 5),
            ("4", 7),
            ("4", 9),
            ("4", 13),
            ("5", 1),
            ("5", 5),
            ("5", 7),
            ("5", 9),
            ("<s>", 1),
            ("<s>", 9),
            ("<s>", 13),
        }:
            return 5
        elif key in {
            ("0", 0),
            ("0", 2),
            ("0", 6),
            ("0", 9),
            ("0", 12),
            ("0", 13),
            ("0", 15),
            ("2", 2),
            ("2", 3),
            ("2", 8),
            ("2", 9),
            ("2", 10),
        }:
            return 13
        elif key in {("0", 8), ("0", 10)}:
            return 10
        elif key in {("<s>", 8), ("<s>", 10)}:
            return 0
        return 9

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, mlp_1_0_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(attn_0_1_output, attn_1_2_output):
        key = (attn_0_1_output, attn_1_2_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "5"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "3"),
            ("3", "<s>"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 3
        elif key in {("3", "5"), ("4", "3")}:
            return 11
        elif key in {
            ("0", "4"),
            ("2", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "5"),
            ("5", "4"),
        }:
            return 2
        elif key in {("3", "2"), ("4", "<s>"), ("<s>", "4")}:
            return 5
        return 0

    mlp_3_1_outputs = [
        mlp_3_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_2_outputs)
    ]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_var0_embedding, num_attn_3_2_output):
        key = (num_var0_embedding, num_attn_3_2_output)
        if key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
            (16, 0),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (20, 0),
            (20, 1),
            (21, 0),
            (21, 1),
            (22, 0),
            (22, 1),
            (23, 0),
            (23, 1),
            (23, 2),
            (24, 0),
            (24, 1),
            (24, 2),
            (25, 0),
            (25, 1),
            (25, 2),
            (26, 0),
            (26, 1),
            (26, 2),
            (27, 0),
            (27, 1),
            (27, 2),
            (28, 0),
            (28, 1),
            (28, 2),
            (29, 0),
            (29, 1),
            (29, 2),
            (30, 0),
            (30, 1),
            (30, 2),
            (31, 0),
            (31, 1),
            (31, 2),
            (32, 0),
            (32, 1),
            (32, 2),
            (33, 0),
            (33, 1),
            (33, 2),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
        }:
            return 5
        return 2

    num_mlp_3_0_outputs = [
        num_mlp_3_0(k0, k1) for k0, k1 in zip(num_var0_embeddings, num_attn_3_2_outputs)
    ]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_0_2_output, num_attn_1_2_output):
        key = (num_attn_0_2_output, num_attn_1_2_output)
        if key in {
            (0, 0),
            (0, 1),
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
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (1, 4),
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
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (3, 60),
            (3, 61),
            (3, 62),
            (3, 63),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (4, 60),
            (4, 61),
            (4, 62),
            (4, 63),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (5, 60),
            (5, 61),
            (5, 62),
            (5, 63),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (6, 60),
            (6, 61),
            (6, 62),
            (6, 63),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (7, 60),
            (7, 61),
            (7, 62),
            (7, 63),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (8, 60),
            (8, 61),
            (8, 62),
            (8, 63),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (9, 60),
            (9, 61),
            (9, 62),
            (9, 63),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (10, 60),
            (10, 61),
            (10, 62),
            (10, 63),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (11, 60),
            (11, 61),
            (11, 62),
            (11, 63),
            (12, 58),
            (12, 59),
            (12, 60),
            (12, 61),
            (12, 62),
            (12, 63),
            (13, 63),
        }:
            return 8
        return 4

    num_mlp_3_1_outputs = [
        num_mlp_3_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_3_1_output_scores = classifier_weights.loc[
        [("num_mlp_3_1_outputs", str(v)) for v in num_mlp_3_1_outputs]
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
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                attn_3_0_output_scores,
                attn_3_1_output_scores,
                attn_3_2_output_scores,
                attn_3_3_output_scores,
                mlp_3_0_output_scores,
                mlp_3_1_output_scores,
                num_mlp_3_0_output_scores,
                num_mlp_3_1_output_scores,
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
                num_attn_3_0_output_scores,
                num_attn_3_1_output_scores,
                num_attn_3_2_output_scores,
                num_attn_3_3_output_scores,
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
