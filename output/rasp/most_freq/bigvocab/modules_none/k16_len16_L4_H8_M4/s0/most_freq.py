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
        "output\rasp\most_freq\bigvocab\modules_none\k16_len16_L4_H8_M4\s0\most_freq_weights.csv",
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
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 12
        elif q_position in {11, 12, 13, 15}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 13

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1, 6, 14}:
            return k_position == 4
        elif q_position in {2, 5}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {10, 4}:
            return k_position == 7
        elif q_position in {8, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {12, 13}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 15

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 5}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 9, 13, 14}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11, 12}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 7

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3, 5}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {12, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 2

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 15}:
            return token == "0"
        elif position in {2}:
            return token == "5"
        elif position in {3}:
            return token == "<pad>"
        elif position in {4, 5, 6, 7}:
            return token == "8"
        elif position in {8, 9, 10, 11, 12, 13, 14}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 11
        elif q_position in {1, 13}:
            return k_position == 1
        elif q_position in {2, 14, 15}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {12, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {11}:
            return k_position == 13

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14}:
            return token == "<s>"
        elif position in {12}:
            return token == "13"
        elif position in {15}:
            return token == "11"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 12, 14}:
            return token == "1"
        elif position in {1}:
            return token == "10"
        elif position in {9, 2, 3, 4}:
            return token == "<s>"
        elif position in {8, 5, 6, 7}:
            return token == "13"
        elif position in {10}:
            return token == "5"
        elif position in {11}:
            return token == "6"
        elif position in {13, 15}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("10", 5),
            ("10", 6),
            ("10", 7),
            ("10", 8),
            ("10", 9),
            ("10", 10),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 8),
            ("11", 9),
            ("11", 10),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 8),
            ("12", 9),
            ("12", 10),
            ("13", 5),
            ("13", 6),
            ("13", 7),
            ("13", 8),
            ("13", 9),
            ("13", 10),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("6", 5),
            ("6", 6),
            ("6", 7),
            ("6", 8),
            ("6", 9),
            ("6", 10),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("7", 10),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("8", 8),
            ("8", 9),
            ("8", 10),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 8),
            ("9", 9),
            ("9", 10),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
        }:
            return 4
        elif key in {
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("10", 1),
            ("10", 2),
            ("10", 3),
            ("11", 1),
            ("11", 2),
            ("11", 3),
            ("12", 1),
            ("12", 2),
            ("12", 3),
            ("13", 1),
            ("13", 2),
            ("13", 3),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("5", 1),
            ("5", 2),
            ("5", 3),
            ("6", 2),
            ("6", 3),
            ("7", 1),
            ("7", 2),
            ("7", 3),
            ("8", 1),
            ("8", 2),
            ("8", 3),
            ("9", 2),
            ("9", 3),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 6
        elif key in {
            ("0", 0),
            ("1", 0),
            ("10", 0),
            ("11", 0),
            ("12", 0),
            ("2", 0),
            ("5", 0),
            ("6", 0),
            ("7", 0),
            ("8", 0),
            ("9", 0),
            ("<s>", 0),
        }:
            return 11
        elif key in {("13", 0), ("4", 0)}:
            return 7
        elif key in {("6", 1), ("9", 1)}:
            return 12
        elif key in {("3", 0)}:
            return 0
        return 10

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, position):
        key = (attn_0_0_output, position)
        if key in {
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 14),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 14),
            ("1", 15),
            ("10", 5),
            ("10", 6),
            ("10", 7),
            ("10", 14),
            ("10", 15),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 14),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 14),
            ("13", 5),
            ("13", 6),
            ("13", 7),
            ("13", 14),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 14),
            ("2", 15),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 14),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 14),
            ("4", 15),
            ("5", 5),
            ("5", 7),
            ("5", 14),
            ("6", 5),
            ("6", 6),
            ("6", 7),
            ("6", 14),
            ("6", 15),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 14),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("8", 14),
            ("8", 15),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 14),
            ("9", 15),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 14),
        }:
            return 11
        elif key in {
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("11", 2),
            ("11", 3),
            ("11", 4),
            ("12", 2),
            ("13", 2),
            ("13", 3),
            ("13", 4),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("5", 2),
            ("6", 2),
            ("7", 2),
            ("7", 3),
            ("7", 4),
            ("8", 2),
            ("8", 3),
            ("9", 2),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 3
        elif key in {
            ("0", 0),
            ("0", 1),
            ("1", 0),
            ("1", 1),
            ("10", 0),
            ("10", 1),
            ("11", 0),
            ("11", 1),
            ("12", 0),
            ("12", 1),
            ("2", 0),
            ("2", 1),
            ("3", 0),
            ("3", 1),
            ("5", 0),
            ("5", 1),
            ("9", 1),
            ("<s>", 0),
            ("<s>", 1),
        }:
            return 9
        elif key in {
            ("13", 0),
            ("13", 1),
            ("4", 0),
            ("4", 1),
            ("6", 0),
            ("6", 1),
            ("7", 0),
            ("7", 1),
            ("8", 0),
            ("8", 1),
        }:
            return 6
        elif key in {("3", 2), ("3", 3), ("3", 4), ("9", 3), ("9", 4)}:
            return 10
        elif key in {("6", 3), ("6", 4), ("8", 4)}:
            return 13
        elif key in {("5", 3), ("5", 4)}:
            return 2
        elif key in {("12", 3), ("12", 4)}:
            return 5
        elif key in {("9", 0)}:
            return 15
        return 0

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 15

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 15

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 8, 11}:
            return k_position == 5
        elif q_position in {1, 2, 6, 7}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 13

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {13, 5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9, 11, 14}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {15}:
            return k_position == 15

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 1
        elif mlp_0_0_output in {1, 10, 12, 9}:
            return position == 3
        elif mlp_0_0_output in {2, 3}:
            return position == 6
        elif mlp_0_0_output in {8, 4}:
            return position == 5
        elif mlp_0_0_output in {13, 11, 5}:
            return position == 2
        elif mlp_0_0_output in {6}:
            return position == 10
        elif mlp_0_0_output in {7}:
            return position == 14
        elif mlp_0_0_output in {14}:
            return position == 12
        elif mlp_0_0_output in {15}:
            return position == 8

    attn_1_2_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {13, 5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {8, 7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10, 12}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 12

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1}:
            return token == "7"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 11}:
            return token == "8"
        elif position in {12, 13, 14, 15}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
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
            return k_token == "<s>"
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
            return k_token == "3"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0}:
            return token == "10"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "<s>"
        elif position in {3}:
            return token == "9"
        elif position in {4, 5, 6, 7, 8, 9}:
            return token == "11"
        elif position in {10}:
            return token == "8"
        elif position in {11, 12, 13, 14, 15}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_var0_embeddings)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"10"}:
            return k_token == "8"
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
        elif q_token in {"9", "6", "8"}:
            return k_token == "<s>"
        elif q_token in {"7"}:
            return k_token == "7"
        elif q_token in {"<s>"}:
            return k_token == "10"

    num_attn_1_3_pattern = select(tokens, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_var0_embeddings)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, position):
        key = (attn_1_2_output, position)
        if key in {
            ("0", 0),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 13),
            ("1", 2),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("11", 2),
            ("11", 3),
            ("11", 4),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 8),
            ("11", 9),
            ("11", 13),
            ("12", 2),
            ("13", 4),
            ("2", 0),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 8),
            ("4", 12),
            ("4", 13),
            ("5", 0),
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("6", 2),
            ("7", 0),
            ("7", 2),
            ("7", 3),
            ("7", 4),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("7", 10),
            ("7", 13),
            ("8", 2),
            ("8", 3),
            ("9", 0),
            ("9", 2),
            ("9", 3),
            ("9", 4),
            ("9", 5),
            ("9", 7),
            ("9", 8),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 2
        elif key in {
            ("1", 10),
            ("12", 10),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 9),
            ("4", 10),
            ("6", 6),
            ("6", 7),
            ("6", 8),
            ("6", 9),
            ("6", 10),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("8", 8),
            ("8", 9),
            ("8", 10),
            ("<s>", 10),
        }:
            return 9
        elif key in {
            ("0", 2),
            ("0", 3),
            ("1", 0),
            ("1", 3),
            ("10", 0),
            ("11", 10),
            ("12", 0),
            ("13", 0),
            ("13", 2),
            ("13", 3),
            ("13", 10),
            ("6", 0),
            ("6", 3),
            ("8", 0),
            ("<s>", 0),
        }:
            return 4
        elif key in {
            ("1", 1),
            ("10", 1),
            ("11", 0),
            ("11", 1),
            ("3", 0),
            ("3", 1),
            ("6", 1),
            ("7", 1),
            ("<s>", 1),
        }:
            return 0
        elif key in {("0", 1), ("12", 1), ("13", 1), ("2", 1), ("5", 1), ("8", 1)}:
            return 5
        elif key in {("4", 0), ("4", 1)}:
            return 10
        elif key in {("9", 1)}:
            return 15
        return 14

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("10", 0),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 5),
            ("10", 6),
            ("10", 7),
            ("10", 8),
            ("12", 2),
            ("12", 3),
            ("12", 7),
            ("12", 8),
            ("13", 0),
            ("13", 1),
            ("13", 2),
            ("13", 3),
            ("13", 4),
            ("13", 5),
            ("13", 6),
            ("13", 7),
            ("13", 8),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("5", 0),
            ("5", 1),
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("6", 2),
            ("6", 3),
            ("6", 4),
            ("8", 2),
            ("8", 3),
            ("9", 2),
            ("9", 3),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 14
        elif key in {
            ("12", 0),
            ("12", 4),
            ("12", 5),
            ("12", 6),
            ("2", 0),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("7", 0),
            ("7", 3),
            ("7", 4),
            ("7", 5),
            ("7", 6),
            ("7", 8),
            ("8", 0),
            ("8", 4),
            ("9", 0),
            ("9", 4),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 8),
            ("<s>", 0),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
        }:
            return 11
        elif key in {
            ("11", 0),
            ("11", 2),
            ("11", 3),
            ("11", 4),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 8),
        }:
            return 4
        elif key in {("6", 0), ("6", 5), ("6", 6), ("6", 7), ("6", 8)}:
            return 9
        elif key in {("8", 5), ("8", 6), ("8", 7), ("8", 8)}:
            return 8
        elif key in {("10", 1), ("12", 1), ("7", 1), ("9", 1)}:
            return 10
        elif key in {("2", 1), ("2", 2), ("2", 3), ("6", 1)}:
            return 13
        elif key in {("7", 7)}:
            return 1
        elif key in {("8", 1)}:
            return 2
        elif key in {("7", 2)}:
            return 3
        elif key in {("11", 1)}:
            return 5
        return 0

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
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
            (24, 0),
            (24, 1),
            (25, 0),
            (25, 1),
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
        }:
            return 13
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_var0_embedding):
        key = (num_attn_1_1_output, num_var0_embedding)
        return 8

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1) for k0, k1 in zip(num_attn_1_1_outputs, num_var0_embeddings)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1, 2, 13, 14, 15}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6, 7, 8, 9, 10, 11}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 2

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1, 11}:
            return k_position == 6
        elif q_position in {2, 4}:
            return k_position == 13
        elif q_position in {9, 3, 12}:
            return k_position == 14
        elif q_position in {10, 5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 3
        elif q_position in {14, 15}:
            return k_position == 2

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 11
        elif q_position in {1, 2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {9, 10, 11, 12}:
            return k_position == 5
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14, 15}:
            return k_position == 0

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 2, 3}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 14
        elif q_position in {5, 6, 7, 8, 9, 10, 11, 12}:
            return k_position == 0
        elif q_position in {13, 14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 15

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, num_mlp_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, token):
        if attn_1_2_output in {"0"}:
            return token == "0"
        elif attn_1_2_output in {"1"}:
            return token == "1"
        elif attn_1_2_output in {"10"}:
            return token == "10"
        elif attn_1_2_output in {"11"}:
            return token == "11"
        elif attn_1_2_output in {"12"}:
            return token == "12"
        elif attn_1_2_output in {"13"}:
            return token == "13"
        elif attn_1_2_output in {"2"}:
            return token == "2"
        elif attn_1_2_output in {"3"}:
            return token == "3"
        elif attn_1_2_output in {"4"}:
            return token == "4"
        elif attn_1_2_output in {"5"}:
            return token == "5"
        elif attn_1_2_output in {"6"}:
            return token == "6"
        elif attn_1_2_output in {"7"}:
            return token == "7"
        elif attn_1_2_output in {"8"}:
            return token == "8"
        elif attn_1_2_output in {"9"}:
            return token == "9"
        elif attn_1_2_output in {"<s>"}:
            return token == "<s>"

    num_attn_2_0_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, token):
        if attn_1_1_output in {"0", "13", "10", "12", "7"}:
            return token == "8"
        elif attn_1_1_output in {"1"}:
            return token == "1"
        elif attn_1_1_output in {"11"}:
            return token == "11"
        elif attn_1_1_output in {"2", "5", "6", "8"}:
            return token == "<pad>"
        elif attn_1_1_output in {"3"}:
            return token == "3"
        elif attn_1_1_output in {"4"}:
            return token == "4"
        elif attn_1_1_output in {"9"}:
            return token == "9"
        elif attn_1_1_output in {"<s>"}:
            return token == "13"

    num_attn_2_1_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_var0_embeddings)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 11, 5}:
            return token == "12"
        elif mlp_0_1_output in {2, 4, 8, 10, 12, 13}:
            return token == "<pad>"
        elif mlp_0_1_output in {3, 7}:
            return token == "<s>"
        elif mlp_0_1_output in {6}:
            return token == "8"
        elif mlp_0_1_output in {9, 15}:
            return token == "3"
        elif mlp_0_1_output in {14}:
            return token == "9"

    num_attn_2_2_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, token):
        if attn_1_1_output in {"0"}:
            return token == "0"
        elif attn_1_1_output in {"1"}:
            return token == "<pad>"
        elif attn_1_1_output in {"10"}:
            return token == "10"
        elif attn_1_1_output in {"6", "2", "4", "5", "11", "9"}:
            return token == "<s>"
        elif attn_1_1_output in {"12"}:
            return token == "12"
        elif attn_1_1_output in {"13"}:
            return token == "11"
        elif attn_1_1_output in {"3"}:
            return token == "3"
        elif attn_1_1_output in {"7"}:
            return token == "7"
        elif attn_1_1_output in {"8"}:
            return token == "8"
        elif attn_1_1_output in {"<s>"}:
            return token == "13"

    num_attn_2_3_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_var0_embeddings)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            ("0", 4),
            ("0", 5),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("11", 0),
            ("11", 3),
            ("11", 4),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 8),
            ("11", 9),
            ("11", 12),
            ("12", 3),
            ("12", 4),
            ("13", 1),
            ("13", 2),
            ("13", 3),
            ("13", 4),
            ("13", 5),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("5", 0),
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 14),
            ("7", 3),
            ("7", 4),
            ("8", 4),
            ("8", 5),
            ("8", 6),
            ("8", 7),
            ("9", 1),
            ("9", 2),
            ("9", 15),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
        }:
            return 2
        elif key in {
            ("0", 1),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("10", 1),
            ("10", 13),
            ("10", 14),
            ("10", 15),
            ("11", 1),
            ("11", 13),
            ("11", 14),
            ("11", 15),
            ("12", 1),
            ("12", 13),
            ("12", 14),
            ("12", 15),
            ("13", 13),
            ("13", 14),
            ("13", 15),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("3", 1),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("4", 13),
            ("4", 14),
            ("4", 15),
            ("5", 13),
            ("6", 13),
            ("7", 1),
            ("7", 13),
            ("7", 14),
            ("7", 15),
            ("8", 1),
            ("8", 13),
            ("8", 14),
            ("8", 15),
            ("9", 13),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
        }:
            return 13
        elif key in {
            ("0", 2),
            ("0", 3),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 5),
            ("11", 2),
            ("12", 2),
            ("2", 1),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 8),
            ("4", 9),
            ("7", 2),
            ("8", 2),
            ("8", 3),
            ("9", 0),
            ("9", 3),
            ("9", 4),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 8),
            ("9", 9),
            ("9", 11),
            ("9", 14),
            ("<s>", 1),
        }:
            return 6
        elif key in {
            ("10", 12),
            ("12", 10),
            ("12", 12),
            ("13", 10),
            ("4", 7),
            ("4", 10),
            ("4", 12),
            ("6", 1),
            ("6", 2),
            ("9", 10),
            ("9", 12),
        }:
            return 14
        elif key in {
            ("13", 12),
            ("2", 0),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("2", 10),
        }:
            return 11
        elif key in {("2", 2), ("2", 3), ("2", 4), ("2", 5), ("2", 12)}:
            return 1
        elif key in {("10", 0), ("10", 10), ("11", 10), ("4", 0), ("5", 1)}:
            return 8
        elif key in {("5", 15)}:
            return 7
        elif key in {("6", 15)}:
            return 15
        return 4

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            ("0", "10"),
            ("0", "5"),
            ("1", "10"),
            ("1", "5"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "11"),
            ("10", "12"),
            ("10", "13"),
            ("10", "2"),
            ("10", "3"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "<s>"),
            ("11", "10"),
            ("11", "5"),
            ("12", "10"),
            ("12", "5"),
            ("13", "10"),
            ("13", "5"),
            ("2", "10"),
            ("2", "5"),
            ("3", "10"),
            ("3", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "10"),
            ("5", "11"),
            ("5", "12"),
            ("5", "13"),
            ("5", "2"),
            ("5", "3"),
            ("5", "6"),
            ("5", "7"),
            ("5", "8"),
            ("5", "<s>"),
            ("6", "10"),
            ("6", "5"),
            ("7", "10"),
            ("7", "5"),
            ("8", "10"),
            ("8", "5"),
            ("<s>", "10"),
            ("<s>", "5"),
        }:
            return 15
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("11", "4"),
            ("12", "4"),
            ("13", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "11"),
            ("4", "12"),
            ("4", "13"),
            ("4", "2"),
            ("4", "3"),
            ("4", "6"),
            ("4", "7"),
            ("4", "8"),
            ("4", "<s>"),
            ("6", "4"),
            ("7", "4"),
            ("8", "4"),
            ("<s>", "4"),
        }:
            return 9
        elif key in {
            ("0", "9"),
            ("1", "9"),
            ("12", "9"),
            ("13", "9"),
            ("2", "9"),
            ("3", "9"),
            ("6", "9"),
            ("7", "9"),
            ("8", "9"),
            ("9", "0"),
            ("9", "1"),
            ("9", "11"),
            ("9", "12"),
            ("9", "13"),
            ("9", "2"),
            ("9", "3"),
            ("9", "6"),
            ("9", "7"),
            ("9", "8"),
            ("9", "9"),
            ("9", "<s>"),
            ("<s>", "9"),
        }:
            return 0
        elif key in {("10", "9"), ("5", "9"), ("9", "10"), ("9", "5")}:
            return 1
        elif key in {("10", "4"), ("4", "10"), ("4", "5"), ("5", "4")}:
            return 11
        elif key in {("11", "9"), ("4", "9"), ("9", "4")}:
            return 3
        return 13

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 1

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output):
        key = num_attn_2_3_output
        if key in {0}:
            return 5
        return 12

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_3_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(mlp_0_1_output, position):
        if mlp_0_1_output in {0}:
            return position == 0
        elif mlp_0_1_output in {1, 4, 5, 7, 10, 12, 13, 14}:
            return position == 1
        elif mlp_0_1_output in {8, 2}:
            return position == 5
        elif mlp_0_1_output in {3}:
            return position == 6
        elif mlp_0_1_output in {9, 6}:
            return position == 4
        elif mlp_0_1_output in {11}:
            return position == 11
        elif mlp_0_1_output in {15}:
            return position == 10

    attn_3_0_pattern = select_closest(positions, mlp_0_1_outputs, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, mlp_0_1_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(q_token, k_token):
        if q_token in {"6", "0", "7", "13", "10", "1", "2", "4", "5", "11", "9"}:
            return k_token == "3"
        elif q_token in {"8", "3", "12"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_3_1_pattern = select_closest(tokens, tokens, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, mlp_0_0_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(q_token, k_token):
        if q_token in {"6", "0", "7", "13", "1", "3", "12", "11", "8", "<s>"}:
            return k_token == "<s>"
        elif q_token in {"10"}:
            return k_token == "6"
        elif q_token in {"2"}:
            return k_token == "9"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "8"
        elif q_token in {"9"}:
            return k_token == "4"

    attn_3_2_pattern = select_closest(tokens, tokens, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, mlp_2_1_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 0
        elif q_position in {1, 12}:
            return k_position == 11
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {8, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9, 11}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 1
        elif q_position in {13, 15}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 13

    attn_3_3_pattern = select_closest(positions, positions, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_0_0_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(attn_1_1_output, token):
        if attn_1_1_output in {"0"}:
            return token == "0"
        elif attn_1_1_output in {"1", "6"}:
            return token == "5"
        elif attn_1_1_output in {"10", "4", "5", "11", "8"}:
            return token == "11"
        elif attn_1_1_output in {"7", "2", "9", "12"}:
            return token == "<pad>"
        elif attn_1_1_output in {"13", "3"}:
            return token == "10"
        elif attn_1_1_output in {"<s>"}:
            return token == "3"

    num_attn_3_0_pattern = select(tokens, attn_1_1_outputs, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, num_attn_1_3_outputs)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(attn_1_1_output, num_mlp_0_0_output):
        if attn_1_1_output in {
            "6",
            "0",
            "7",
            "13",
            "10",
            "2",
            "3",
            "4",
            "5",
            "12",
            "11",
            "9",
            "8",
            "<s>",
        }:
            return num_mlp_0_0_output == 12
        elif attn_1_1_output in {"1"}:
            return num_mlp_0_0_output == 14

    num_attn_3_1_pattern = select(
        num_mlp_0_0_outputs, attn_1_1_outputs, num_predicate_3_1
    )
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_attn_0_0_outputs)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(attn_1_0_output, num_mlp_0_0_output):
        if attn_1_0_output in {"1", "0"}:
            return num_mlp_0_0_output == 4
        elif attn_1_0_output in {
            "6",
            "7",
            "10",
            "2",
            "3",
            "4",
            "5",
            "12",
            "11",
            "9",
            "<s>",
        }:
            return num_mlp_0_0_output == 13
        elif attn_1_0_output in {"13", "8"}:
            return num_mlp_0_0_output == 10

    num_attn_3_2_pattern = select(
        num_mlp_0_0_outputs, attn_1_0_outputs, num_predicate_3_2
    )
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, num_attn_0_0_outputs)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(attn_1_1_output, num_mlp_0_0_output):
        if attn_1_1_output in {"4", "6", "12", "0"}:
            return num_mlp_0_0_output == 6
        elif attn_1_1_output in {"7", "13", "10", "1", "2", "3", "11", "9"}:
            return num_mlp_0_0_output == 11
        elif attn_1_1_output in {"5"}:
            return num_mlp_0_0_output == 14
        elif attn_1_1_output in {"8"}:
            return num_mlp_0_0_output == 2
        elif attn_1_1_output in {"<s>"}:
            return num_mlp_0_0_output == 1

    num_attn_3_3_pattern = select(
        num_mlp_0_0_outputs, attn_1_1_outputs, num_predicate_3_3
    )
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, num_attn_0_2_outputs)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(attn_2_1_output, attn_1_3_output):
        key = (attn_2_1_output, attn_1_3_output)
        if key in {
            ("0", "8"),
            ("1", "8"),
            ("10", "6"),
            ("10", "8"),
            ("11", "8"),
            ("13", "8"),
            ("3", "6"),
            ("3", "8"),
            ("4", "8"),
            ("5", "8"),
            ("6", "6"),
            ("6", "8"),
            ("7", "8"),
            ("8", "0"),
            ("8", "1"),
            ("8", "10"),
            ("8", "13"),
            ("8", "3"),
            ("8", "4"),
            ("8", "5"),
            ("8", "6"),
            ("8", "7"),
            ("8", "8"),
            ("8", "9"),
            ("8", "<s>"),
            ("9", "8"),
            ("<s>", "8"),
        }:
            return 11
        elif key in {
            ("0", "12"),
            ("1", "12"),
            ("1", "4"),
            ("10", "12"),
            ("11", "12"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "13"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "7"),
            ("12", "9"),
            ("12", "<s>"),
            ("13", "12"),
            ("3", "12"),
            ("4", "12"),
            ("5", "12"),
            ("6", "12"),
            ("7", "12"),
            ("9", "0"),
            ("9", "12"),
            ("9", "13"),
            ("9", "3"),
            ("9", "4"),
            ("<s>", "12"),
        }:
            return 0
        elif key in {
            ("0", "2"),
            ("1", "11"),
            ("1", "2"),
            ("10", "2"),
            ("11", "11"),
            ("11", "2"),
            ("13", "11"),
            ("13", "2"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "13"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "7"),
            ("2", "9"),
            ("2", "<s>"),
            ("3", "2"),
            ("4", "2"),
            ("5", "2"),
            ("6", "2"),
            ("7", "2"),
            ("9", "11"),
            ("9", "2"),
            ("<s>", "2"),
        }:
            return 12
        elif key in {("2", "6"), ("2", "8"), ("8", "11"), ("8", "2")}:
            return 3
        elif key in {("12", "6"), ("12", "8"), ("8", "12")}:
            return 8
        elif key in {("12", "2"), ("2", "12")}:
            return 9
        return 5

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_3_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(attn_1_1_output, attn_2_1_output):
        key = (attn_1_1_output, attn_2_1_output)
        if key in {
            ("0", "12"),
            ("0", "6"),
            ("0", "9"),
            ("1", "12"),
            ("1", "9"),
            ("10", "11"),
            ("10", "12"),
            ("10", "6"),
            ("10", "9"),
            ("11", "0"),
            ("11", "1"),
            ("11", "10"),
            ("11", "12"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "6"),
            ("11", "8"),
            ("11", "9"),
            ("12", "0"),
            ("12", "12"),
            ("12", "2"),
            ("12", "6"),
            ("12", "9"),
            ("13", "10"),
            ("13", "13"),
            ("13", "<s>"),
            ("2", "12"),
            ("3", "12"),
            ("3", "9"),
            ("4", "12"),
            ("4", "6"),
            ("4", "9"),
            ("5", "12"),
            ("6", "12"),
            ("8", "12"),
            ("8", "6"),
            ("8", "9"),
            ("9", "12"),
            ("9", "6"),
            ("9", "9"),
            ("<s>", "11"),
            ("<s>", "12"),
            ("<s>", "6"),
            ("<s>", "9"),
        }:
            return 8
        elif key in {
            ("0", "7"),
            ("1", "7"),
            ("10", "7"),
            ("12", "7"),
            ("13", "7"),
            ("2", "7"),
            ("3", "10"),
            ("3", "3"),
            ("3", "7"),
            ("3", "<s>"),
            ("4", "7"),
            ("5", "7"),
            ("6", "7"),
            ("7", "0"),
            ("7", "1"),
            ("7", "10"),
            ("7", "2"),
            ("7", "3"),
            ("7", "4"),
            ("7", "5"),
            ("7", "6"),
            ("7", "7"),
            ("7", "8"),
            ("7", "9"),
            ("7", "<s>"),
            ("8", "7"),
            ("9", "7"),
            ("<s>", "7"),
        }:
            return 6
        elif key in {
            ("0", "11"),
            ("0", "13"),
            ("1", "11"),
            ("1", "13"),
            ("10", "13"),
            ("11", "11"),
            ("11", "13"),
            ("11", "<s>"),
            ("12", "11"),
            ("12", "13"),
            ("13", "0"),
            ("13", "1"),
            ("13", "11"),
            ("13", "12"),
            ("13", "2"),
            ("13", "3"),
            ("13", "4"),
            ("13", "5"),
            ("13", "6"),
            ("13", "8"),
            ("13", "9"),
            ("2", "11"),
            ("2", "13"),
            ("3", "11"),
            ("3", "13"),
            ("4", "11"),
            ("4", "13"),
            ("5", "11"),
            ("5", "13"),
            ("6", "11"),
            ("6", "13"),
            ("8", "11"),
            ("8", "13"),
            ("9", "11"),
            ("9", "13"),
            ("<s>", "13"),
        }:
            return 13
        elif key in {("11", "7"), ("7", "11"), ("7", "12")}:
            return 3
        elif key in {("7", "13")}:
            return 11
        return 15

    mlp_3_1_outputs = [
        mlp_3_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_1_outputs)
    ]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_attn_2_0_output, num_var0_embedding):
        key = (num_attn_2_0_output, num_var0_embedding)
        if key in {
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
            (24, 0),
            (25, 0),
            (26, 0),
            (27, 0),
            (28, 0),
            (29, 0),
            (30, 0),
            (30, 1),
            (31, 0),
            (31, 1),
            (32, 0),
            (32, 1),
            (33, 0),
            (33, 1),
            (34, 0),
            (34, 1),
            (35, 0),
            (35, 1),
            (36, 0),
            (36, 1),
            (37, 0),
            (37, 1),
            (38, 0),
            (38, 1),
            (39, 0),
            (39, 1),
            (40, 0),
            (40, 1),
            (41, 0),
            (41, 1),
            (42, 0),
            (42, 1),
            (43, 0),
            (43, 1),
            (44, 0),
            (44, 1),
            (45, 0),
            (45, 1),
            (46, 0),
            (46, 1),
            (47, 0),
            (47, 1),
            (48, 0),
            (48, 1),
            (49, 0),
            (49, 1),
            (50, 0),
            (50, 1),
            (50, 2),
            (51, 0),
            (51, 1),
            (51, 2),
            (52, 0),
            (52, 1),
            (52, 2),
            (53, 0),
            (53, 1),
            (53, 2),
            (54, 0),
            (54, 1),
            (54, 2),
            (55, 0),
            (55, 1),
            (55, 2),
            (56, 0),
            (56, 1),
            (56, 2),
            (57, 0),
            (57, 1),
            (57, 2),
            (58, 0),
            (58, 1),
            (58, 2),
            (59, 0),
            (59, 1),
            (59, 2),
            (60, 0),
            (60, 1),
            (60, 2),
            (61, 0),
            (61, 1),
            (61, 2),
            (62, 0),
            (62, 1),
            (62, 2),
            (63, 0),
            (63, 1),
            (63, 2),
        }:
            return 10
        return 11

    num_mlp_3_0_outputs = [
        num_mlp_3_0(k0, k1) for k0, k1 in zip(num_attn_2_0_outputs, num_var0_embeddings)
    ]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_1_3_output, num_attn_2_0_output):
        key = (num_attn_1_3_output, num_attn_2_0_output)
        if key in {
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
            (4, 63),
        }:
            return 7
        return 11

    num_mlp_3_1_outputs = [
        num_mlp_3_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_0_outputs)
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


print(run(["<s>", "1", "8", "1", "2", "9", "0", "5", "8", "2", "11", "10", "9"]))
