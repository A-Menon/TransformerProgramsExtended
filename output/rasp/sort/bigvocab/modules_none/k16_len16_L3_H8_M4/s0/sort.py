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
        "output\rasp\sort\bigvocab\modules_none\k16_len16_L3_H8_M4\s0\sort_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 12}:
            return token == "8"
        elif position in {8, 1, 10, 11}:
            return token == "0"
        elif position in {2, 3, 14, 15}:
            return token == "1"
        elif position in {4, 5, 6}:
            return token == "4"
        elif position in {9, 13, 7}:
            return token == "9"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 1, 6}:
            return token == "3"
        elif position in {2, 3}:
            return token == "2"
        elif position in {9, 4, 7}:
            return token == "8"
        elif position in {5}:
            return token == "12"
        elif position in {8, 10, 11, 12, 13, 14}:
            return token == "9"
        elif position in {15}:
            return token == "4"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 1
        elif q_position in {1, 15, 7}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3, 6}:
            return k_position == 2
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {8, 12}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6, 7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 15}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 6
        elif q_position in {12, 13}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 11

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 15}:
            return token == "10"
        elif position in {1, 2}:
            return token == "1"
        elif position in {11, 3, 4, 5}:
            return token == "0"
        elif position in {6, 7, 8, 9, 10}:
            return token == "3"
        elif position in {12, 14}:
            return token == "12"
        elif position in {13}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13, 15}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11, 15}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 0
        elif q_position in {14}:
            return k_position == 15

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 13, 15}:
            return token == "1"
        elif position in {1, 2, 14}:
            return token == "0"
        elif position in {3}:
            return token == "<pad>"
        elif position in {4, 5, 6, 7, 8, 9, 10, 11}:
            return token == "2"
        elif position in {12}:
            return token == "11"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {3, 4}:
            return 3
        elif key in {2}:
            return 2
        elif key in {1}:
            return 10
        return 5

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {3, 4, 5, 6}:
            return 4
        elif key in {1, 2}:
            return 2
        return 1

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {(0, 0)}:
            return 13
        return 14

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_0_output):
        key = (num_attn_0_3_output, num_attn_0_0_output)
        return 2

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 2, 3, 10, 15}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {11, 5}:
            return k_position == 3
        elif q_position in {6, 14}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {8, 13}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 8

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, attn_0_1_output):
        if position in {0, 13}:
            return attn_0_1_output == "9"
        elif position in {1}:
            return attn_0_1_output == "1"
        elif position in {2, 3, 4, 5, 6, 7}:
            return attn_0_1_output == "3"
        elif position in {8}:
            return attn_0_1_output == "11"
        elif position in {9, 15}:
            return attn_0_1_output == "5"
        elif position in {10, 11, 12, 14}:
            return attn_0_1_output == "8"

    attn_1_1_pattern = select_closest(attn_0_1_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {0}:
            return attn_0_0_output == "<pad>"
        elif mlp_0_0_output in {1, 13, 14, 7}:
            return attn_0_0_output == "6"
        elif mlp_0_0_output in {2}:
            return attn_0_0_output == "0"
        elif mlp_0_0_output in {10, 3}:
            return attn_0_0_output == "1"
        elif mlp_0_0_output in {11, 4, 15}:
            return attn_0_0_output == "2"
        elif mlp_0_0_output in {5}:
            return attn_0_0_output == "9"
        elif mlp_0_0_output in {9, 6}:
            return attn_0_0_output == "4"
        elif mlp_0_0_output in {8}:
            return attn_0_0_output == "3"
        elif mlp_0_0_output in {12}:
            return attn_0_0_output == "7"

    attn_1_2_pattern = select_closest(attn_0_0_outputs, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 11}:
            return token == "7"
        elif num_mlp_0_0_output in {1, 2, 3}:
            return token == "11"
        elif num_mlp_0_0_output in {4, 15}:
            return token == "4"
        elif num_mlp_0_0_output in {8, 10, 5, 6}:
            return token == "5"
        elif num_mlp_0_0_output in {7, 9, 12, 13, 14}:
            return token == "9"

    attn_1_3_pattern = select_closest(tokens, num_mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, num_mlp_0_0_output):
        if position in {0, 14}:
            return num_mlp_0_0_output == 3
        elif position in {1, 3}:
            return num_mlp_0_0_output == 2
        elif position in {2, 4, 5, 15}:
            return num_mlp_0_0_output == 1
        elif position in {6}:
            return num_mlp_0_0_output == 5
        elif position in {7}:
            return num_mlp_0_0_output == 4
        elif position in {8}:
            return num_mlp_0_0_output == 6
        elif position in {9}:
            return num_mlp_0_0_output == 8
        elif position in {10, 11, 12, 13}:
            return num_mlp_0_0_output == 15

    num_attn_1_0_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, mlp_0_1_output):
        if position in {0, 2}:
            return mlp_0_1_output == 4
        elif position in {1}:
            return mlp_0_1_output == 3
        elif position in {3, 4}:
            return mlp_0_1_output == 2
        elif position in {5, 6, 7}:
            return mlp_0_1_output == 1
        elif position in {8}:
            return mlp_0_1_output == 6
        elif position in {9}:
            return mlp_0_1_output == 9
        elif position in {10, 11, 12, 13}:
            return mlp_0_1_output == 15
        elif position in {14}:
            return mlp_0_1_output == 10
        elif position in {15}:
            return mlp_0_1_output == 5

    num_attn_1_1_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_var0_embeddings)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, mlp_0_0_output):
        if position in {0, 2, 15}:
            return mlp_0_0_output == 3
        elif position in {1, 5}:
            return mlp_0_0_output == 2
        elif position in {3, 4}:
            return mlp_0_0_output == 5
        elif position in {8, 6, 7}:
            return mlp_0_0_output == 4
        elif position in {9, 11}:
            return mlp_0_0_output == 14
        elif position in {10, 13, 14}:
            return mlp_0_0_output == 15
        elif position in {12}:
            return mlp_0_0_output == 0

    num_attn_1_2_pattern = select(mlp_0_0_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_var0_embeddings)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, mlp_0_1_output):
        if position in {0, 1, 3}:
            return mlp_0_1_output == 4
        elif position in {2, 15}:
            return mlp_0_1_output == 2
        elif position in {4, 5, 6}:
            return mlp_0_1_output == 1
        elif position in {7}:
            return mlp_0_1_output == 5
        elif position in {8, 14}:
            return mlp_0_1_output == 8
        elif position in {9, 10, 11}:
            return mlp_0_1_output == 14
        elif position in {12, 13}:
            return mlp_0_1_output == 15

    num_attn_1_3_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_var0_embeddings)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, position):
        key = (attn_1_2_output, position)
        if key in {
            ("0", 14),
            ("1", 0),
            ("1", 2),
            ("1", 14),
            ("10", 0),
            ("10", 14),
            ("11", 0),
            ("11", 14),
            ("12", 0),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 8),
            ("12", 14),
            ("2", 0),
            ("2", 14),
            ("3", 0),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("3", 14),
            ("4", 0),
            ("4", 5),
            ("4", 7),
            ("4", 8),
            ("4", 14),
            ("5", 0),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 14),
            ("6", 0),
            ("7", 0),
            ("7", 2),
            ("7", 3),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("7", 14),
            ("8", 2),
            ("9", 2),
            ("</s>", 0),
            ("</s>", 14),
            ("<s>", 0),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 14),
        }:
            return 7
        elif key in {
            ("0", 1),
            ("0", 2),
            ("0", 15),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("4", 2),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("8", 1),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 12
        elif key in {
            ("10", 2),
            ("11", 2),
            ("12", 2),
            ("2", 2),
            ("3", 2),
            ("5", 2),
            ("6", 2),
            ("</s>", 2),
            ("<s>", 2),
        }:
            return 3
        elif key in {("10", 1), ("11", 1), ("12", 1)}:
            return 0
        return 9

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, position):
        key = (attn_1_1_output, position)
        if key in {
            ("1", 8),
            ("1", 15),
            ("10", 8),
            ("10", 15),
            ("11", 8),
            ("12", 2),
            ("12", 3),
            ("12", 4),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 8),
            ("12", 15),
            ("2", 8),
            ("2", 15),
            ("4", 2),
            ("4", 3),
            ("4", 8),
            ("4", 15),
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 8),
            ("5", 15),
            ("6", 8),
            ("6", 15),
            ("7", 8),
            ("7", 15),
            ("8", 8),
            ("8", 15),
            ("9", 2),
            ("9", 3),
            ("9", 8),
            ("9", 15),
            ("</s>", 8),
            ("</s>", 15),
            ("<s>", 8),
            ("<s>", 15),
        }:
            return 8
        elif key in {
            ("0", 11),
            ("0", 14),
            ("1", 11),
            ("1", 12),
            ("1", 14),
            ("10", 14),
            ("11", 11),
            ("11", 14),
            ("12", 0),
            ("12", 11),
            ("12", 12),
            ("12", 14),
            ("2", 11),
            ("2", 12),
            ("2", 14),
            ("3", 8),
            ("3", 11),
            ("3", 12),
            ("3", 14),
            ("4", 11),
            ("4", 12),
            ("4", 14),
            ("5", 11),
            ("5", 12),
            ("5", 14),
            ("6", 11),
            ("6", 12),
            ("6", 14),
            ("7", 11),
            ("7", 12),
            ("7", 14),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 14),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 14),
        }:
            return 14
        elif key in {
            ("0", 13),
            ("1", 13),
            ("10", 0),
            ("10", 13),
            ("11", 0),
            ("11", 13),
            ("12", 13),
            ("2", 0),
            ("2", 10),
            ("2", 13),
            ("3", 13),
            ("4", 13),
            ("5", 0),
            ("5", 13),
            ("6", 0),
            ("6", 13),
            ("7", 13),
            ("8", 0),
            ("8", 10),
            ("8", 12),
            ("8", 13),
            ("8", 14),
            ("9", 0),
            ("9", 10),
            ("9", 12),
            ("9", 13),
            ("9", 14),
            ("</s>", 0),
            ("</s>", 13),
            ("<s>", 13),
        }:
            return 12
        elif key in {
            ("0", 10),
            ("1", 0),
            ("1", 9),
            ("1", 10),
            ("10", 9),
            ("10", 10),
            ("11", 9),
            ("11", 10),
            ("2", 9),
            ("3", 0),
            ("3", 6),
            ("3", 7),
            ("3", 9),
            ("3", 10),
            ("4", 0),
            ("4", 9),
            ("4", 10),
            ("5", 9),
            ("5", 10),
            ("6", 9),
            ("6", 10),
            ("7", 0),
            ("7", 9),
            ("7", 10),
            ("8", 9),
            ("</s>", 9),
            ("</s>", 10),
            ("<s>", 0),
            ("<s>", 9),
            ("<s>", 10),
        }:
            return 4
        elif key in {
            ("0", 0),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 15),
            ("10", 5),
            ("2", 1),
            ("3", 1),
            ("3", 4),
            ("3", 5),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("8", 1),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 11
        elif key in {
            ("0", 9),
            ("0", 12),
            ("10", 11),
            ("10", 12),
            ("11", 12),
            ("12", 9),
            ("12", 10),
            ("3", 15),
            ("8", 11),
            ("9", 9),
            ("9", 11),
        }:
            return 1
        elif key in {("0", 1), ("1", 1), ("10", 1), ("11", 1), ("12", 1)}:
            return 2
        elif key in {("0", 8), ("3", 2), ("3", 3)}:
            return 10
        return 6

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        return 13

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_0_output):
        key = (num_attn_0_1_output, num_attn_1_0_output)
        return 7

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 2}:
            return token == "1"
        elif mlp_0_0_output in {1}:
            return token == "8"
        elif mlp_0_0_output in {3}:
            return token == "<s>"
        elif mlp_0_0_output in {8, 10, 4}:
            return token == "12"
        elif mlp_0_0_output in {11, 5}:
            return token == "</s>"
        elif mlp_0_0_output in {6}:
            return token == "4"
        elif mlp_0_0_output in {7}:
            return token == "6"
        elif mlp_0_0_output in {9, 13}:
            return token == "<pad>"
        elif mlp_0_0_output in {12}:
            return token == "10"
        elif mlp_0_0_output in {14, 15}:
            return token == "7"

    attn_2_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0}:
            return token == "11"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 6, 7}:
            return token == "</s>"
        elif position in {8, 9, 4, 5}:
            return token == "<s>"
        elif position in {10}:
            return token == "3"
        elif position in {11}:
            return token == "7"
        elif position in {12, 14}:
            return token == "8"
        elif position in {13}:
            return token == "9"
        elif position in {15}:
            return token == "6"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(num_mlp_1_0_output, token):
        if num_mlp_1_0_output in {0, 2}:
            return token == "10"
        elif num_mlp_1_0_output in {1}:
            return token == "9"
        elif num_mlp_1_0_output in {3, 4, 13, 14}:
            return token == "</s>"
        elif num_mlp_1_0_output in {5}:
            return token == "5"
        elif num_mlp_1_0_output in {11, 6}:
            return token == "3"
        elif num_mlp_1_0_output in {12, 7}:
            return token == "1"
        elif num_mlp_1_0_output in {8}:
            return token == "12"
        elif num_mlp_1_0_output in {9}:
            return token == "4"
        elif num_mlp_1_0_output in {10}:
            return token == "0"
        elif num_mlp_1_0_output in {15}:
            return token == "6"

    attn_2_2_pattern = select_closest(tokens, num_mlp_1_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0}:
            return token == "10"
        elif mlp_0_0_output in {1, 6}:
            return token == "0"
        elif mlp_0_0_output in {2, 3, 7}:
            return token == "<s>"
        elif mlp_0_0_output in {8, 4}:
            return token == "3"
        elif mlp_0_0_output in {5}:
            return token == "2"
        elif mlp_0_0_output in {9, 10, 11}:
            return token == "4"
        elif mlp_0_0_output in {12, 14}:
            return token == "9"
        elif mlp_0_0_output in {13}:
            return token == "7"
        elif mlp_0_0_output in {15}:
            return token == "6"

    attn_2_3_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, token):
        if attn_1_2_output in {"0", "9", "8", "1", "</s>", "7"}:
            return token == "<s>"
        elif attn_1_2_output in {"10"}:
            return token == "11"
        elif attn_1_2_output in {"11", "3", "12", "2", "<s>", "5", "4"}:
            return token == "1"
        elif attn_1_2_output in {"6"}:
            return token == "</s>"

    num_attn_2_0_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, num_mlp_0_0_output):
        if position in {0, 1}:
            return num_mlp_0_0_output == 6
        elif position in {2, 3}:
            return num_mlp_0_0_output == 14
        elif position in {4, 5, 6, 7, 8, 9}:
            return num_mlp_0_0_output == 2
        elif position in {10}:
            return num_mlp_0_0_output == 4
        elif position in {11}:
            return num_mlp_0_0_output == 7
        elif position in {12, 13}:
            return num_mlp_0_0_output == 15
        elif position in {14}:
            return num_mlp_0_0_output == 0
        elif position in {15}:
            return num_mlp_0_0_output == 1

    num_attn_2_1_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, token):
        if attn_1_2_output in {"0", "9", "8", "1", "</s>", "7", "5"}:
            return token == "<s>"
        elif attn_1_2_output in {"11", "10", "3", "12", "<s>", "4"}:
            return token == "10"
        elif attn_1_2_output in {"2", "6"}:
            return token == "1"

    num_attn_2_2_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, num_mlp_1_1_output):
        if position in {0, 2, 3}:
            return num_mlp_1_1_output == 3
        elif position in {1, 7, 8, 9, 10}:
            return num_mlp_1_1_output == 7
        elif position in {4, 5, 6, 15}:
            return num_mlp_1_1_output == 5
        elif position in {11}:
            return num_mlp_1_1_output == 0
        elif position in {12, 14}:
            return num_mlp_1_1_output == 15
        elif position in {13}:
            return num_mlp_1_1_output == 8

    num_attn_2_3_pattern = select(num_mlp_1_1_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_var0_embeddings)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, position):
        key = (attn_2_3_output, position)
        if key in {
            ("0", 4),
            ("0", 5),
            ("0", 13),
            ("1", 13),
            ("10", 13),
            ("11", 13),
            ("12", 13),
            ("2", 13),
            ("3", 10),
            ("3", 13),
            ("4", 13),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("5", 14),
            ("6", 10),
            ("6", 13),
            ("7", 10),
            ("7", 11),
            ("7", 12),
            ("7", 13),
            ("7", 14),
            ("8", 13),
            ("9", 0),
            ("9", 4),
            ("9", 5),
            ("9", 6),
            ("9", 7),
            ("9", 8),
            ("9", 10),
            ("9", 11),
            ("9", 12),
            ("9", 13),
            ("9", 14),
            ("</s>", 13),
            ("<s>", 5),
            ("<s>", 13),
        }:
            return 6
        elif key in {
            ("1", 5),
            ("10", 5),
            ("11", 5),
            ("12", 5),
            ("2", 5),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 11),
            ("3", 12),
            ("3", 14),
            ("4", 5),
            ("5", 5),
            ("6", 5),
            ("6", 12),
            ("7", 0),
            ("7", 2),
            ("7", 4),
            ("7", 5),
            ("7", 6),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("8", 2),
            ("8", 5),
            ("</s>", 5),
        }:
            return 0
        elif key in {
            ("12", 15),
            ("3", 0),
            ("3", 1),
            ("3", 15),
            ("4", 0),
            ("4", 1),
            ("4", 15),
            ("5", 0),
            ("5", 1),
            ("5", 15),
            ("6", 0),
            ("6", 1),
            ("6", 15),
            ("7", 1),
            ("7", 15),
            ("8", 0),
            ("8", 1),
            ("8", 15),
            ("9", 1),
            ("9", 15),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 15),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 15),
        }:
            return 1
        elif key in {
            ("1", 1),
            ("12", 2),
            ("3", 2),
            ("3", 3),
            ("4", 2),
            ("4", 3),
            ("5", 2),
            ("5", 3),
            ("7", 3),
        }:
            return 4
        elif key in {
            ("0", 0),
            ("0", 3),
            ("0", 15),
            ("12", 3),
            ("9", 2),
            ("9", 3),
            ("9", 9),
        }:
            return 5
        elif key in {("11", 1), ("12", 1), ("2", 1)}:
            return 9
        elif key in {("0", 1), ("10", 1), ("2", 2)}:
            return 13
        elif key in {("0", 2)}:
            return 3
        return 15

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 4),
            ("0", 10),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("1", 12),
            ("1", 13),
            ("1", 14),
            ("10", 10),
            ("10", 12),
            ("10", 13),
            ("10", 14),
            ("11", 12),
            ("11", 13),
            ("11", 14),
            ("12", 7),
            ("12", 10),
            ("12", 11),
            ("12", 12),
            ("12", 13),
            ("12", 14),
            ("2", 7),
            ("2", 10),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("3", 7),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("4", 4),
            ("4", 7),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("5", 7),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("6", 10),
            ("6", 12),
            ("6", 13),
            ("6", 14),
            ("7", 7),
            ("7", 10),
            ("8", 7),
            ("8", 10),
            ("8", 11),
            ("8", 12),
            ("9", 10),
            ("9", 12),
            ("</s>", 7),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("<s>", 10),
            ("<s>", 12),
        }:
            return 1
        elif key in {
            ("0", 9),
            ("0", 11),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 15),
            ("10", 3),
            ("10", 4),
            ("10", 7),
            ("10", 11),
            ("10", 15),
            ("11", 10),
            ("11", 11),
            ("2", 3),
            ("2", 4),
            ("2", 11),
            ("3", 3),
            ("6", 4),
            ("6", 7),
            ("6", 9),
            ("6", 11),
            ("8", 9),
            ("9", 7),
            ("9", 9),
            ("9", 11),
            ("</s>", 4),
            ("</s>", 9),
            ("<s>", 3),
            ("<s>", 7),
            ("<s>", 11),
        }:
            return 14
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 15),
            ("12", 1),
            ("12", 15),
            ("2", 0),
            ("2", 1),
            ("2", 15),
            ("3", 1),
            ("3", 15),
            ("4", 1),
            ("4", 15),
            ("5", 1),
            ("5", 15),
            ("6", 0),
            ("6", 1),
            ("6", 15),
            ("7", 1),
            ("7", 15),
            ("8", 0),
            ("8", 1),
            ("8", 15),
            ("9", 0),
            ("9", 1),
            ("9", 15),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 15),
            ("<s>", 1),
            ("<s>", 15),
        }:
            return 8
        elif key in {
            ("2", 2),
            ("4", 2),
            ("4", 3),
            ("5", 2),
            ("5", 3),
            ("6", 2),
            ("6", 3),
            ("6", 5),
            ("6", 6),
            ("6", 8),
            ("7", 2),
            ("7", 3),
            ("8", 3),
            ("9", 3),
            ("</s>", 3),
        }:
            return 6
        elif key in {
            ("0", 2),
            ("1", 0),
            ("1", 2),
            ("12", 2),
            ("8", 2),
            ("8", 8),
            ("9", 2),
            ("</s>", 2),
        }:
            return 12
        elif key in {
            ("0", 7),
            ("11", 2),
            ("11", 3),
            ("11", 6),
            ("11", 7),
            ("11", 8),
            ("3", 2),
        }:
            return 10
        elif key in {("0", 6), ("0", 8), ("10", 1), ("10", 2), ("9", 8)}:
            return 5
        elif key in {("1", 1), ("11", 1)}:
            return 3
        return 15

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_2_output):
        key = (num_attn_1_2_output, num_attn_2_2_output)
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
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
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
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
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
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
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
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
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
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
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
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
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
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
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
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
            (11, 47),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (14, 40),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (15, 40),
            (15, 41),
            (15, 42),
            (15, 43),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (16, 37),
            (16, 38),
            (16, 39),
            (16, 40),
            (16, 41),
            (16, 42),
            (16, 43),
            (16, 44),
            (16, 45),
            (16, 46),
            (16, 47),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (17, 39),
            (17, 40),
            (17, 41),
            (17, 42),
            (17, 43),
            (17, 44),
            (17, 45),
            (17, 46),
            (17, 47),
            (18, 37),
            (18, 38),
            (18, 39),
            (18, 40),
            (18, 41),
            (18, 42),
            (18, 43),
            (18, 44),
            (18, 45),
            (18, 46),
            (18, 47),
            (19, 39),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 43),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (20, 42),
            (20, 43),
            (20, 44),
            (20, 45),
            (20, 46),
            (20, 47),
            (21, 44),
            (21, 45),
            (21, 46),
            (21, 47),
            (22, 46),
            (22, 47),
        }:
            return 3
        return 15

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0}:
            return 14
        return 2

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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


print(run(["<s>", "3", "3", "2", "4", "</s>"]))
