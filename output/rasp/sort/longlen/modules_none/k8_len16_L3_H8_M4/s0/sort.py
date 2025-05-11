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
        "output\rasp\sort\longlen\modules_none\k8_len16_L3_H8_M4\s0\sort_weights.csv",
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
        if position in {0, 1, 3, 5, 8, 9, 10, 11, 13, 14}:
            return token == "4"
        elif position in {2, 4, 6, 7, 12}:
            return token == "3"
        elif position in {15}:
            return token == "1"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"1", "3", "0", "<s>", "2"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"</s>"}:
            return k_token == "2"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"1", "3", "0", "</s>", "<s>", "2"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"<s>", "3", "2", "0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"4", "</s>"}:
            return k_token == "4"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 4, 15}:
            return token == "0"
        elif position in {5, 6, 7, 8, 9, 10, 12, 13, 14}:
            return token == ""
        elif position in {11}:
            return token == "</s>"

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
        elif q_position in {2, 3}:
            return k_position == 5
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
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13, 14}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 3

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 6, 7, 8, 9, 11}:
            return token == "0"
        elif position in {1, 2, 3, 5}:
            return token == "</s>"
        elif position in {4, 10, 12, 13, 14, 15}:
            return token == "1"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10, 11, 13}:
            return k_position == 14
        elif q_position in {12, 14}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 2

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {3, 4, 5}:
            return 5
        elif key in {1, 2, 15}:
            return 6
        elif key in {0, 6}:
            return 2
        elif key in {7}:
            return 12
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {11, 12, 13, 14}:
            return 8
        elif key in {1, 2, 3, 15}:
            return 9
        elif key in {4}:
            return 15
        return 3

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 6

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        if key in {(0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15)}:
            return 10
        return 12

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, attn_0_0_output):
        if position in {0, 5, 6, 9, 10, 11, 12, 14}:
            return attn_0_0_output == "4"
        elif position in {1, 13}:
            return attn_0_0_output == "0"
        elif position in {2}:
            return attn_0_0_output == ""
        elif position in {8, 3, 7}:
            return attn_0_0_output == "3"
        elif position in {4}:
            return attn_0_0_output == "2"
        elif position in {15}:
            return attn_0_0_output == "1"

    attn_1_0_pattern = select_closest(attn_0_0_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 2, 15}:
            return token == "3"
        elif num_mlp_0_0_output in {1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14}:
            return token == "4"
        elif num_mlp_0_0_output in {9}:
            return token == ""
        elif num_mlp_0_0_output in {13}:
            return token == "1"

    attn_1_1_pattern = select_closest(tokens, num_mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 2, 4, 5, 11}:
            return token == "3"
        elif num_mlp_0_0_output in {1, 3, 9}:
            return token == ""
        elif num_mlp_0_0_output in {6, 7, 8, 10, 12, 13, 14, 15}:
            return token == "4"

    attn_1_2_pattern = select_closest(tokens, num_mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, mlp_0_0_output):
        if position in {0, 3, 6, 7, 8, 9, 10, 11, 14}:
            return mlp_0_0_output == 5
        elif position in {1, 12}:
            return mlp_0_0_output == 1
        elif position in {2, 4}:
            return mlp_0_0_output == 12
        elif position in {5}:
            return mlp_0_0_output == 6
        elif position in {13}:
            return mlp_0_0_output == 14
        elif position in {15}:
            return mlp_0_0_output == 3

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {0, 4, 5, 7, 8, 9, 10, 11, 14, 15}:
            return token == "2"
        elif num_mlp_0_1_output in {1, 2, 3, 6, 13}:
            return token == "0"
        elif num_mlp_0_1_output in {12}:
            return token == "1"

    num_attn_1_0_pattern = select(tokens, num_mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 15}:
            return token == "1"
        elif mlp_0_0_output in {1, 7, 10, 12, 14}:
            return token == ""
        elif mlp_0_0_output in {2, 3, 4, 5, 6, 11}:
            return token == "0"
        elif mlp_0_0_output in {8, 13}:
            return token == "<pad>"
        elif mlp_0_0_output in {9}:
            return token == "</s>"

    num_attn_1_1_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {0, 6}:
            return attn_0_0_output == "1"
        elif mlp_0_1_output in {1}:
            return attn_0_0_output == "</s>"
        elif mlp_0_1_output in {2, 4, 5, 9, 13, 15}:
            return attn_0_0_output == "0"
        elif mlp_0_1_output in {3, 8, 10, 11, 14}:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {7}:
            return attn_0_0_output == "2"
        elif mlp_0_1_output in {12}:
            return attn_0_0_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_0_outputs, mlp_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 3, 4, 5, 6, 7, 15}:
            return token == "1"
        elif position in {8, 9}:
            return token == "</s>"
        elif position in {10, 12, 13, 14}:
            return token == ""
        elif position in {11}:
            return token == "<s>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_var0_embeddings)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position):
        key = position
        if key in {3, 4, 5, 15}:
            return 4
        elif key in {0, 1, 2}:
            return 6
        return 7

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in positions]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("0", 4),
            ("0", 10),
            ("1", 3),
            ("1", 4),
            ("2", 3),
            ("2", 4),
            ("2", 15),
            ("3", 3),
            ("3", 4),
            ("3", 15),
            ("4", 3),
            ("4", 4),
            ("4", 15),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 15),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 15),
        }:
            return 7
        elif key in {
            ("0", 13),
            ("0", 14),
            ("1", 13),
            ("1", 14),
            ("2", 13),
            ("2", 14),
            ("3", 13),
            ("3", 14),
            ("4", 13),
            ("4", 14),
            ("</s>", 13),
            ("</s>", 14),
            ("<s>", 13),
            ("<s>", 14),
        }:
            return 13
        elif key in {
            ("0", 1),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 3
        elif key in {
            ("0", 2),
            ("0", 3),
            ("1", 2),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("</s>", 2),
            ("<s>", 2),
        }:
            return 11
        elif key in {("0", 0), ("0", 9), ("0", 15)}:
            return 2
        return 4

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 5

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 1

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 2, 5, 8, 10}:
            return token == ""
        elif mlp_0_0_output in {1, 3}:
            return token == "4"
        elif mlp_0_0_output in {4}:
            return token == "<s>"
        elif mlp_0_0_output in {13, 6}:
            return token == "0"
        elif mlp_0_0_output in {7}:
            return token == "1"
        elif mlp_0_0_output in {9, 15}:
            return token == "2"
        elif mlp_0_0_output in {11, 12, 14}:
            return token == "3"

    attn_2_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_0_output, position):
        if mlp_1_0_output in {0, 9, 2}:
            return position == 10
        elif mlp_1_0_output in {1, 11, 6}:
            return position == 12
        elif mlp_1_0_output in {3}:
            return position == 15
        elif mlp_1_0_output in {8, 4}:
            return position == 1
        elif mlp_1_0_output in {15, 5, 7}:
            return position == 3
        elif mlp_1_0_output in {10, 12, 14}:
            return position == 5
        elif mlp_1_0_output in {13}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, mlp_1_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_0_output, num_mlp_1_1_output):
        if mlp_0_0_output in {0, 6}:
            return num_mlp_1_1_output == 1
        elif mlp_0_0_output in {1, 12, 5}:
            return num_mlp_1_1_output == 6
        elif mlp_0_0_output in {2}:
            return num_mlp_1_1_output == 7
        elif mlp_0_0_output in {3, 7}:
            return num_mlp_1_1_output == 10
        elif mlp_0_0_output in {4}:
            return num_mlp_1_1_output == 3
        elif mlp_0_0_output in {8, 9}:
            return num_mlp_1_1_output == 15
        elif mlp_0_0_output in {10, 11}:
            return num_mlp_1_1_output == 5
        elif mlp_0_0_output in {13}:
            return num_mlp_1_1_output == 12
        elif mlp_0_0_output in {14}:
            return num_mlp_1_1_output == 9
        elif mlp_0_0_output in {15}:
            return num_mlp_1_1_output == 2

    attn_2_2_pattern = select_closest(
        num_mlp_1_1_outputs, mlp_0_0_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, token):
        if position in {0, 3, 6, 7}:
            return token == ""
        elif position in {1, 15}:
            return token == "1"
        elif position in {2}:
            return token == "<s>"
        elif position in {8, 4}:
            return token == "3"
        elif position in {5, 10, 11, 12, 13, 14}:
            return token == "4"
        elif position in {9}:
            return token == "</s>"

    attn_2_3_pattern = select_closest(tokens, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15}:
            return token == "2"
        elif mlp_0_1_output in {8}:
            return token == "1"
        elif mlp_0_1_output in {14}:
            return token == "3"

    num_attn_2_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 2, 3, 4, 11}:
            return token == "1"
        elif mlp_0_1_output in {5, 6, 7, 9, 12, 15}:
            return token == "2"
        elif mlp_0_1_output in {8, 10, 13, 14}:
            return token == ""

    num_attn_2_1_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {0, 1, 2, 3, 4, 15}:
            return token == "1"
        elif position in {5, 6, 7}:
            return token == "0"
        elif position in {8}:
            return token == "</s>"
        elif position in {9}:
            return token == "<pad>"
        elif position in {10, 11, 12, 13, 14}:
            return token == ""

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, token):
        if attn_1_3_output in {"0"}:
            return token == "1"
        elif attn_1_3_output in {"1", "3", "4", "</s>", "<s>", "2"}:
            return token == "0"

    num_attn_2_3_pattern = select(tokens, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
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
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 14),
            ("4", 0),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("</s>", 4),
            ("<s>", 2),
            ("<s>", 4),
            ("<s>", 15),
        }:
            return 4
        elif key in {
            ("0", 14),
            ("1", 14),
            ("2", 14),
            ("4", 14),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 14),
            ("<s>", 3),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 14),
        }:
            return 12
        elif key in {
            ("0", 10),
            ("0", 12),
            ("1", 10),
            ("1", 12),
            ("2", 10),
            ("2", 12),
            ("4", 10),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("<s>", 0),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
        }:
            return 2
        elif key in {
            ("0", 1),
            ("0", 8),
            ("0", 15),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 10
        elif key in {("1", 1), ("2", 1)}:
            return 0
        elif key in {("1", 2), ("1", 15), ("2", 0), ("2", 2), ("2", 15)}:
            return 3
        elif key in {("1", 0)}:
            return 5
        elif key in {("1", 11)}:
            return 1
        return 14

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, attn_2_2_output):
        key = (attn_2_3_output, attn_2_2_output)
        if key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "</s>"),
            ("</s>", "4"),
        }:
            return 12
        elif key in {
            ("0", "0"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("2", "1"),
            ("2", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 5
        elif key in {("3", "<s>"), ("4", "<s>"), ("<s>", "4")}:
            return 10
        elif key in {("3", "3")}:
            return 4
        return 14

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_2_1_output):
        key = (num_attn_1_1_output, num_attn_2_1_output)
        if key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (9, 2),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
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
            (22, 5),
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
            (26, 6),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
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
            (30, 7),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
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
            (47, 10),
            (47, 11),
        }:
            return 5
        elif key in {
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
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (6, 46),
            (6, 47),
        }:
            return 13
        return 6

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output):
        key = num_attn_2_2_output
        return 14

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_2_outputs]
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


print(run(["<s>", "0", "4", "3", "3", "0", "1", "1", "4", "2", "</s>"]))
