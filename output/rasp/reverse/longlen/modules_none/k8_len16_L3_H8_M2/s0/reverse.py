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
        "output\rasp\reverse\longlen\modules_none\k8_len16_L3_H8_M2\s0\reverse_weights.csv",
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
        if q_position in {0, 1, 2, 9}:
            return k_position == 6
        elif q_position in {8, 3}:
            return k_position == 3
        elif q_position in {11, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 15
        elif q_position in {10, 6, 7}:
            return k_position == 2
        elif q_position in {12, 13, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 8

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 6
        elif q_position in {1, 6}:
            return k_position == 9
        elif q_position in {2, 4, 14}:
            return k_position == 11
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 12, 13}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 0
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 8, 3, 9}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {10, 11, 4, 12}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6, 15}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {13, 14}:
            return k_position == 15

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5, 15}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8, 14}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 4
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {12, 13}:
            return k_position == 1

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2, 10, 7}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {4, 5}:
            return k_position == 15
        elif q_position in {6, 11, 12, 13, 14, 15}:
            return k_position == 1
        elif q_position in {8, 9}:
            return k_position == 4

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1, 2}:
            return k_position == 13
        elif q_position in {3, 4, 5, 6, 7}:
            return k_position == 15
        elif q_position in {8, 9, 12}:
            return k_position == 1
        elif q_position in {10, 11, 13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 4

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3}:
            return k_position == 10
        elif q_position in {4, 5, 6, 9, 11}:
            return k_position == 15
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8, 12}:
            return k_position == 0
        elif q_position in {10}:
            return k_position == 4
        elif q_position in {13, 14, 15}:
            return k_position == 1

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2, 3, 4, 5, 6, 7}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9, 10, 11, 12, 13, 14, 15}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, num_var0_embeddings)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, position):
        key = (token, position)
        if key in {
            ("0", 5),
            ("0", 14),
            ("1", 5),
            ("1", 14),
            ("2", 5),
            ("2", 14),
            ("3", 5),
            ("3", 14),
            ("4", 5),
            ("4", 14),
            ("<s>", 5),
            ("<s>", 14),
        }:
            return 1
        elif key in {
            ("0", 7),
            ("1", 7),
            ("2", 7),
            ("3", 7),
            ("4", 7),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 9),
            ("</s>", 10),
            ("<s>", 7),
        }:
            return 9
        elif key in {
            ("0", 11),
            ("1", 11),
            ("2", 11),
            ("3", 11),
            ("4", 11),
            ("</s>", 11),
            ("<s>", 11),
        }:
            return 4
        elif key in {("0", 8), ("1", 8), ("2", 8), ("3", 8), ("4", 8), ("<s>", 8)}:
            return 7
        elif key in {("</s>", 12), ("</s>", 13), ("</s>", 14), ("</s>", 15)}:
            return 12
        elif key in {("</s>", 1), ("</s>", 2), ("</s>", 5), ("</s>", 8)}:
            return 6
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        if key in {(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}:
            return 4
        elif key in {(0, 0), (0, 1), (0, 2)}:
            return 7
        return 5

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2, 3}:
            return k_position == 9
        elif q_position in {8, 4}:
            return k_position == 7
        elif q_position in {9, 5, 6, 7}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 4
        elif q_position in {11, 14}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {15}:
            return k_position == 11

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 7
        elif q_position in {1, 2}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {13, 4, 12}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {7, 9, 10, 11, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 11

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 5
        elif q_position in {1, 2}:
            return k_position == 12
        elif q_position in {3, 15}:
            return k_position == 11
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 15
        elif q_position in {8, 11, 6, 14}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {9, 12, 13}:
            return k_position == 2

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 11}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2, 12, 7}:
            return k_position == 2
        elif q_position in {3, 13}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {14}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 1

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_3_output):
        if position in {0, 5}:
            return attn_0_3_output == "</s>"
        elif position in {1, 4}:
            return attn_0_3_output == "<pad>"
        elif position in {2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            return attn_0_3_output == ""
        elif position in {15}:
            return attn_0_3_output == "4"

    num_attn_1_0_pattern = select(attn_0_3_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 4, 5, 6, 13, 14}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {1, 3}:
            return k_mlp_0_0_output == 11
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 0
        elif q_mlp_0_0_output in {7}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {8, 9, 10}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {11}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {12, 15}:
            return k_mlp_0_0_output == 7

    num_attn_1_1_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 1, 2, 3, 4, 9, 11, 13, 14, 15}:
            return token == ""
        elif position in {5, 6, 7, 10, 12}:
            return token == "</s>"
        elif position in {8}:
            return token == "3"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"0", "3", "2"}:
            return attn_0_0_output == "<pad>"
        elif attn_0_3_output in {"4", "<s>", "1"}:
            return attn_0_0_output == ""
        elif attn_0_3_output in {"</s>"}:
            return attn_0_0_output == "<s>"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_var0_embeddings)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_0_2_output):
        key = (attn_1_0_output, attn_0_2_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 1
        elif key in {("<s>", "2")}:
            return 3
        elif key in {("4", "</s>")}:
            return 2
        elif key in {("0", "</s>"), ("</s>", "0")}:
            return 7
        elif key in {("</s>", "2")}:
            return 9
        elif key in {("3", "</s>")}:
            return 5
        elif key in {("</s>", "3")}:
            return 15
        elif key in {("</s>", "</s>"), ("</s>", "<s>")}:
            return 8
        elif key in {("2", "</s>")}:
            return 10
        return 14

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        if key in {0}:
            return 15
        return 14

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 10, 13, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3, 5}:
            return k_position == 10
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {8, 9}:
            return k_position == 11
        elif q_position in {11, 12}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 3

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 4, 7, 12, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 3}:
            return k_position == 10
        elif q_position in {9, 10, 11, 5}:
            return k_position == 8
        elif q_position in {13, 6}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 1

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 3, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {8, 5}:
            return k_position == 12
        elif q_position in {11, 6}:
            return k_position == 11
        elif q_position in {9, 10, 13, 7}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 2

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 9, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {10, 3, 4}:
            return k_position == 10
        elif q_position in {13, 5}:
            return k_position == 13
        elif q_position in {6}:
            return k_position == 1
        elif q_position in {8, 7}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 2

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, attn_1_1_output):
        if attn_0_3_output in {"4", "0", "3", "</s>"}:
            return attn_1_1_output == ""
        elif attn_0_3_output in {"1", "2"}:
            return attn_1_1_output == "<s>"
        elif attn_0_3_output in {"<s>"}:
            return attn_1_1_output == "</s>"

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 3
        elif position in {1}:
            return mlp_0_0_output == 12
        elif position in {10, 2, 3}:
            return mlp_0_0_output == 6
        elif position in {9, 4, 6}:
            return mlp_0_0_output == 2
        elif position in {5}:
            return mlp_0_0_output == 7
        elif position in {8, 7}:
            return mlp_0_0_output == 9
        elif position in {11}:
            return mlp_0_0_output == 4
        elif position in {12}:
            return mlp_0_0_output == 8
        elif position in {13}:
            return mlp_0_0_output == 10
        elif position in {14, 15}:
            return mlp_0_0_output == 1

    num_attn_2_1_pattern = select(mlp_0_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, mlp_0_0_output):
        if position in {0, 15}:
            return mlp_0_0_output == 3
        elif position in {1}:
            return mlp_0_0_output == 0
        elif position in {2, 3, 5, 7}:
            return mlp_0_0_output == 6
        elif position in {4, 6, 14}:
            return mlp_0_0_output == 8
        elif position in {8}:
            return mlp_0_0_output == 7
        elif position in {9, 11, 12}:
            return mlp_0_0_output == 4
        elif position in {10}:
            return mlp_0_0_output == 12
        elif position in {13}:
            return mlp_0_0_output == 1

    num_attn_2_2_pattern = select(mlp_0_0_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_2_output, attn_0_1_output):
        if attn_1_2_output in {"1", "0", "3"}:
            return attn_0_1_output == "<s>"
        elif attn_1_2_output in {"4", "2", "</s>"}:
            return attn_0_1_output == ""
        elif attn_1_2_output in {"<s>"}:
            return attn_0_1_output == "</s>"

    num_attn_2_3_pattern = select(attn_0_1_outputs, attn_1_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_0_output, attn_0_1_output):
        key = (attn_1_0_output, attn_0_1_output)
        if key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "<s>"),
        }:
            return 7
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "<s>"),
        }:
            return 12
        elif key in {("3", "</s>"), ("</s>", "3")}:
            return 2
        elif key in {("</s>", "</s>"), ("</s>", "<s>"), ("<s>", "</s>")}:
            return 15
        elif key in {("4", "</s>"), ("</s>", "4")}:
            return 3
        elif key in {("1", "</s>"), ("</s>", "1")}:
            return 10
        return 5

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 6

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_2_outputs]
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


print(run(["<s>", "0", "4", "3", "3", "0", "1", "1", "4", "2", "</s>"]))
