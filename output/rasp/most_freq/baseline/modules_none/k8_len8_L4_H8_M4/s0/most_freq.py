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
        "output\rasp\most_freq\baseline\modules_none\k8_len8_L4_H8_M4\s0\most_freq_weights.csv",
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
        if q_position in {0, 7}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 6}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 6

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 6

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"<s>", "0"}:
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

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 7}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {2}:
            return token == "<pad>"
        elif position in {3}:
            return token == "3"
        elif position in {4, 5}:
            return token == "1"
        elif position in {6}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, num_var0_embeddings)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 0

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, num_var0_embeddings)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 7}:
            return token == "1"
        elif position in {1}:
            return token == "2"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "0"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, num_var0_embeddings)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 6}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 0

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
            ("0", 6),
            ("0", 7),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 7
        elif key in {
            ("0", 0),
            ("1", 0),
            ("2", 0),
            ("3", 0),
            ("4", 0),
            ("5", 0),
            ("<s>", 0),
        }:
            return 3
        elif key in {("0", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("<s>", 1)}:
            return 6
        elif key in {("5", 1)}:
            return 4
        return 0

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 0),
            ("0", 6),
            ("0", 7),
            ("1", 0),
            ("1", 6),
            ("1", 7),
            ("2", 0),
            ("2", 6),
            ("2", 7),
            ("3", 0),
            ("3", 6),
            ("3", 7),
            ("4", 0),
            ("4", 6),
            ("4", 7),
            ("5", 0),
            ("5", 6),
            ("5", 7),
            ("<s>", 0),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 5
        elif key in {("1", 5), ("2", 5), ("3", 5), ("4", 5), ("5", 5), ("<s>", 5)}:
            return 2
        elif key in {("2", 3), ("2", 4)}:
            return 4
        elif key in {("2", 2)}:
            return 7
        return 0

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_var0_embedding, num_attn_0_1_output):
        key = (num_var0_embedding, num_attn_0_1_output)
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1) for k0, k1 in zip(num_var0_embeddings, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_var0_embedding, num_attn_0_1_output):
        key = (num_var0_embedding, num_attn_0_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
        }:
            return 6
        elif key in {(3, 0), (4, 0), (5, 0), (5, 1), (6, 1), (7, 1), (7, 2)}:
            return 2
        elif key in {(6, 0), (7, 0)}:
            return 4
        return 7

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(num_var0_embeddings, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_num_mlp_0_1_output, k_num_mlp_0_1_output):
        if q_num_mlp_0_1_output in {0}:
            return k_num_mlp_0_1_output == 7
        elif q_num_mlp_0_1_output in {1, 2, 6}:
            return k_num_mlp_0_1_output == 6
        elif q_num_mlp_0_1_output in {3}:
            return k_num_mlp_0_1_output == 2
        elif q_num_mlp_0_1_output in {4, 5}:
            return k_num_mlp_0_1_output == 4
        elif q_num_mlp_0_1_output in {7}:
            return k_num_mlp_0_1_output == 0

    attn_1_0_pattern = select_closest(
        num_mlp_0_1_outputs, num_mlp_0_1_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {0, 4, 5, 7}:
            return mlp_0_0_output == 0
        elif mlp_0_1_output in {1}:
            return mlp_0_0_output == 4
        elif mlp_0_1_output in {2, 6}:
            return mlp_0_0_output == 7
        elif mlp_0_1_output in {3}:
            return mlp_0_0_output == 6

    attn_1_1_pattern = select_closest(mlp_0_0_outputs, mlp_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {0, 4, 6}:
            return mlp_0_1_output == 0
        elif num_mlp_0_1_output in {1}:
            return mlp_0_1_output == 7
        elif num_mlp_0_1_output in {2}:
            return mlp_0_1_output == 6
        elif num_mlp_0_1_output in {3}:
            return mlp_0_1_output == 1
        elif num_mlp_0_1_output in {5}:
            return mlp_0_1_output == 5
        elif num_mlp_0_1_output in {7}:
            return mlp_0_1_output == 4

    attn_1_2_pattern = select_closest(
        mlp_0_1_outputs, num_mlp_0_1_outputs, predicate_1_2
    )
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 5, 6}:
            return token == "3"
        elif position in {1, 4}:
            return token == "5"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {7}:
            return token == "4"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_3_output):
        if position in {0, 3, 4, 5, 6, 7}:
            return attn_0_3_output == "1"
        elif position in {1}:
            return attn_0_3_output == "2"
        elif position in {2}:
            return attn_0_3_output == "<s>"

    num_attn_1_0_pattern = select(attn_0_3_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_var0_embeddings)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, token):
        if attn_0_0_output in {"4", "0", "<s>"}:
            return token == "5"
        elif attn_0_0_output in {"3", "1"}:
            return token == "<s>"
        elif attn_0_0_output in {"2"}:
            return token == "<pad>"
        elif attn_0_0_output in {"5"}:
            return token == "4"

    num_attn_1_1_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_var0_embeddings)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 2, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {3, 4, 5, 7}:
            return token == "2"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_var0_embeddings)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 7}:
            return token == "<s>"
        elif mlp_0_0_output in {1}:
            return token == "4"
        elif mlp_0_0_output in {2, 3, 4, 5, 6}:
            return token == "0"

    num_attn_1_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_var0_embeddings)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "<s>"),
            (2, "4"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "<s>"),
        }:
            return 6
        elif key in {
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "4"),
            (4, "<s>"),
            (5, "0"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "<s>"),
        }:
            return 3
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
        }:
            return 7
        elif key in {(3, "5"), (4, "3"), (4, "5"), (5, "1"), (5, "5")}:
            return 2
        elif key in {(2, "0"), (2, "1"), (2, "2"), (2, "3")}:
            return 0
        elif key in {(1, "5"), (2, "5")}:
            return 4
        return 5

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("1", 7),
            ("2", 7),
            ("3", 3),
            ("3", 4),
            ("3", 6),
            ("3", 7),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 4
        elif key in {
            ("1", 2),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("5", 1),
            ("5", 2),
            ("<s>", 2),
        }:
            return 3
        elif key in {("1", 0), ("2", 0), ("3", 0), ("4", 0), ("5", 0), ("<s>", 0)}:
            return 0
        elif key in {("0", 0), ("4", 1)}:
            return 2
        return 1

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 4

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_2_output):
        key = (num_attn_0_1_output, num_attn_1_2_output)
        return 4

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "<s>"
        elif q_token in {"1", "4", "3", "<s>", "2"}:
            return k_token == "0"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"0"}:
            return attn_1_0_output == "2"
        elif attn_1_2_output in {"<s>", "1"}:
            return attn_1_0_output == "4"
        elif attn_1_2_output in {"2"}:
            return attn_1_0_output == "<s>"
        elif attn_1_2_output in {"3", "4"}:
            return attn_1_0_output == "3"
        elif attn_1_2_output in {"5"}:
            return attn_1_0_output == "5"

    attn_2_1_pattern = select_closest(attn_1_0_outputs, attn_1_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 2, 7}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {3}:
            return token == "5"
        elif position in {4, 5, 6}:
            return token == "2"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, mlp_0_0_output):
        if position in {0, 1}:
            return mlp_0_0_output == 6
        elif position in {2, 3}:
            return mlp_0_0_output == 3
        elif position in {4, 5, 6}:
            return mlp_0_0_output == 7
        elif position in {7}:
            return mlp_0_0_output == 4

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_0_output, token):
        if attn_0_0_output in {"1", "4", "<s>", "0", "5"}:
            return token == "2"
        elif attn_0_0_output in {"2"}:
            return token == "<s>"
        elif attn_0_0_output in {"3"}:
            return token == "3"

    num_attn_2_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_var0_embeddings)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_1_output):
        if position in {0}:
            return attn_1_1_output == "<s>"
        elif position in {1}:
            return attn_1_1_output == "5"
        elif position in {2, 3, 4, 5, 6, 7}:
            return attn_1_1_output == "1"

    num_attn_2_1_pattern = select(attn_1_1_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_var0_embeddings)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_0_output, attn_0_3_output):
        if attn_0_0_output in {"1", "4", "3", "<s>", "0", "5"}:
            return attn_0_3_output == "2"
        elif attn_0_0_output in {"2"}:
            return attn_0_3_output == "3"

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, token):
        if attn_0_0_output in {"1", "3", "<s>", "0", "5", "2"}:
            return token == "4"
        elif attn_0_0_output in {"4"}:
            return token == "<s>"

    num_attn_2_3_pattern = select(tokens, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("1", 0),
            ("2", 2),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("<s>", 0),
            ("<s>", 1),
        }:
            return 2
        elif key in {
            ("0", 3),
            ("1", 3),
            ("2", 0),
            ("2", 1),
            ("2", 3),
            ("3", 3),
            ("4", 3),
            ("5", 3),
            ("<s>", 3),
        }:
            return 0
        elif key in {
            ("0", 2),
            ("1", 1),
            ("1", 2),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("5", 2),
            ("<s>", 2),
        }:
            return 5
        elif key in {
            ("0", 4),
            ("1", 4),
            ("2", 4),
            ("4", 4),
            ("5", 1),
            ("5", 4),
            ("<s>", 4),
        }:
            return 3
        elif key in {("3", 4)}:
            return 6
        return 4

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_1_3_output):
        key = (attn_2_1_output, attn_1_3_output)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("1", "5"),
            ("2", "3"),
            ("3", "1"),
            ("3", "3"),
            ("5", "3"),
            ("5", "5"),
            ("<s>", "3"),
            ("<s>", "5"),
        }:
            return 5
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("4", "1"),
            ("4", "4"),
            ("5", "4"),
            ("<s>", "4"),
        }:
            return 4
        elif key in {
            ("0", "2"),
            ("1", "2"),
            ("2", "2"),
            ("3", "0"),
            ("3", "2"),
            ("3", "<s>"),
            ("<s>", "2"),
        }:
            return 3
        elif key in {
            ("0", "5"),
            ("2", "5"),
            ("3", "4"),
            ("3", "5"),
            ("4", "3"),
            ("4", "5"),
        }:
            return 6
        elif key in {("4", "2")}:
            return 2
        return 1

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_0_output, num_attn_2_1_output):
        key = (num_attn_0_0_output, num_attn_2_1_output)
        return 6

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_0_3_output):
        key = (num_attn_2_1_output, num_attn_0_3_output)
        return 2

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # attn_3_0 ####################################################
    def predicate_3_0(position, mlp_0_0_output):
        if position in {0, 2, 3}:
            return mlp_0_0_output == 6
        elif position in {1, 7}:
            return mlp_0_0_output == 7
        elif position in {4, 5}:
            return mlp_0_0_output == 4
        elif position in {6}:
            return mlp_0_0_output == 5

    attn_3_0_pattern = select_closest(mlp_0_0_outputs, positions, predicate_3_0)
    attn_3_0_outputs = aggregate(attn_3_0_pattern, mlp_1_0_outputs)
    attn_3_0_output_scores = classifier_weights.loc[
        [("attn_3_0_outputs", str(v)) for v in attn_3_0_outputs]
    ]

    # attn_3_1 ####################################################
    def predicate_3_1(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"3", "5", "0", "<s>"}:
            return mlp_0_0_output == 7
        elif attn_0_0_output in {"1"}:
            return mlp_0_0_output == 6
        elif attn_0_0_output in {"2", "4"}:
            return mlp_0_0_output == 5

    attn_3_1_pattern = select_closest(mlp_0_0_outputs, attn_0_0_outputs, predicate_3_1)
    attn_3_1_outputs = aggregate(attn_3_1_pattern, num_mlp_1_0_outputs)
    attn_3_1_output_scores = classifier_weights.loc[
        [("attn_3_1_outputs", str(v)) for v in attn_3_1_outputs]
    ]

    # attn_3_2 ####################################################
    def predicate_3_2(mlp_2_0_output, mlp_0_0_output):
        if mlp_2_0_output in {0, 3, 6, 7}:
            return mlp_0_0_output == 7
        elif mlp_2_0_output in {1}:
            return mlp_0_0_output == 5
        elif mlp_2_0_output in {2, 5}:
            return mlp_0_0_output == 3
        elif mlp_2_0_output in {4}:
            return mlp_0_0_output == 6

    attn_3_2_pattern = select_closest(mlp_0_0_outputs, mlp_2_0_outputs, predicate_3_2)
    attn_3_2_outputs = aggregate(attn_3_2_pattern, mlp_2_1_outputs)
    attn_3_2_output_scores = classifier_weights.loc[
        [("attn_3_2_outputs", str(v)) for v in attn_3_2_outputs]
    ]

    # attn_3_3 ####################################################
    def predicate_3_3(token, mlp_0_0_output):
        if token in {"<s>", "0"}:
            return mlp_0_0_output == 7
        elif token in {"2", "1"}:
            return mlp_0_0_output == 1
        elif token in {"3"}:
            return mlp_0_0_output == 6
        elif token in {"4"}:
            return mlp_0_0_output == 0
        elif token in {"5"}:
            return mlp_0_0_output == 4

    attn_3_3_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_3_3)
    attn_3_3_outputs = aggregate(attn_3_3_pattern, mlp_0_0_outputs)
    attn_3_3_output_scores = classifier_weights.loc[
        [("attn_3_3_outputs", str(v)) for v in attn_3_3_outputs]
    ]

    # num_attn_3_0 ####################################################
    def num_predicate_3_0(position, attn_0_3_output):
        if position in {0, 3, 4, 5, 6}:
            return attn_0_3_output == "0"
        elif position in {1}:
            return attn_0_3_output == "1"
        elif position in {2}:
            return attn_0_3_output == "2"
        elif position in {7}:
            return attn_0_3_output == "<pad>"

    num_attn_3_0_pattern = select(attn_0_3_outputs, positions, num_predicate_3_0)
    num_attn_3_0_outputs = aggregate_sum(num_attn_3_0_pattern, num_var0_embeddings)
    num_attn_3_0_output_scores = classifier_weights.loc[
        [("num_attn_3_0_outputs", "_") for v in num_attn_3_0_outputs]
    ].mul(num_attn_3_0_outputs, axis=0)

    # num_attn_3_1 ####################################################
    def num_predicate_3_1(position, attn_1_1_output):
        if position in {0}:
            return attn_1_1_output == "0"
        elif position in {1}:
            return attn_1_1_output == "1"
        elif position in {2}:
            return attn_1_1_output == "<s>"
        elif position in {3, 4, 5}:
            return attn_1_1_output == "2"
        elif position in {6}:
            return attn_1_1_output == "3"
        elif position in {7}:
            return attn_1_1_output == "<pad>"

    num_attn_3_1_pattern = select(attn_1_1_outputs, positions, num_predicate_3_1)
    num_attn_3_1_outputs = aggregate_sum(num_attn_3_1_pattern, num_var0_embeddings)
    num_attn_3_1_output_scores = classifier_weights.loc[
        [("num_attn_3_1_outputs", "_") for v in num_attn_3_1_outputs]
    ].mul(num_attn_3_1_outputs, axis=0)

    # num_attn_3_2 ####################################################
    def num_predicate_3_2(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "0"
        elif attn_0_0_output in {"1", "3", "<s>", "5", "2"}:
            return token == "4"
        elif attn_0_0_output in {"4"}:
            return token == "5"

    num_attn_3_2_pattern = select(tokens, attn_0_0_outputs, num_predicate_3_2)
    num_attn_3_2_outputs = aggregate_sum(num_attn_3_2_pattern, num_attn_1_3_outputs)
    num_attn_3_2_output_scores = classifier_weights.loc[
        [("num_attn_3_2_outputs", "_") for v in num_attn_3_2_outputs]
    ].mul(num_attn_3_2_outputs, axis=0)

    # num_attn_3_3 ####################################################
    def num_predicate_3_3(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 7}:
            return token == "4"
        elif mlp_0_0_output in {1}:
            return token == "2"
        elif mlp_0_0_output in {2, 3}:
            return token == "<s>"
        elif mlp_0_0_output in {4, 5, 6}:
            return token == "0"

    num_attn_3_3_pattern = select(tokens, mlp_0_0_outputs, num_predicate_3_3)
    num_attn_3_3_outputs = aggregate_sum(num_attn_3_3_pattern, num_attn_1_3_outputs)
    num_attn_3_3_output_scores = classifier_weights.loc[
        [("num_attn_3_3_outputs", "_") for v in num_attn_3_3_outputs]
    ].mul(num_attn_3_3_outputs, axis=0)

    # mlp_3_0 #####################################################
    def mlp_3_0(attn_2_1_output, attn_1_1_output):
        key = (attn_2_1_output, attn_1_1_output)
        if key in {
            ("0", "2"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "5"),
            ("2", "<s>"),
            ("5", "2"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 4
        elif key in {("2", "3"), ("2", "4"), ("3", "2"), ("4", "2"), ("<s>", "4")}:
            return 2
        return 0

    mlp_3_0_outputs = [
        mlp_3_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_1_outputs)
    ]
    mlp_3_0_output_scores = classifier_weights.loc[
        [("mlp_3_0_outputs", str(v)) for v in mlp_3_0_outputs]
    ]

    # mlp_3_1 #####################################################
    def mlp_3_1(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            ("0", 1),
            ("0", 2),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("4", 1),
            ("4", 2),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 6
        elif key in {
            ("0", 0),
            ("0", 3),
            ("0", 4),
            ("1", 3),
            ("1", 4),
            ("2", 3),
            ("2", 4),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 1
        elif key in {("3", 3), ("3", 4), ("5", 1), ("5", 2)}:
            return 2
        elif key in {("3", 0), ("3", 2), ("5", 0), ("5", 3), ("5", 4)}:
            return 3
        elif key in {("4", 3), ("4", 4)}:
            return 0
        elif key in {("4", 0)}:
            return 4
        return 5

    mlp_3_1_outputs = [mlp_3_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_3_1_output_scores = classifier_weights.loc[
        [("mlp_3_1_outputs", str(v)) for v in mlp_3_1_outputs]
    ]

    # num_mlp_3_0 #################################################
    def num_mlp_3_0(num_attn_1_0_output, num_attn_2_1_output):
        key = (num_attn_1_0_output, num_attn_2_1_output)
        return 3

    num_mlp_3_0_outputs = [
        num_mlp_3_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_3_0_output_scores = classifier_weights.loc[
        [("num_mlp_3_0_outputs", str(v)) for v in num_mlp_3_0_outputs]
    ]

    # num_mlp_3_1 #################################################
    def num_mlp_3_1(num_attn_2_1_output):
        key = num_attn_2_1_output
        if key in {0}:
            return 7
        return 3

    num_mlp_3_1_outputs = [num_mlp_3_1(k0) for k0 in num_attn_2_1_outputs]
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


print(run(["<s>", "1", "5", "1", "2", "0", "3"]))
