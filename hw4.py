from Bio import SeqIO
import numpy as np
import pandas as pd


def read_score_matrix(file_path):
    # Read matrix as a DataFrame, ignoring lines starting with "#"
    df = pd.read_csv(file_path, sep="\s+", comment="#", index_col=0)
    return df


def parse_fasta(file_path):
    # Parse sequences and identifiers from a FASTA file
    records = list(SeqIO.parse(file_path, "fasta"))
    return records[0].id, str(records[0].seq), records[1].id, str(records[1].seq)


# Performs global alignment using affine gap penalties and a substitution matrix
def global_alignment(seq1, seq2, score_matrix, gap_open, gap_extend):
    n, m = len(seq1), len(seq2)
    max_score = 0

    # Initialize three score matrices: M (match), X (gap in seq2), Y (gap in seq1)
    M = np.full((n + 1, m + 1), -float("inf"), dtype=float)
    X = np.full((n + 1, m + 1), -float("inf"), dtype=float)
    Y = np.full((n + 1, m + 1), -float("inf"), dtype=float)

    # Initialize the first row and column of the matrices
    M[0, 0] = 0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
        Y[i, 0] = gap_open + (i - 1) * gap_extend
        M[i, 0] = -float("inf")
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend
        X[0, j] = gap_open + (j - 1) * gap_extend
        M[0, j] = -float("inf")

    # Fill the score matrices using the scoring formula
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = score_matrix.loc[seq1[i - 1], seq2[j - 1]]
            M[i, j] = (
                max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + match_score
            )
            X[i, j] = max(M[i, j - 1] + gap_open, X[i, j - 1] + gap_extend)
            Y[i, j] = max(M[i - 1, j] + gap_open, Y[i - 1, j] + gap_extend)

    # Compute the final alignment score
    max_score = max(M[n, m], X[n, m], Y[n, m])

    # Trace back to reconstruct the optimal alignment
    align1, align2 = "", ""
    i, j = n, m
    current_matrix = np.argmax([M[n, m], X[n, m], Y[n, m]])

    while i > 0 or j > 0:
        if current_matrix == 0:  # From the M matrix
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
            if i > 0 and j > 0:
                current_matrix = np.argmax([M[i, j], X[i, j], Y[i, j]])
        elif current_matrix == 1:  # From the X matrix
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1
            current_matrix = 1 if X[i, j] > M[i, j] + gap_open else 0
        else:  # From the Y matrix
            align1 = seq1[i - 1] + align1
            align2 = "-" + align2
            i -= 1
            current_matrix = 2 if Y[i, j] > M[i, j] + gap_open else 0

    print("Alignment 1: ", align1)
    print("Alignment 2: ", align2)

    return [(align1, align2)], max_score


# Performs local alignment using affine gap penalties and a substitution matrix
def local_alignment(seq1, seq2, score_matrix, gap_open, gap_extend):
    n, m = len(seq1), len(seq2)
    scores = np.zeros((n + 1, m + 1), dtype=int)
    traceback = np.zeros((n + 1, m + 1), dtype=int)
    max_score = 0
    max_position = None

    # Fill the score and traceback matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = (
                scores[i - 1, j - 1] + score_matrix.loc[seq1[i - 1], seq2[j - 1]]
            )
            delete_score = scores[i - 1, j] + (
                gap_open if traceback[i - 1, j] != 1 else gap_extend
            )
            insert_score = scores[i, j - 1] + (
                gap_open if traceback[i, j - 1] != 2 else gap_extend
            )
            scores[i, j] = max(0, match_score, delete_score, insert_score)

            if scores[i, j] > max_score:
                max_score = scores[i, j]
                max_position = (i, j)

            if scores[i, j] == match_score:
                traceback[i, j] = 0
            elif scores[i, j] == delete_score:
                traceback[i, j] = 1
            elif scores[i, j] == insert_score:
                traceback[i, j] = 2

    # Trace back to reconstruct the optimal alignment
    align1, align2 = "", ""
    i, j = max_position
    while i > 0 and j > 0 and scores[i, j] > 0:
        if traceback[i, j] == 0:
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:
            align1 = seq1[i - 1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    print("Alignment 1: ", align1)
    print("Alignment 2: ", align2)

    return [(align1, align2)], max_score


def alignment(input_path, score_path, output_path, aln, gap_open, gap_extend):
    score_matrix = read_score_matrix(score_path)
    id1, seq1, id2, seq2 = parse_fasta(input_path)

    if aln == "global":
        alignments, score = global_alignment(
            seq1, seq2, score_matrix, gap_open, gap_extend
        )
    elif aln == "local":
        alignments, score = local_alignment(
            seq1, seq2, score_matrix, gap_open, gap_extend
        )

    # Write alignment results to output file in FASTA format
    with open(output_path, "w") as f:
        for align1, align2 in alignments:
            f.write(f">{id1}\n{align1}\n")
            f.write(f">{id2}\n{align2}\n")
            f.write(f"Your score:{score}\n")
