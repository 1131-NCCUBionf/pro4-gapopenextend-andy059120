# pro4.GapOpenExtend
<your name + student ID>
## Description

* Write a Python script to perform pairwise alignment with two-type gap penalties.
* Creating your script, i.e. hw4.py.
* In this program, the library Biostrings is only used to parse fasta files.
* You should write a program with a function named alignment, ie.
```
def alignment(input_path, score_path, output_path, aln, gap_open, gap_extend):
    .
    .
    .
    .
```
* If there is more than one local alignment with the same highest score, you should output one with longest alignment length

## Files

* hw4_ref.py: You can start from this reference code and try to write your comment in English.
* pam100.txt
* pam250.txt
* test.fasta
* result_global.fasta: An example of a global output file in FASTA format.
* result_local.fasta: An example of  a local output file in FASTA format.

## Parameters

* input: fasta file (ex. test.fasta)
* score: score file (ex. pam250.txt)
* aln: global|local
* gap_open: open gap score
* gap_extend: extend gap score
* output: fasta file

## Command

I'd like you to please be sure to execute your code with the following command.

```Python
alignment("test.fasta", "pam250.txt", "global", "result_global.fasta", -10, -2), threshold = 45
alignment("test.fasta", "pam250.txt", "local", "result_local.fasta", -10, -2), threshold = 59
```

## Evaluation

10 testing data(5 public, 5 private)

Each testing data has a threshold score; output alignment score >= threshold gets 10 points.

### Penalty

* High code similarity to others: YOUR SCORE = 0
* the number of submission > 50: -5


## References
Please provide the code along with its reference. For example, you can cite it as: ```# ChatGPT, respond to “your prompt,” on February 16, 2023```. Below is an example of a reference format summarizing the use of ChatGPT for R programming

>You are the R Language expert.
>Please help me to write a function called “k_fold”.
>Using a given dataset to train the random forest model, and using the k-fold cross-validation to evaluate the best model parameters. Here is the instruction for the function requirements:\
>Function name: k_fold\
>Function parameters:
