CFPQ with bool vectors algorithm implemented with nsparse SpGEMM.

To build SpGEMM:

1) cd <path_to_nsparse>/cuda-c/
2) make spgemm_hash


To run SpGEMM:

./bin/spgemm_hash_s <path_to_grammar> <path_to_matrix_data>
