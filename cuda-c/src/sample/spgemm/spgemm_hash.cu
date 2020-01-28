#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <algorithm>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>
#ifdef FLOAT
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;


// not used in last version
void csr_copy(sfCSR * src, sfCSR * dst) {
    release_csr(*dst);
    dst->M = src->M;
    dst->N = src->N;
    dst->nnz = src->nnz;
    dst->nnz_max = src->nnz_max;

    checkCudaErrors(cudaMalloc((void **)&(dst->d_rpt), sizeof(int) * (dst->M + 1)));
    checkCudaErrors(cudaMalloc((void **)&(dst->d_col), sizeof(int) * dst->nnz));
    checkCudaErrors(cudaMalloc((void **)&(dst->d_val), sizeof(real) * dst->nnz));

    checkCudaErrors(cudaMemcpy(dst->d_rpt, src->d_rpt, sizeof(int) * (src->M + 1), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst->d_col, src->d_col, sizeof(int) * src->nnz, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst->d_val, src->d_val, sizeof(real) * src->nnz, cudaMemcpyDeviceToDevice));
}


// only for debug
__global__ void print_sum(int nnz, real * val) {
    int sumAmount = 0;
    int t;
    for (t = 0; t < nnz; t++) {
        if ((val[t] & 0x1) == 0x1) {
            sumAmount++;
        }
    }
    printf("SumAmountPart: %d\n", sumAmount);
}


// only for debug
__global__ void print_matrix(int sz, int * rpt, int * col, real * val) {
    int i, j, cnt = 0;
    printf("RPT: \n");
    for (i = 1; i <= sz; i++) {
        printf("%d ", rpt[i]);
    }
    printf("\n");

    printf("(Col, VAL)\n");
    for (i = 1; i <= sz; i++) {
        while (cnt < rpt[i]) {
            printf("(%d, %d) ", col[cnt], val[cnt]);
            cnt++;
        }
        printf("\n");
    }
}

__device__ int flagNoChange = true;
__device__ int nnzSum = 0;


__global__ void set_nnz_sum(int * rptC, int sz) {
    rptC[0] = 0;
    int i;
    int sum = 0;
    for (i = 1; i <= sz; i++) {
        sum += rptC[i];
        rptC[i] = sum;
    }
    nnzSum = sum;
}

__global__ void sumSparse_kernel(int sz, int * rptA, int * colA, real * valA, int * rptB, int * colB, real * valB, int * rptC, int * colC, real * valC)
{
    flagNoChange = true;
    int colAcnt;
    int colBcnt;
    int i;
    int idx = threadIdx.x;
    int toThread = sz / 1024;
    toThread = sz % 1024 == 0 ? toThread : toThread + 1;
    int rpt_start_index = idx * toThread;
    int colCcnt = rptC[rpt_start_index];
    int rpt_end_index = (idx + 1) * toThread > sz ? sz : (idx + 1) * toThread;
    for (i = rpt_start_index; i < rpt_end_index; i++) {
        colAcnt = rptA[i];
        colBcnt = rptB[i];

        while (colAcnt < rptA[i + 1] || colBcnt < rptB[i + 1]) {

            if (colAcnt < rptA[i + 1] && valA[colAcnt] == 0) {
                colAcnt++;
                continue;
            }

            if (colBcnt < rptB[i + 1] && valB[colBcnt] == 0) {
                colBcnt++;
                continue;
            }

            // if both matrix are in game
            if (colAcnt < rptA[i + 1] && colBcnt < rptB[i + 1]) {
                if (colA[colAcnt] <= colB[colBcnt]) {
                    colC[colCcnt] = colA[colAcnt];
                    if (colA[colAcnt] == colB[colBcnt]) {
                        valC[colCcnt] = valA[colAcnt] | valB[colBcnt];
                        if (valC[colCcnt] != valA[colAcnt]) {
                            flagNoChange = false;
                        }
                        colBcnt++;
                    } else {
                        valC[colCcnt] = valA[colAcnt];
                    }
                    colCcnt++;
                    colAcnt++;
                } else {
                    colC[colCcnt] = colB[colBcnt];
                    valC[colCcnt] = valB[colBcnt];
                    flagNoChange = false;
                    colCcnt++;
                    colBcnt++;
                }
            } else if (colAcnt < rptA[i + 1]) {
                colC[colCcnt] = colA[colAcnt];
                valC[colCcnt] = valA[colAcnt];
                colCcnt++;
                colAcnt++;
            } else {
                colC[colCcnt] = colB[colBcnt];
                valC[colCcnt] = valB[colBcnt];
                flagNoChange = false;
                colCcnt++;
                colBcnt++;
            }
        }
    }
}


__global__ void precount_kernel(int sz, int * rptA, int * colA, real * valA, int * rptB, int * colB, real * valB, int * rptC) {
    int colAcnt;
    int colBcnt;
    int idx = threadIdx.x;
    int i;
    int counter;
    int toThread = sz / 1024;
    toThread = sz % 1024 == 0 ? toThread : toThread + 1;
    int rpt_start_index = idx * toThread;
    int rpt_end_index = (idx + 1) * toThread > sz ? sz : (idx + 1) * toThread;
    for (i = rpt_start_index; i < rpt_end_index; i++) {
        colAcnt = rptA[i];
        colBcnt = rptB[i];
        counter = 0;

        while (colAcnt < rptA[i + 1] || colBcnt < rptB[i + 1]) {

            if (colAcnt < rptA[i + 1] && valA[colAcnt] == 0) {
                colAcnt++;
                continue;
            }

            if (colBcnt < rptB[i + 1] && valB[colBcnt] == 0) {
                colBcnt++;
                continue;
            }

            counter++;

            // if both matrix are in game
            if (colAcnt < rptA[i + 1] && colBcnt < rptB[i + 1]) {
                if (colA[colAcnt] <= colB[colBcnt]) {
                    if (colA[colAcnt] == colB[colBcnt]) {
                        colBcnt++;
                    }
                    colAcnt++;
                } else {
                    colBcnt++;
                }
            } else if (colAcnt < rptA[i + 1]) {
                colAcnt++;
            } else {
                colBcnt++;
            }
        }

        rptC[i + 1] = counter;
    }
}


// C = A | B and check if C == A (if they are equal flagNoChange will be true)
void sumSparse(sfCSR * a, sfCSR * b, sfCSR * c) {
    int gridAmount = 1024;
    precount_kernel<<<1, gridAmount>>>(a->M, a->d_rpt, a->d_col, a->d_val, b->d_rpt, b->d_col, b->d_val, c->d_rpt);
    cudaThreadSynchronize();
    int nnzS = -1;
    cudaError_t result = cudaGetLastError();
    set_nnz_sum<<<1, 1>>>(c->d_rpt, c->M); // always in one thread!!!!
    cudaThreadSynchronize();
    result = cudaGetLastError();
    sumSparse_kernel<<<1, gridAmount>>>(a->M, a->d_rpt, a->d_col, a->d_val, b->d_rpt, b->d_col, b->d_val, c->d_rpt, c->d_col, c->d_val);
    cudaThreadSynchronize();
}
#endif


void spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c, int grSize, unsigned short int * grBody, unsigned int * grTail)
{

    int i;

    cudaEvent_t event[2];
    float msec, msec_sum;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    csr_memcpy(a);
    csr_memcpy(b);

    /* Execution of SpGEMM on Device */
    unsigned int ave_msec_sum = 0, ave_msec_copy = 0;
    cudaEventRecord(event[0], 0);
#ifdef FLOAT
    int noChange = 0;
    bool first = true;
    int nnzS = 0;
    int u = 0;
    while (!noChange) {
        u++;
        if (first) {
            first = false;
#endif
            spgemm_kernel_hash(a, a, c, grSize, grBody, grTail, true);
#ifdef FLOAT
        }
        else {
            release_csr(*c);
            spgemm_kernel_hash(a, a, c, grSize, grBody, grTail, false);
        }

        cudaThreadSynchronize();
        cudaFree(b->d_col);
        cudaFree(b->d_val);
        checkCudaErrors(cudaMalloc((void **)&(b->d_col), sizeof(int) * (a->nnz + c->nnz)));
        checkCudaErrors(cudaMalloc((void **)&(b->d_val), sizeof(real) * (a->nnz + c->nnz)));

        high_resolution_clock::time_point begin_sum_time = high_resolution_clock::now();
        sumSparse(a, c, b);
        high_resolution_clock::time_point end_sum_time = high_resolution_clock::now();
        milliseconds elapsed_secs = duration_cast<milliseconds>(end_sum_time - begin_sum_time);
        ave_msec_sum += static_cast<unsigned int>(elapsed_secs.count());

        high_resolution_clock::time_point begin_copy_time = high_resolution_clock::now();
        cudaMemcpyFromSymbol(&nnzS, nnzSum, sizeof(int), 0, cudaMemcpyDeviceToHost);
        b->nnz = nnzS;
        sfCSR * tmp = b;
        b = a;
        a = tmp;
        high_resolution_clock::time_point end_copy_time = high_resolution_clock::now();

        milliseconds elapsed_secs_c = duration_cast<milliseconds>(end_copy_time - begin_copy_time);
        ave_msec_copy += static_cast<unsigned int>(elapsed_secs_c.count());

        cudaMemcpyFromSymbol(&noChange, flagNoChange, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
    }
    printf("Average 'in sum' time: %d %d %f\n", ave_msec_sum, u, ave_msec_sum / (double)u);
    printf("Average 'in copy' time: %d %d %f\n", ave_msec_copy, u, ave_msec_copy / (double)u);
    printf("Amount of mults: %d\n", u);
#endif
    cudaEventRecord(event[1], 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);

    printf("Total algorithm time: %f[ms]\n", msec);

#ifdef FLOAT
    c = a;
#endif

    csr_memcpyDtH(c);
#ifdef FLOAT
    int t, sumAmount = 0;
    for (t = 0; t < c->nnz; t++) {
        if ((c->val[t] & 0x1) == 0x1) {
            sumAmount++;
        }
    }
    printf("SumAmount: %d\n", sumAmount);
#endif
#ifndef FLOAT
    release_csr(*c);
#endif
    release_csr(*a);
    release_csr(*b);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

#ifdef FLOAT
unsigned char toBoolVector(unsigned int number) {
    return ((unsigned short)0x1) << number;
}

std::unordered_map<std::string, std::vector<int> > terminal_to_nonterminals;

int load_grammar(const std::string & grammar_filename, unsigned short ** grammar_body, unsigned int ** grammar_tail) {
    std::ifstream chomsky_stream(grammar_filename);

    std::string line, tmp;
    unsigned int nonterminals_count = 0;

    std::map<std::string, unsigned int> nonterminal_to_index;
    std::vector<unsigned int> epsilon_nonterminals;
    std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int> > > rules;

    while (getline(chomsky_stream, line)) {
        vector <std::string> terms;
        istringstream iss(line);
        while (iss >> tmp) {
            terms.push_back(tmp);
        }
        if (!nonterminal_to_index.count(terms[0])) {
            nonterminal_to_index[terms[0]] = nonterminals_count++;
        }
        if (terms.size() == 1) {
            epsilon_nonterminals.push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 2) {
            if (!terminal_to_nonterminals.count(terms[1])) {
                terminal_to_nonterminals[terms[1]] = {};
            }
            terminal_to_nonterminals[terms[1]].push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 3) {
            if (!nonterminal_to_index.count(terms[1])) {
                nonterminal_to_index[terms[1]] = nonterminals_count++;
            }
            if (!nonterminal_to_index.count(terms[2])) {
                nonterminal_to_index[terms[2]] = nonterminals_count++;
            }
            rules.push_back(
                    {nonterminal_to_index[terms[0]], {nonterminal_to_index[terms[1]], nonterminal_to_index[terms[2]]}});
        }
    }
    chomsky_stream.close();

    int grammar_size = rules.size();
    *grammar_body = (unsigned short *)calloc(grammar_size, sizeof(unsigned short));
    *grammar_tail = (unsigned int *)calloc(grammar_size, sizeof(unsigned int));

    for (size_t i = 0; i < rules.size(); i++) {
        (*grammar_body)[i] = toBoolVector(rules[i].first);
        (*grammar_tail)[i] = (((unsigned int)toBoolVector(rules[i].second.first)) << 16) | (unsigned int)toBoolVector(rules[i].second.second);
    }

    return grammar_size;
}

void load_graph(const std::string & graph_filename, sfCSR * matrix) {
    std::vector<std::pair<std::string, std::pair<unsigned int, unsigned int> > > edges;
    unsigned int vertices_count = 0;

    std::ifstream graph_stream(graph_filename);
    unsigned int from, to;
    std::string terminal;
    while (graph_stream >> from >> terminal >> to) {
        edges.push_back({terminal, {from, to}});
        vertices_count = max(vertices_count, max(from, to) + 1);
    }
    graph_stream.close();

    matrix->nnz = 0;
    matrix->M = vertices_count;
    matrix->N = vertices_count;
    int * col_coo = (int *)malloc(sizeof(int) * edges.size());
    int * row_coo = (int *)malloc(sizeof(int) * edges.size());
    real * val_coo = (real *)malloc(sizeof(real) * edges.size());
    vector<pair<pair<int, int>, unsigned short> > tempVec;

    for (auto & edge : edges) {
        if (terminal_to_nonterminals.count(edge.first) == 0) {
            continue;
        }
        auto nonterminals = terminal_to_nonterminals.at(edge.first);
        unsigned short bool_vector = 0;
        for (auto nonterminal : nonterminals) {
            bool_vector |= toBoolVector(nonterminal);
        }

        tempVec.push_back({{edge.second.first, edge.second.second}, bool_vector});
    }

    sort(tempVec.begin(), tempVec.end(),
    [](const pair<pair<int, int>, unsigned short> & a, const pair<pair<int, int>, unsigned short> & b) -> bool
    {
        return a.first.second < b.first.second;
    });

    for (int i = 0; i < tempVec.size(); i++) {
        row_coo[i] = tempVec[i].first.first;
        col_coo[i] = tempVec[i].first.second;
        val_coo[i] = tempVec[i].second;
    }


    /* Count the number of non-zero in each row */
    int num = tempVec.size();
    int i;
    int * nnz_num = (int *)malloc(sizeof(int) * matrix->M);
    for (i = 0; i < matrix->M; i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
    }

    for (i = 0; i < matrix->M; i++) {
        matrix->nnz += nnz_num[i];
    }

    // Store matrix in CSR format
    /* Allocation of rpt, col, val */
    int * rpt_ = (int *)malloc(sizeof(int) * (matrix->M + 1));
    int * col_ = (int *)malloc(sizeof(int) * matrix->nnz);
    real * val_ = (real *)malloc(sizeof(real) * matrix->nnz);

    int offset = 0;
    matrix->nnz_max = 0;
    for (i = 0; i < matrix->M; i++) {
        rpt_[i] = offset; // looks like we have amount of not null in rows before this row
        offset += nnz_num[i];
        if(matrix->nnz_max < nnz_num[i]){
            matrix->nnz_max = nnz_num[i];
        }
    }
    rpt_[matrix->M] = offset; // amount of all not null

    int * each_row_index = (int *)malloc(sizeof(int) * matrix->M);
    for (i = 0; i < matrix->M; i++) {
        each_row_index[i] = 0;
    }

    for (i = 0; i < num; i++) {
        col_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        val_[rpt_[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    }

    matrix->rpt = rpt_;
    matrix->col = col_;
    matrix->val = val_;

    free(nnz_num);
    free(row_coo);
    free(col_coo);
    free(val_coo);
    free(each_row_index);
}
#endif


/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
  
    /* Set CSR reading from MM file */
    int grammar_size;
    unsigned short * grammar_body;
    unsigned int * grammar_tail;
#ifndef FLOAT
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
#endif

#ifdef FLOAT
    grammar_size = load_grammar(argv[1], &grammar_body, &grammar_tail);
    printf("Grammar:\n");
    int q;
    printf("Grammar size: %d\n", grammar_size);
    for (q = 0; q < grammar_size; q++) {
        printf("%p -> %p\n", grammar_body[q], grammar_tail[q]);
    }
    load_graph(argv[2], &mat_a);
    load_graph(argv[2], &mat_b);
    printf("NNZ_A: %d, NNZ_B: %d SIZE_A: %d\n", mat_a.nnz, mat_b.nnz, mat_a.M);
#endif
    spgemm_csr(&mat_a, &mat_b, &mat_c, grammar_size, grammar_body, grammar_tail);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
#ifndef FLOAT
    release_cpu_csr(mat_c);
#endif
    
    return 0;
}
