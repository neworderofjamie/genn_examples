#define SYNAPSE_MATRIX_CONNECTIVITY_DENSE
#define SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_DENSE
    #ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::DENSE_INDIVIDUALG
    #else
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::DENSE_GLOBALG
    #endif
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_DENSE

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
    #ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::SPARSE_INDIVIDUALG
    #else
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::SPARSE_GLOBALG
    #endif
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_BITMASK
    #ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        #error Bitmask connectivity only supported with global weights
    #else
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::BITMASK_GLOBALG
    #endif

#endif  // SYNAPSE_MATRIX_CONNECTIVITY_BITMASK

namespace Parameters
{
    constexpr unsigned int numPre = 10000;
    constexpr unsigned int numPost = 10000;
    constexpr double connectionProbability = 0.5;
}
