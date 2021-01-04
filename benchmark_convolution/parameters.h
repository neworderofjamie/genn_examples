//#define SYNAPSE_MATRIX_CONNECTIVITY_PROCEDURAL
//#define SYNAPSE_MATRIX_WEIGHT_PROCEDURAL

#define SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
#define SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_DENSE
    #ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::DENSE_INDIVIDUALG
    #elif defined(SYNAPSE_MATRIX_WEIGHT_PROCEDURAL)
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::DENSE_PROCEDURALG
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

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_PROCEDURAL
    #ifdef SYNAPSE_MATRIX_WEIGHT_PROCEDURAL
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::PROCEDURAL_PROCEDURALG
    #else
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::PROCEDURAL_GLOBALG
    #endif
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_PROCEDURAL

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_BITMASK
    #ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        #error Bitmask connectivity only supported with global weights
    #else
        #define SYNAPSE_MATRIX_TYPE SynapseMatrixType::BITMASK_GLOBALG
    #endif

#endif  // SYNAPSE_MATRIX_CONNECTIVITY_BITMASK


namespace Parameters
{
    constexpr unsigned int numNeurons = 80000;
    constexpr double connectionProbability = 0.1;
    constexpr double inputRate = 10.0;
}
