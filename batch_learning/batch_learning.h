#pragma once

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
//! Calculate transpose of matrix using CUDA
void transposeCUDA(float *d_in, float *d_out, 
                   unsigned int numInRows, unsigned int numInCols);


//! Apply fixed rate learning to dense weights
void fixedRateLearningCUDA(float *d_DeltaG, float *d_G, 
                           unsigned int numRows, unsigned int numCols, 
                           float learningRate);

//! Apply fixed rate learning to dense weights and then transfer to transpose
void fixedRateLearningTransposeCUDA(float *d_DeltaGIn, float *d_GIn, float *d_GOut, 
                                    unsigned int numInRows, unsigned int numInCols, 
                                    float learningRate);

//! Apply Adam optimizer to dense weights
void adamOptimizerCUDA(float *d_DeltaG, float *d_M, float *d_V, float *d_G, 
                       unsigned int numRows, unsigned int numCols, unsigned int t, 
                       float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, 
                       float epsilon = 1E-8);

//! Apply Adam optimizer to dense weights and then transfer to transpose
void adamOptimizerTransposeCUDA(float *d_DeltaGIn, float *d_MIn, float *d_VIn, float *d_GIn, 
                                float *d_GOut, unsigned int numInRows, unsigned int numInCols, 
                                unsigned int t, float alpha = 0.001, float beta1 = 0.9, 
                                float beta2 = 0.999, float epsilon = 1E-8);

//! Apply RMaxProp to dense weights
void rMaxPropCUDA(float *d_M, float *d_Upsilon, float *d_G,
                  unsigned int numRows, unsigned int numCols,
                  float updateTime, float tauRMS, float r0, float epsilon, float wMin, float wMax);

//! Apply RMaxProp to dense weights and then transfer to transpose
void rMaxPropTransposeCUDA(float *d_MIn, float *d_UpsilonIn, float *d_GIn,
                           float *d_GOut, unsigned int numInRows, unsigned int numInCols,
                           float updateTime, float tauRMS, float r0, float epsilon, float wMin, float wMax);
}
