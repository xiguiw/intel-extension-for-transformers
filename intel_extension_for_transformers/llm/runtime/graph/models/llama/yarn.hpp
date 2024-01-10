/* copy right */

#ifndef __YARN_HPP__
#define __YARN_HPP__

typedef enum {
 RET_SUCCESS       = 0,
 RET_INVALID_INPUT = 1,
 RET_ERROR,
} RET_TYPE;

class yarn {

public:

    RET_TYPE calculate_embbedding_pos(float &m_scale,
                                    float scale, float base, int dim, 
                                    int32_t original_max_position_embeddings=2048,
                                    int32_t max_position_embeddings=2048);
    RET_TYPE yarn_forward(void); 

    float get_scale();

    yarn(int32_t dim, int32_t max_position_embeddings=2048,
            float base=10000.0, float scale=1.0,
            int32_t original_max_position_embeddings=2048,
            float extrapolation_factor=1.0, float attn_factor=1.0,
            float beta_fast=32, float beta_slow=1.0,
            bool finetuned=false); //, device=None):

    ~yarn(); //float *emb_cos, float *emb_sin)

private:
    int32_t dim;
    int32_t max_position_embeddings;
    float base;
    float scale;
    int32_t original_max_position_embeddings;
    float extrapolation_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool finetuned;
    //, device=None):
};

#endif
