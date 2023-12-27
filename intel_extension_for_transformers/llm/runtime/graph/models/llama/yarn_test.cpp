#include <math.h>
#include <stdio.h>

#define TEST_INTERFACE
#ifdef TEST_INTERFACE
    #include "yarn.hpp"
#else
    #include "yarn_scale.cpp"
#endif

int main()
{
    int32_t dim = 128;
    float base = 10000.0;
    int32_t origin_max_position_embeddings = 2048;
    float scale = 64.0; 
    float m_scale = 0.0;

    int32_t max_position_embeddings = (int)scale * origin_max_position_embeddings;

#ifdef TEST_INTERFACE

    yarn_init(dim, max_position_embeddings, base, scale);

    calclate_embbedding_pos(m_scale, scale,
            base, dim, origin_max_position_embeddings, max_position_embeddings);

    printf("m_scale %f\n", m_scale);
    yarn_deinit();

#else

    //float *emb_cos, *emb_sin;
    float low_rot = 32.0, high_rot = 1.0;

    float tmp = yarn_find_correct_dim(low_rot,
                    dim, base, origin_max_position_embeddings);
    int32_t low = floor(tmp);

    printf("hello, tmp %f, low = %d!\n", tmp, low);


    float *inv_freq = new float[dim/2];

    tmp = yarn_find_correct_dim(high_rot,
                    dim, base, origin_max_position_embeddings);
    int32_t high = ceil(tmp);
    printf("hello, tmp %f, high = %d!\n", tmp, high);

    int32_t max, min;
    yarn_find_correction_range(max, min,
                low_rot, high_rot, dim,
                base, origin_max_position_embeddings);
    printf("max, min  %d, %d\n", max, min);

    calclate_inverse_frequency(inv_freq, base, dim, 
        scale, origin_max_position_embeddings);

    yarn_init(dim, max_position_embeddings, base, scale);

    calclate_embbedding_pos(m_scale, scale,
            base, dim, origin_max_position_embeddings, max_position_embeddings);

    printf("m_scale %f\n", m_scale);
    yarn_deinit();

    delete [] inv_freq;
#endif

    return 0;
}
