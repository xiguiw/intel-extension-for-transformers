#include <math.h>
#include <string.h>
#include <stdio.h>
#include "yarn.hpp"

#define PI (3.1415926)
#define EPS (1e-6)


void dump_float_array(float *data, int len)
{
    for (int i = 0; i < len; i++) {
        if (i % 6 == 0)
            printf("\n");
        printf("%9.4e, ", data[i]);
    }
    printf("\n");

    return;
}

/***
# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))
****/
//inline
float yarn_find_correct_dim(int32_t num_rotations, int32_t dim, 
                int64_t base = 10000, int32_t max_pos_embedding =2048)
{
    float corr_dim = (float)dim * log((float)max_pos_embedding / (2.0 * num_rotations * PI))
                    / (2 * log(float(base)));
    return corr_dim;
}

/***
# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case
****/

RET_TYPE yarn_find_correction_range(int32_t &max, int32_t &min,
                float low_rot, float high_rot, int32_t dim,
                float base=10000.0, int32_t max_position_embeddings=2048)
{
    float tmp = yarn_find_correct_dim(low_rot,
                    dim, base, max_position_embeddings);
    min = floor(tmp);
    if (min < 0) min = 0;

    tmp = yarn_find_correct_dim(high_rot,
                    dim, base, max_position_embeddings);
    max = ceil(tmp);
    if (max > dim - 1) max = dim - 1;

    return RET_SUCCESS;
}

/***
def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
****/

/* output array length must be greate than dim */
RET_TYPE yarn_linear_ramp_mask(float *output,
            float min, float max, int dim)
{
    if (output == NULL)
    {
        printf("Invalid input");
        return RET_INVALID_INPUT;
    }

    if ((fabs(max - min)) < EPS)
    {
        max = max + 0.01; //Prevent singularity
    }

    float range = max - min;
    float tmp;
    for (int i = 0; i < dim; i++)
    {
        tmp = (float(i) - min) / range;
        //clamp asm ?
        if (tmp < 0.0)
            output[i] = 0.0;
        else if (tmp > 1.0)
            output[i] = 1.0;
        else 
            output[i] = tmp;
    }

    return RET_SUCCESS;
}

/***
def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

****/
inline float yarn_get_mscale(float scale=1.0)
{
    float ret_val = 1.0;
    if (scale <= 1.0)
        ret_val = 1.0;
    else
        ret_val = 0.1 * log(scale) + 1.0;

    return ret_val;
}


/* create (inverse of) position frequency vector */
RET_TYPE calclate_inverse_frequency(float *inv_freq,
        float base, int dim,
        float scale, int32_t max_position_embeddings=2048)
{
    /* not exposed factor */
    float extrapolation_factor = 1.0;
    float beta_fast = 32.0;
    float beta_slow = 1.0;

    RET_TYPE ret_v = RET_SUCCESS;

    if (inv_freq == NULL) {
        printf("null input pointer\n");
        return RET_INVALID_INPUT;
    }
    if (dim % 2 != 0) {
        printf("dim is not even, error\n");
        return RET_INVALID_INPUT;
    }

    /* create at once at initialized stage */
    /* For GPU Verson, need to store all vector */
    float *pos_freqs = new float[dim/2];
    if (!pos_freqs)
        return RET_INVALID_INPUT;
    float *inv_freq_ext_polat = new float[dim/2]; //let inv_freq_ext_polat, inv_freq_in_polat reuse pos_freq memory?   
    float *inv_freq_in_polat = new float[dim/2];  //TODO: reuse buffer, removed some buffer allocation
    float *inv_freq_mask = new float[dim/2];

    for (int i = 0; i < dim/2; i++) {
        pos_freqs[i] = pow(base, (2.0*i)/dim);
        inv_freq_ext_polat[i] = 1.0 / pos_freqs[i];
        inv_freq_in_polat[i] = inv_freq_ext_polat[i] / scale;
    }

    printf("pos_freqs:");
    dump_float_array(pos_freqs, dim/2);
    printf("inv_freq_ext_polat: ");
    dump_float_array(inv_freq_ext_polat, dim/2);
    printf("inv_freq_in_polat: ");
    dump_float_array(inv_freq_in_polat, dim/2);

    int32_t high, low;
    ret_v = yarn_find_correction_range(high, low,
                beta_fast, beta_slow, dim,
                base, max_position_embeddings);
    if (ret_v != RET_SUCCESS) {
        goto error_process;
    }

    /* inv_freq_mask = 1 - pos_freqs */
    ret_v = yarn_linear_ramp_mask(pos_freqs,
                low, high, dim/2);
    for (int i = 0; i < dim/2; i++) {
        pos_freqs[i] = 1.0 - pos_freqs[i];
    }
    printf("inv_freq_maks: ");
    dump_float_array(pos_freqs, dim/2);
            
    for (int i = 0; i < dim/2; i++) {
        inv_freq[i] = (inv_freq_in_polat[i] * (1.0 - pos_freqs[i])) +
                    inv_freq_ext_polat[i] * pos_freqs[i];
    }

    printf("inv_freq: ");
    dump_float_array(inv_freq, dim/2);

    if (ret_v != RET_SUCCESS) {
        goto error_process;
    }

error_process:
    if (pos_freqs) {
        delete [] pos_freqs;
        pos_freqs = NULL;
    }
    if (inv_freq_ext_polat) {
        delete [] inv_freq_ext_polat;
        inv_freq_ext_polat = NULL;
    }
    if (inv_freq_in_polat) {
        delete [] inv_freq_in_polat;
        inv_freq_in_polat = NULL;
    }
    delete [] inv_freq_mask;

    return ret_v;
}


static float *emb_cos;
static float *emb_sin;

void
yarn_init(int32_t dim, int32_t max_position_embeddings,
            float base, float scale,
            int32_t original_max_position_embeddings,
            float extrapolation_factor, float attn_factor,
            float beta_fast, float beta_slow,
            bool finetuned) //, device=None):
{
    //assert((int)scale * original_max_position_embeddings == max_position_embeddings);

    emb_cos = new float[max_position_embeddings * dim];
    if (emb_cos == NULL) {
        printf("memeory new failed\n");
    }
    emb_sin = new float[max_position_embeddings * dim];
    if (emb_sin == NULL) {
        printf("memeory new failed\n");
    }

    return;
}

/* create (inverse of) position frequency matrix once at initialized stage */
RET_TYPE calclate_embbedding_pos(float &m_scale,
                                    float scale, float base, int dim, 
                                    int32_t original_max_position_embeddings,
                                    int32_t max_position_embeddings)
{
    float attn_factor = 1.0;
    float *inv_freq = new float[dim/2];
    if (!inv_freq) return RET_ERROR;

    int32_t scaled_pos_len = max_position_embeddings;

    int32_t rows = max_position_embeddings;
    int32_t cols = dim / 2;

    float **data_cos = new float*[max_position_embeddings];
    float **data_sin= new float*[max_position_embeddings];
    for (int i = 0; i < max_position_embeddings; i++) {
        data_cos[i] = emb_cos + i * dim;
        data_sin[i] = emb_sin + i * dim;
    }

    m_scale = yarn_get_mscale(scale) * attn_factor;

    
    RET_TYPE ret_val = calclate_inverse_frequency(inv_freq, base, dim, 
                        scale, original_max_position_embeddings);
    if (ret_val != RET_SUCCESS) return RET_ERROR;

    printf("m_scale %f\n", m_scale);
    for (int i = 0; i < max_position_embeddings; i++) {
        /* fill the left half cols */
        for (int k = 0; k < cols; k++) {
            //emb_cos[i*dim + k] = (float)i * inv_freq[k];
            //emb_cos[i*dim + k] = cos(emb_cos[i*dim + k]) * m_scale;
            float tmp = (float)i * inv_freq[k];
            emb_cos[i*dim + k] = cos(tmp) * m_scale;
            emb_sin[i*dim + k] = sin(tmp) * m_scale;
        }
    }
    /* copy the left half to right half cols */
    float *dst = emb_cos + cols;
    float *src = emb_cos;
    for (int i = 0; i < max_position_embeddings; i++) {
        memcpy(dst, src, cols*sizeof(float));
        dst += dim;
        src += dim;
    }
    dst = emb_sin + cols;
    src = emb_sin;
    for (int i = 0; i < max_position_embeddings; i++) {
        memcpy(dst, src, cols*sizeof(float));
        dst += dim;
        src += dim;
    }
    FILE *fp = fopen("sin_emb_bf16.txt", "w");
    for (int i = 0; i < max_position_embeddings * dim; i++) {
        unsigned int *int_p = (unsigned int *)(&emb_sin[i]);
        *int_p = *int_p & 0xFFFF0000;
        fprintf(fp, "%1.6f\n", emb_sin[i]);
    }
    fclose(fp);
   
    printf("%f", emb_cos[1234*dim + 111]);

    delete []inv_freq;
    delete [] data_cos;
    delete [] data_sin;

    return RET_SUCCESS;
}

void
yarn_deinit() //float *emb_cos, float *emb_sin)
{
    /* release resource */
    if (emb_cos != NULL) {
        delete [] emb_cos;
        emb_cos = NULL;
    }
    if (emb_sin != NULL) {
        delete [] emb_sin;
        emb_sin = NULL;
    }

    return;
}

/***
class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation
***/

