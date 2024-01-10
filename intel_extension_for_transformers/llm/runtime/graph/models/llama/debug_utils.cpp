#include "../../core/ne_layers.h"

static int ne_nrows(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

void yarn_dump_tensor(const struct ne_tensor* src0) {
  //struct ne_compute_params* params
  //if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
  //  return;
  //}
  // make sure each file different as the multi-node will use the file
  int random_num = rand();
  char file_name[255];
  sprintf(file_name, "%s_%d.txt", src0->name, random_num);
  FILE* file = fopen(file_name, "w");
  if (file == NULL) {
    NE_ASSERT(false);
  }

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];
  const int64_t nr = ne_nrows(src0);

  fprintf(file, "Total element is %ld\n", ne_nelements(src0));
  fprintf(file, "ne[0] size is %ld ne[1] size is %ld ne[2] size is %ld ne[3] size is %ld \n", ne00, ne01, ne02, ne03);
  switch (src0->type) {
    case NE_TYPE_F32: {
      for (int64_t ir = 0; ir < nr; ++ir) {
        for (int64_t i0 = 0; i0 < ne00; ++i0) {
          fprintf(file, "%.4f ", *(((float*)src0->data) + i0 + ne00 * ir));
        }
        fprintf(file, "\n");
      }
    } break;
    case NE_TYPE_F16: {
      for (int64_t ir = 0; ir < nr; ++ir) {
        for (int64_t i0 = 0; i0 < ne00; ++i0) {
          float src_data = (float)(*(((ne_fp16_t*)src0->data) + i0 + ne00 * ir));
          fprintf(file, "%f ", src_data);
        }
        fprintf(file, "\n");
      }
    } break;
    case NE_TYPE_Q4_0: {
      for (int64_t ir = 0; ir < nr; ++ir) {
        for (int64_t i0 = 0; i0 < ne00 / 2; ++i0) {
          int high_half = *(((char*)src0->data) + i0 + ne00 / 2 * ir) >> 4;
          int low_half = *(((char*)src0->data) + i0 + ne00 / 2 * ir) & 0x0F;
          fprintf(file, "%d %d ", high_half, low_half);
        }
        fprintf(file, "\n");
      }
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
  fclose(file);
  printf("NE assert\n");
  //NE_ASSERT(false);
}
