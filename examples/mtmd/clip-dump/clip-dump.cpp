// Encodes an image with the encoder (mmproj) and saves the embeddings in fp32: used to compare the encoder on CPU across
// different configurations (flash attention, threads) and against a reference.
//   llama-clip-dump <mmproj.gguf> <image> <threads> <fa: 0|1> <output.bin> [gpu: 0|1]

#include "clip.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION      // clip.cpp resizes with stbir, in mtmd the implementation is in mtmd-helper
#include "stb/stb_image_resize2.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <mmproj> <image> <threads> <fa 0|1> <output.bin> [gpu 0|1]\n", argv[0]);
        return 2;
    }
    const int n_threads = atoi(argv[3]);
    clip_context_params p = {
        /* use_gpu          */ argc > 6 && atoi(argv[6]) != 0,
        /* verbosity        */ GGML_LOG_LEVEL_INFO,
        /* flash_attn_type  */ atoi(argv[4]) ? CLIP_FLASH_ATTN_TYPE_ENABLED : CLIP_FLASH_ATTN_TYPE_DISABLED,
        /* image_min_tokens */ -1,
        /* image_max_tokens */ -1,
        /* kq_type          */ GGML_TYPE_F32,
    };
    clip_ctx * ctx = clip_init(argv[1], p).ctx_v;
    if (ctx == nullptr) {
        fprintf(stderr, "encoder not loaded\n");
        return 1;
    }
    int nx = 0, ny = 0, nc = 0;
    unsigned char * rgb = stbi_load(argv[2], &nx, &ny, &nc, 3);
    if (rgb == nullptr) {
        fprintf(stderr, "image not read: %s\n", argv[2]);
        return 1;
    }
    clip_image_u8 * u8 = clip_image_u8_init();
    clip_build_img_from_pixels(rgb, nx, ny, u8);
    clip_image_f32_batch * batch = clip_image_f32_batch_init();
    if (!clip_image_preprocess(ctx, u8, batch)) {
        fprintf(stderr, "preprocess failed\n");
        return 1;
    }
    FILE * out = fopen(argv[5], "wb");
    const int n_embd = clip_n_mmproj_embd(ctx);
    for (size_t i = 0; i < clip_image_f32_batch_n_images(batch); ++i) {
        clip_image_f32 * img = clip_image_f32_get_img(batch, (int) i);
        const int n_tok = clip_n_output_tokens(ctx, img);
        std::vector<float> v((size_t) n_tok * n_embd);
        if (!clip_image_encode(ctx, n_threads, img, v.data())) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        double s = 0, s2 = 0;
        for (float f : v) {
            s += f;
            s2 += (double) f * f;
        }
        printf("image %zu: %zux%zu -> %d tokens x %d, mean %.6f rms %.6f, first %.5f %.5f %.5f %.5f\n", i,
               clip_image_f32_batch_nx(batch, (int) i), clip_image_f32_batch_ny(batch, (int) i), n_tok, n_embd,
               s / v.size(), std::sqrt(s2 / v.size()), v[0], v[1], v[2], v[3]);
        fwrite(v.data(), sizeof(float), v.size(), out);
    }
    fclose(out);
    clip_free(ctx);
    return 0;
}
