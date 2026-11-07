#ifndef SC_UPSCALER_H
#define SC_UPSCALER_H

#include "common.h"

#include <stdbool.h>
#include <stdint.h>
#include <SDL3/SDL.h>

#include "coords.h"
#include "options.h"

struct sc_upscaler_gl;
struct sc_vsr;

// Upscales video frames on the GPU right before they are drawn (no extra frame
// of latency), with:
//  - NVIDIA Image Scaling (NIS): a directional scaling and adaptive sharpening
//    compute shader;
//  - NVIDIA Video Super Resolution (VSR): an AI upscaler that also removes
//    compression artifacts, run on CUDA (see vsr.h). If it is not available,
//    NIS is used instead.
//
// It reads the NV12 planes of the frame texture directly, so it only works
// with frames imported from VA-API on an OpenGL 4.3+ renderer. Otherwise
// sc_upscaler_process() returns NULL and the frame is drawn as usual.
struct sc_upscaler {
    SDL_Renderer *renderer;
    float sharpness; // 0 to 1, for NIS
    unsigned vsr_quality;

    // allocated on first use, NULL until then or if initialization failed
    struct sc_upscaler_gl *gl;
    bool failed; // do not retry
    bool unsupported_logged;

    struct sc_vsr *vsr; // created on first use
    bool vsr_failed; // use NIS instead

    SDL_Texture *output;
    uint32_t output_id;
    bool output_prepared; // its GL filters are set
    struct sc_size input_size;
    struct sc_size output_size;
    SDL_Colorspace colorspace;
};

// sharpness: 0 to 100 (NIS)
// vsr_quality: the VSR QualityLevel
void
sc_upscaler_init(struct sc_upscaler *up, SDL_Renderer *renderer,
                 uint8_t sharpness, uint8_t vsr_quality);

// The renderer's OpenGL context must be current
void
sc_upscaler_destroy(struct sc_upscaler *up);

// Upscale the NV12 texture `input` (a frame of `input_size`) to `output_size`
// with `mode` (SC_UPSCALE_NIS or SC_UPSCALE_VSR).
//
// Return the upscaled texture (exactly `output_size`, to be drawn 1:1), or
// NULL if the frame cannot be upscaled; the caller then draws `input` as
// usual. Only upscaling is supported: NULL if output_size is not larger than
// input_size in both dimensions.
SDL_Texture *
sc_upscaler_process(struct sc_upscaler *up, enum sc_upscale mode,
                    SDL_Texture *input, struct sc_size input_size,
                    struct sc_size output_size);

#endif
