#ifndef SC_VSR_H
#define SC_VSR_H

#include "common.h"

#include <stdbool.h>
#include <stdint.h>

#include "coords.h"

// NVIDIA Video Super Resolution (the AI upscaler of the NVIDIA Video Effects
// SDK), run on CUDA between two OpenGL textures.
//
// The SDK libraries are loaded at runtime: libVideoFX.so and its dependencies
// must be in the library search path (e.g. LD_LIBRARY_PATH pointing to the
// nvvfx/libs directory of the nvidia-vfx Python wheel). CUDA comes from the
// NVIDIA driver (libcuda.so.1).
//
// All functions must be called with the OpenGL context current.

struct sc_vsr;

// quality: the VSR QualityLevel (see --upscale-vsr-quality)
// Return NULL if VSR is not available (the reason is logged)
struct sc_vsr *
sc_vsr_new(unsigned quality);

void
sc_vsr_destroy(struct sc_vsr *vsr);

// Upscale the GL_RGBA8 texture `input` into the GL_RGBA8 texture `output`
// (both GL_TEXTURE_2D). The OpenGL commands issued afterwards wait for the
// result.
//
// A new input size reloads the model (about 120 ms), a new output size does
// not (window resizing).
bool
sc_vsr_process(struct sc_vsr *vsr, uint32_t input, struct sc_size input_size,
               uint32_t output, struct sc_size output_size);

// Forget the OpenGL textures passed to sc_vsr_process(): must be called before
// deleting them
void
sc_vsr_release_textures(struct sc_vsr *vsr);

#endif
