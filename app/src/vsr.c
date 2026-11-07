#include "vsr.h"

#include <assert.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "util/tick.h"

#define FFNV_LOG_FUNC(logctx, msg, ...) LOGW("VSR: " msg, __VA_ARGS__)
#define FFNV_DEBUG_LOG_FUNC(logctx, msg, ...)
#include <ffnvcodec/dynlink_loader.h>

#ifndef GL_TEXTURE_2D
# define GL_TEXTURE_2D 0x0DE1
#endif

#define SC_VSR_EVENT_SLOTS 4
#define SC_VSR_TIMING_SAMPLES 300

// From the NVIDIA Video Effects SDK headers (nvCVStatus.h, nvCVImage.h,
// nvVideoEffects.h, MIT license), declared here because the SDK is loaded at
// runtime
typedef int sc_nvcv_status;
#define SC_NVCV_SUCCESS 0
#define SC_NVCV_ERR_SELECTOR -5
#define SC_NVCV_RGBA 6
#define SC_NVCV_U8 1
#define SC_NVCV_INTERLEAVED 0
#define SC_NVCV_GPU 1

struct sc_nvcv_image {
    unsigned int width;
    unsigned int height;
    signed int pitch;
    int pixel_format;
    int component_type;
    unsigned char pixel_bytes;
    unsigned char component_bytes;
    unsigned char num_components;
    unsigned char planar;
    unsigned char gpu_mem;
    unsigned char colorspace;
    unsigned char reserved[2];
    void *pixels;
    void *delete_ptr;
    void (*delete_proc)(void *p);
    unsigned long long buffer_bytes;
};

static_assert(sizeof(struct sc_nvcv_image) == 64, "unexpected NvCVImage size");

typedef struct sc_nvvfx_object *sc_nvvfx_handle;

struct sc_vfx_api {
    sc_nvcv_status (*GetVersion)(unsigned int *version);
    sc_nvcv_status (*CreateEffect)(const char *code, sc_nvvfx_handle *effect);
    void (*DestroyEffect)(sc_nvvfx_handle effect);
    sc_nvcv_status (*SetU32)(sc_nvvfx_handle effect, const char *name,
                             unsigned int val);
    sc_nvcv_status (*SetImage)(sc_nvvfx_handle effect, const char *name,
                               struct sc_nvcv_image *im);
    sc_nvcv_status (*SetCudaStream)(sc_nvvfx_handle effect, const char *name,
                                    CUstream stream);
    sc_nvcv_status (*Load)(sc_nvvfx_handle effect);
    sc_nvcv_status (*Run)(sc_nvvfx_handle effect, int async);
    sc_nvcv_status (*ImageAlloc)(struct sc_nvcv_image *im, unsigned width,
                                 unsigned height, int format, int type,
                                 unsigned layout, unsigned mem_space,
                                 unsigned alignment);
    void (*ImageDealloc)(struct sc_nvcv_image *im);
    const char *(*GetErrorString)(sc_nvcv_status code);
};

// Not loaded by ffnvcodec
typedef CUresult CUDAAPI sc_cu_event_elapsed_time_fn(float *ms, CUevent start,
                                                      CUevent end);

enum {
    SC_VSR_INPUT,
    SC_VSR_OUTPUT,
};

struct sc_vsr {
    void *lib;
    struct sc_vfx_api vfx;
    CudaFunctions *cu;
    sc_cu_event_elapsed_time_fn *EventElapsedTime; // may be NULL

    CUdevice device;
    CUcontext context; // the primary context, also used by the SDK
    CUstream stream;

    unsigned quality;
    sc_nvvfx_handle effect;
    bool loaded;

    // CUDA buffers read and written by the effect
    struct sc_nvcv_image src;
    struct sc_nvcv_image dst;

    // OpenGL textures registered with CUDA
    CUgraphicsResource resources[2];
    uint32_t textures[2];
    struct sc_size texture_sizes[2];

    CUevent events[SC_VSR_EVENT_SLOTS][2];
    bool events_created;
    unsigned event_next;
    unsigned event_pending;
    double time_ms;
    unsigned time_samples;
};

static bool
sc_vsr_cuda_ok(struct sc_vsr *vsr, CUresult r, const char *what) {
    if (r == CUDA_SUCCESS) {
        return true;
    }

    const char *name = NULL;
    vsr->cu->cuGetErrorName(r, &name);
    LOGE("VSR: %s failed: %s (%d)", what, name ? name : "unknown", (int) r);
    return false;
}

static bool
sc_vsr_nvcv_ok(struct sc_vsr *vsr, sc_nvcv_status s, const char *what) {
    if (s == SC_NVCV_SUCCESS) {
        return true;
    }

    const char *msg = vsr->vfx.GetErrorString ? vsr->vfx.GetErrorString(s)
                                              : NULL;
    LOGE("VSR: %s failed: %s (%d)", what, msg ? msg : "unknown", s);
    return false;
}

static bool
sc_vsr_load_library(struct sc_vsr *vsr) {
    vsr->lib = dlopen("libVideoFX.so", RTLD_NOW | RTLD_GLOBAL);
    if (!vsr->lib) {
        LOGW("VSR unavailable: %s (LD_LIBRARY_PATH must contain the "
             "nvvfx/libs directory of the nvidia-vfx wheel)", dlerror());
        return false;
    }

#define SC_LOAD(field, name) \
    *(void **) &vsr->vfx.field = dlsym(vsr->lib, name); \
    if (!vsr->vfx.field) { \
        LOGW("VSR unavailable: %s not found in libVideoFX.so", name); \
        return false; \
    }
    SC_LOAD(GetVersion, "NvVFX_GetVersion");
    SC_LOAD(CreateEffect, "NvVFX_CreateEffect");
    SC_LOAD(DestroyEffect, "NvVFX_DestroyEffect");
    SC_LOAD(SetU32, "NvVFX_SetU32");
    SC_LOAD(SetImage, "NvVFX_SetImage");
    SC_LOAD(SetCudaStream, "NvVFX_SetCudaStream");
    SC_LOAD(Load, "NvVFX_Load");
    SC_LOAD(Run, "NvVFX_Run");
    // Exported by libNVCVImage.so, a dependency of libVideoFX.so
    SC_LOAD(ImageAlloc, "NvCVImage_Alloc");
    SC_LOAD(ImageDealloc, "NvCVImage_Dealloc");
    SC_LOAD(GetErrorString, "NvCV_GetErrorStringFromCode");
#undef SC_LOAD

    return true;
}

static void
sc_vsr_unregister(struct sc_vsr *vsr, unsigned i) {
    if (vsr->resources[i]) {
        vsr->cu->cuGraphicsUnregisterResource(vsr->resources[i]);
        vsr->resources[i] = NULL;
        vsr->textures[i] = 0;
    }
}

void
sc_vsr_destroy(struct sc_vsr *vsr) {
    if (vsr->cu && vsr->context) {
        CudaFunctions *cu = vsr->cu;
        cu->cuCtxPushCurrent(vsr->context);
        if (vsr->stream) {
            cu->cuStreamSynchronize(vsr->stream);
        }
        sc_vsr_unregister(vsr, SC_VSR_INPUT);
        sc_vsr_unregister(vsr, SC_VSR_OUTPUT);
        if (vsr->src.pixels) {
            vsr->vfx.ImageDealloc(&vsr->src);
        }
        if (vsr->dst.pixels) {
            vsr->vfx.ImageDealloc(&vsr->dst);
        }
        if (vsr->effect) {
            vsr->vfx.DestroyEffect(vsr->effect);
        }
        if (vsr->events_created) {
            for (unsigned i = 0; i < SC_VSR_EVENT_SLOTS; ++i) {
                cu->cuEventDestroy(vsr->events[i][0]);
                cu->cuEventDestroy(vsr->events[i][1]);
            }
        }
        if (vsr->stream) {
            cu->cuStreamDestroy(vsr->stream);
        }
        CUcontext popped;
        cu->cuCtxPopCurrent(&popped);
        cu->cuDevicePrimaryCtxRelease(vsr->device);
    }
    if (vsr->cu) {
        cuda_free_functions(&vsr->cu);
    }
    // The SDK library is not unloaded: its CUDA runtime may still have
    // handlers registered
    free(vsr);
}

static bool
sc_vsr_init_cuda(struct sc_vsr *vsr) {
    if (cuda_load_functions(&vsr->cu, NULL) < 0) {
        LOGW("VSR unavailable: could not load the CUDA driver");
        return false;
    }

    CudaFunctions *cu = vsr->cu;
    *(void **) &vsr->EventElapsedTime =
        dlsym(cu->lib, "cuEventElapsedTime"); // optional, for timing logs

    if (!sc_vsr_cuda_ok(vsr, cu->cuInit(0), "cuInit")) {
        return false;
    }

    // The CUDA device of the GPU that runs the OpenGL renderer
    unsigned count = 0;
    CUdevice devices[4];
    CUresult r = cu->cuGLGetDevices(&count, devices, 4,
                                    CU_GL_DEVICE_LIST_ALL);
    if (r != CUDA_SUCCESS || !count) {
        LOGW("VSR unavailable: the OpenGL renderer does not run on an NVIDIA "
             "GPU");
        return false;
    }
    vsr->device = devices[0];
    if (vsr->device != 0) {
        // The SDK uses the current device of the CUDA runtime (0 by default)
        LOGW("VSR: the renderer GPU is CUDA device %d, the SDK may use "
             "device 0", (int) vsr->device);
    }

    if (!sc_vsr_cuda_ok(vsr, cu->cuDevicePrimaryCtxRetain(&vsr->context,
                                                          vsr->device),
                        "cuDevicePrimaryCtxRetain")) {
        vsr->context = NULL;
        return false;
    }

    return true;
}

static bool
sc_vsr_init_effect(struct sc_vsr *vsr) {
    CudaFunctions *cu = vsr->cu;

    if (!sc_vsr_cuda_ok(vsr, cu->cuStreamCreate(&vsr->stream,
                                                CU_STREAM_NON_BLOCKING),
                        "cuStreamCreate")) {
        vsr->stream = NULL;
        return false;
    }

    if (vsr->EventElapsedTime) {
        bool ok = true;
        for (unsigned i = 0; i < SC_VSR_EVENT_SLOTS && ok; ++i) {
            ok = cu->cuEventCreate(&vsr->events[i][0], 0) == CUDA_SUCCESS
              && cu->cuEventCreate(&vsr->events[i][1], 0) == CUDA_SUCCESS;
        }
        vsr->events_created = ok;
        if (!ok) {
            // Not fatal, the events created are leaked
            vsr->EventElapsedTime = NULL;
        }
    }

    struct sc_vfx_api *vfx = &vsr->vfx;
    if (!sc_vsr_nvcv_ok(vsr, vfx->CreateEffect("VideoSuperRes", &vsr->effect),
                        "creating the VideoSuperRes effect")) {
        vsr->effect = NULL;
        return false;
    }

    // RGB8 (not RGB10A2); older SDKs do not have this selector
    sc_nvcv_status s = vfx->SetU32(vsr->effect, "ImageEncodingMode", 0);
    if (s != SC_NVCV_SUCCESS && s != SC_NVCV_ERR_SELECTOR) {
        sc_vsr_nvcv_ok(vsr, s, "setting ImageEncodingMode");
        return false;
    }

    if (!sc_vsr_nvcv_ok(vsr, vfx->SetU32(vsr->effect, "QualityLevel",
                                         vsr->quality),
                        "setting QualityLevel")) {
        return false;
    }

    if (!sc_vsr_nvcv_ok(vsr, vfx->SetCudaStream(vsr->effect, "CudaStream",
                                                vsr->stream),
                        "setting the CUDA stream")) {
        return false;
    }

    return true;
}

struct sc_vsr *
sc_vsr_new(unsigned quality) {
    struct sc_vsr *vsr = calloc(1, sizeof(*vsr));
    if (!vsr) {
        LOG_OOM();
        return NULL;
    }
    vsr->quality = quality;

    if (!sc_vsr_load_library(vsr) || !sc_vsr_init_cuda(vsr)) {
        sc_vsr_destroy(vsr);
        return NULL;
    }

    CudaFunctions *cu = vsr->cu;
    if (!sc_vsr_cuda_ok(vsr, cu->cuCtxPushCurrent(vsr->context),
                        "cuCtxPushCurrent")) {
        sc_vsr_destroy(vsr);
        return NULL;
    }

    bool ok = sc_vsr_init_effect(vsr);
    CUcontext popped;
    cu->cuCtxPopCurrent(&popped);
    if (!ok) {
        sc_vsr_destroy(vsr);
        return NULL;
    }

    unsigned version = 0;
    vsr->vfx.GetVersion(&version);
    LOGI("Upscaling: NVIDIA Video Super Resolution ready (VFX SDK %u.%u.%u, "
         "quality %u)", version >> 24, (version >> 16) & 0xFF,
         (version >> 8) & 0xFF, quality);
    return vsr;
}

static bool
sc_vsr_register(struct sc_vsr *vsr, unsigned i, uint32_t texture,
                struct sc_size size, unsigned flags) {
    if (vsr->resources[i] && vsr->textures[i] == texture
            && vsr->texture_sizes[i].width == size.width
            && vsr->texture_sizes[i].height == size.height) {
        return true;
    }

    sc_vsr_unregister(vsr, i);

    CUresult r = vsr->cu->cuGraphicsGLRegisterImage(&vsr->resources[i],
                                                    texture, GL_TEXTURE_2D,
                                                    flags);
    if (!sc_vsr_cuda_ok(vsr, r, "registering an OpenGL texture")) {
        vsr->resources[i] = NULL;
        return false;
    }

    vsr->textures[i] = texture;
    vsr->texture_sizes[i] = size;
    return true;
}

void
sc_vsr_release_textures(struct sc_vsr *vsr) {
    if (vsr->resources[0] || vsr->resources[1]) {
        CudaFunctions *cu = vsr->cu;
        cu->cuCtxPushCurrent(vsr->context);
        cu->cuStreamSynchronize(vsr->stream);
        sc_vsr_unregister(vsr, SC_VSR_INPUT);
        sc_vsr_unregister(vsr, SC_VSR_OUTPUT);
        CUcontext popped;
        cu->cuCtxPopCurrent(&popped);
    }
}

static bool
sc_vsr_alloc_image(struct sc_vsr *vsr, struct sc_nvcv_image *im,
                   struct sc_size size) {
    if (im->pixels) {
        vsr->vfx.ImageDealloc(im); // synchronizes with the device
    }
    memset(im, 0, sizeof(*im));
    return sc_vsr_nvcv_ok(vsr, vsr->vfx.ImageAlloc(im, size.width,
                                                   size.height, SC_NVCV_RGBA,
                                                   SC_NVCV_U8,
                                                   SC_NVCV_INTERLEAVED,
                                                   SC_NVCV_GPU, 32),
                          "allocating a CUDA image");
}

// Allocate the CUDA buffers for these sizes, and (re)load the model if needed
static bool
sc_vsr_prepare(struct sc_vsr *vsr, struct sc_size in, struct sc_size out) {
    struct sc_vfx_api *vfx = &vsr->vfx;

    bool input_changed = !vsr->src.pixels || vsr->src.width != in.width
                                          || vsr->src.height != in.height;
    bool output_changed = !vsr->dst.pixels || vsr->dst.width != out.width
                                           || vsr->dst.height != out.height;
    if (!input_changed && !output_changed) {
        return true;
    }

    if (input_changed) {
        if (!sc_vsr_alloc_image(vsr, &vsr->src, in)
                || !sc_vsr_nvcv_ok(vsr, vfx->SetImage(vsr->effect, "SrcImage0",
                                                      &vsr->src),
                                   "setting the input image")) {
            return false;
        }
    }

    if (output_changed) {
        if (!sc_vsr_alloc_image(vsr, &vsr->dst, out)
                || !sc_vsr_nvcv_ok(vsr, vfx->SetImage(vsr->effect, "DstImage0",
                                                      &vsr->dst),
                                   "setting the output image")) {
            return false;
        }
    }

    // The output size can change without reloading the model (verified with
    // VFX SDK 1.3: the result is correct, as its Python bindings assume)
    if (!vsr->loaded || input_changed) {
        sc_tick start = sc_tick_now();
        if (!sc_vsr_nvcv_ok(vsr, vfx->Load(vsr->effect),
                            "loading the model")) {
            return false;
        }
        vsr->loaded = true;
        LOGD("VSR: model loaded in %" PRItick " ms",
             SC_TICK_TO_MS(sc_tick_now() - start));
    }

    LOGD("VSR: %" PRIu16 "x%" PRIu16 " -> %" PRIu16 "x%" PRIu16,
         in.width, in.height, out.width, out.height);
    return true;
}

// Read the GPU time of finished frames (never waits)
static void
sc_vsr_collect_timings(struct sc_vsr *vsr) {
    CudaFunctions *cu = vsr->cu;
    while (vsr->event_pending) {
        unsigned idx = (vsr->event_next + SC_VSR_EVENT_SLOTS
                                        - vsr->event_pending)
                     % SC_VSR_EVENT_SLOTS;
        if (cu->cuEventQuery(vsr->events[idx][1]) != CUDA_SUCCESS) {
            return;
        }

        float ms = 0;
        vsr->EventElapsedTime(&ms, vsr->events[idx][0], vsr->events[idx][1]);
        --vsr->event_pending;

        vsr->time_ms += ms;
        if (++vsr->time_samples == SC_VSR_TIMING_SAMPLES) {
            LOGD("VSR: %.3f ms GPU time per frame (average of %u)",
                 vsr->time_ms / vsr->time_samples, vsr->time_samples);
            vsr->time_ms = 0;
            vsr->time_samples = 0;
        }
    }
}

static bool
sc_vsr_run(struct sc_vsr *vsr, struct sc_size in, struct sc_size out) {
    CudaFunctions *cu = vsr->cu;

    bool timed = false;
    if (vsr->EventElapsedTime) {
        sc_vsr_collect_timings(vsr);
        timed = vsr->event_pending < SC_VSR_EVENT_SLOTS;
        if (timed) {
            cu->cuEventRecord(vsr->events[vsr->event_next][0], vsr->stream);
        }
    }

    // Waits for the OpenGL commands that wrote the input
    if (!sc_vsr_cuda_ok(vsr, cu->cuGraphicsMapResources(2, vsr->resources,
                                                        vsr->stream),
                        "mapping the OpenGL textures")) {
        return false;
    }

    bool ok = false;
    CUarray in_array;
    CUarray out_array;
    if (!sc_vsr_cuda_ok(vsr,
            cu->cuGraphicsSubResourceGetMappedArray(
                &in_array, vsr->resources[SC_VSR_INPUT], 0, 0),
            "getting the input array")
         || !sc_vsr_cuda_ok(vsr,
            cu->cuGraphicsSubResourceGetMappedArray(
                &out_array, vsr->resources[SC_VSR_OUTPUT], 0, 0),
            "getting the output array")) {
        goto unmap;
    }

    CUDA_MEMCPY2D copy_in = {
        .srcMemoryType = CU_MEMORYTYPE_ARRAY,
        .srcArray = in_array,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstDevice = (CUdeviceptr) vsr->src.pixels,
        .dstPitch = vsr->src.pitch,
        .WidthInBytes = (size_t) in.width * 4,
        .Height = in.height,
    };
    if (!sc_vsr_cuda_ok(vsr, cu->cuMemcpy2DAsync(&copy_in, vsr->stream),
                        "copying the input")) {
        goto unmap;
    }

    if (!sc_vsr_nvcv_ok(vsr, vsr->vfx.Run(vsr->effect, 1),
                        "running the effect")) {
        goto unmap;
    }

    CUDA_MEMCPY2D copy_out = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = (CUdeviceptr) vsr->dst.pixels,
        .srcPitch = vsr->dst.pitch,
        .dstMemoryType = CU_MEMORYTYPE_ARRAY,
        .dstArray = out_array,
        .WidthInBytes = (size_t) out.width * 4,
        .Height = out.height,
    };
    ok = sc_vsr_cuda_ok(vsr, cu->cuMemcpy2DAsync(&copy_out, vsr->stream),
                        "copying the output");

unmap:
    // The OpenGL commands issued afterwards wait for the CUDA work
    if (!sc_vsr_cuda_ok(vsr, cu->cuGraphicsUnmapResources(2, vsr->resources,
                                                          vsr->stream),
                        "unmapping the OpenGL textures")) {
        ok = false;
    }

    if (timed) {
        cu->cuEventRecord(vsr->events[vsr->event_next][1], vsr->stream);
        vsr->event_next = (vsr->event_next + 1) % SC_VSR_EVENT_SLOTS;
        ++vsr->event_pending;
    }

    return ok;
}

bool
sc_vsr_process(struct sc_vsr *vsr, uint32_t input, struct sc_size input_size,
               uint32_t output, struct sc_size output_size) {
    CudaFunctions *cu = vsr->cu;
    if (!sc_vsr_cuda_ok(vsr, cu->cuCtxPushCurrent(vsr->context),
                        "cuCtxPushCurrent")) {
        return false;
    }

    bool ok = sc_vsr_register(vsr, SC_VSR_INPUT, input, input_size,
                              CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)
           && sc_vsr_register(vsr, SC_VSR_OUTPUT, output, output_size,
                              CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)
           && sc_vsr_prepare(vsr, input_size, output_size)
           && sc_vsr_run(vsr, input_size, output_size);

    CUcontext popped;
    cu->cuCtxPopCurrent(&popped);
    return ok;
}
