#include "upscaler.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL_opengl.h>

#include "util/log.h"
#ifdef HAVE_VSR
# include "vsr.h"
#endif

#include "nis/nis_coef.h"
#include "nis_scaler_glsl.inc" // sc_nis_scaler_glsl, from nis/NIS_Scaler.h

// Dispatch layout recommended by the NIS SDK for upscaling on NVIDIA GPUs
#define SC_NIS_BLOCK_WIDTH 32
#define SC_NIS_BLOCK_HEIGHT 24
#define SC_NIS_THREAD_GROUP_SIZE 128

// GPU time queries in flight (results are read a few frames later, never
// waited for)
#define SC_UPSCALER_QUERY_COUNT 4
#define SC_UPSCALER_TIMING_SAMPLES 300

#define SC_STR_(x) #x
#define SC_STR(x) SC_STR_(x)

static const char sc_upscaler_shader_prefix[] =
    "#version 430\n"
    "#define NIS_GLSL 1\n"
    "#define NIS_GLSL_OPENGL 1\n"
    "#define NIS_SCALER 1\n"
    "#define NIS_HDR_MODE 0\n"
    "#define NIS_NV12_SUPPORT 1\n"
    "#define NIS_USE_HALF_PRECISION 0\n"
    "#define NIS_BLOCK_WIDTH " SC_STR(SC_NIS_BLOCK_WIDTH) "\n"
    "#define NIS_BLOCK_HEIGHT " SC_STR(SC_NIS_BLOCK_HEIGHT) "\n"
    "#define NIS_THREAD_GROUP_SIZE " SC_STR(SC_NIS_THREAD_GROUP_SIZE) "\n"
    "layout(std140, binding = 0) uniform const_buffer {\n"
    "    float kDetectRatio;\n"
    "    float kDetectThres;\n"
    "    float kMinContrastRatio;\n"
    "    float kRatioNorm;\n"
    "    float kContrastBoost;\n"
    "    float kEps;\n"
    "    float kSharpStartY;\n"
    "    float kSharpScaleY;\n"
    "    float kSharpStrengthMin;\n"
    "    float kSharpStrengthScale;\n"
    "    float kSharpLimitMin;\n"
    "    float kSharpLimitScale;\n"
    "    float kScaleX;\n"
    "    float kScaleY;\n"
    "    float kDstNormX;\n"
    "    float kDstNormY;\n"
    "    float kSrcNormX;\n"
    "    float kSrcNormY;\n"
    "    uint kInputViewportOriginX;\n"
    "    uint kInputViewportOriginY;\n"
    "    uint kInputViewportWidth;\n"
    "    uint kInputViewportHeight;\n"
    "    uint kOutputViewportOriginX;\n"
    "    uint kOutputViewportOriginY;\n"
    "    uint kOutputViewportWidth;\n"
    "    uint kOutputViewportHeight;\n"
    "    float reserved0;\n"
    "    float reserved1;\n"
    "};\n"
    "layout(binding = 0) uniform sampler2D in_texture_y;\n"
    "layout(binding = 1) uniform sampler2D in_texture_uv;\n"
    "layout(binding = 2) uniform sampler2D coef_scaler;\n"
    "layout(binding = 3) uniform sampler2D coef_usm;\n"
    "layout(rgba8, binding = 0) uniform writeonly image2D out_texture;\n"
    "uniform mat3 sc_yuv_matrix;\n"
    "uniform vec3 sc_yuv_offset;\n"
    "vec3 sc_yuv_to_rgb(vec3 yuv) {\n"
    "    return clamp(sc_yuv_matrix * (yuv - sc_yuv_offset), 0.0, 1.0);\n"
    "}\n"
    "#define NIS_YUV_TO_RGB(yuv) sc_yuv_to_rgb(yuv)\n"
    // Report errors in NIS_Scaler.h with its own line numbers
    "#line 1 1\n";

static const char sc_upscaler_shader_suffix[] =
    "\n"
    "#line 1 2\n"
    "layout(local_size_x = NIS_THREAD_GROUP_SIZE) in;\n"
    "void main() {\n"
    "    NVScaler(gl_WorkGroupID.xy, gl_LocalInvocationID.x);\n"
    "}\n";

// NV12 to RGBA at the frame size: the input of VSR
#define SC_CONVERT_GROUP_SIZE 16
static const char sc_upscaler_convert_shader[] =
    "#version 430\n"
    "layout(local_size_x = " SC_STR(SC_CONVERT_GROUP_SIZE) ", "
           "local_size_y = " SC_STR(SC_CONVERT_GROUP_SIZE) ") in;\n"
    "layout(binding = 0) uniform sampler2D in_texture_y;\n"
    "layout(binding = 1) uniform sampler2D in_texture_uv;\n"
    "layout(rgba8, binding = 0) uniform writeonly image2D out_texture;\n"
    "uniform mat3 sc_yuv_matrix;\n"
    "uniform vec3 sc_yuv_offset;\n"
    "void main() {\n"
    "    ivec2 size = imageSize(out_texture);\n"
    "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (pos.x >= size.x || pos.y >= size.y) {\n"
    "        return;\n"
    "    }\n"
    "    float y = texelFetch(in_texture_y, pos, 0).r;\n"
    "    vec2 uv = textureLod(in_texture_uv,\n"
    "                         (vec2(pos) + 0.5) / vec2(size), 0.0).rg;\n"
    "    vec3 rgb = clamp(sc_yuv_matrix * (vec3(y, uv) - sc_yuv_offset),\n"
    "                     0.0, 1.0);\n"
    "    imageStore(out_texture, pos, vec4(rgb, 1.0));\n"
    "}\n";

// std140 layout of the const_buffer uniform block (NISConfig in NIS_Config.h)
struct sc_nis_config {
    float kDetectRatio;
    float kDetectThres;
    float kMinContrastRatio;
    float kRatioNorm;
    float kContrastBoost;
    float kEps;
    float kSharpStartY;
    float kSharpScaleY;
    float kSharpStrengthMin;
    float kSharpStrengthScale;
    float kSharpLimitMin;
    float kSharpLimitScale;
    float kScaleX;
    float kScaleY;
    float kDstNormX;
    float kDstNormY;
    float kSrcNormX;
    float kSrcNormY;
    uint32_t kInputViewportOriginX;
    uint32_t kInputViewportOriginY;
    uint32_t kInputViewportWidth;
    uint32_t kInputViewportHeight;
    uint32_t kOutputViewportOriginX;
    uint32_t kOutputViewportOriginY;
    uint32_t kOutputViewportWidth;
    uint32_t kOutputViewportHeight;
    float reserved0;
    float reserved1;
    float padding[4]; // a whole number of vec4
};

static_assert(sizeof(struct sc_nis_config) == 128, "unexpected NIS config size");

// OpenGL 1.x entry points have no PFN typedefs in glext.h
typedef const GLubyte *(APIENTRYP sc_gl_get_string_fn)(GLenum name);
typedef GLenum (APIENTRYP sc_gl_get_error_fn)(void);
typedef void (APIENTRYP sc_gl_get_integerv_fn)(GLenum pname, GLint *data);
typedef void (APIENTRYP sc_gl_bind_texture_fn)(GLenum target, GLuint texture);
typedef void (APIENTRYP sc_gl_gen_textures_fn)(GLsizei n, GLuint *textures);
typedef void (APIENTRYP sc_gl_delete_textures_fn)(GLsizei n,
                                                  const GLuint *textures);
typedef void (APIENTRYP sc_gl_tex_image_2d_fn)(GLenum target, GLint level,
                                               GLint internalformat,
                                               GLsizei width, GLsizei height,
                                               GLint border, GLenum format,
                                               GLenum type,
                                               const void *pixels);
typedef void (APIENTRYP sc_gl_tex_parameteri_fn)(GLenum target, GLenum pname,
                                                 GLint param);
typedef void (APIENTRYP sc_gl_pixel_storei_fn)(GLenum pname, GLint param);

#define SC_UPSCALER_GL_FUNCTIONS(X) \
    X(sc_gl_get_string_fn, GetString) \
    X(sc_gl_get_error_fn, GetError) \
    X(sc_gl_get_integerv_fn, GetIntegerv) \
    X(sc_gl_bind_texture_fn, BindTexture) \
    X(sc_gl_gen_textures_fn, GenTextures) \
    X(sc_gl_delete_textures_fn, DeleteTextures) \
    X(sc_gl_tex_image_2d_fn, TexImage2D) \
    X(sc_gl_tex_parameteri_fn, TexParameteri) \
    X(sc_gl_pixel_storei_fn, PixelStorei) \
    X(PFNGLACTIVETEXTUREPROC, ActiveTexture) \
    X(PFNGLCREATESHADERPROC, CreateShader) \
    X(PFNGLSHADERSOURCEPROC, ShaderSource) \
    X(PFNGLCOMPILESHADERPROC, CompileShader) \
    X(PFNGLGETSHADERIVPROC, GetShaderiv) \
    X(PFNGLGETSHADERINFOLOGPROC, GetShaderInfoLog) \
    X(PFNGLDELETESHADERPROC, DeleteShader) \
    X(PFNGLCREATEPROGRAMPROC, CreateProgram) \
    X(PFNGLATTACHSHADERPROC, AttachShader) \
    X(PFNGLLINKPROGRAMPROC, LinkProgram) \
    X(PFNGLGETPROGRAMIVPROC, GetProgramiv) \
    X(PFNGLGETPROGRAMINFOLOGPROC, GetProgramInfoLog) \
    X(PFNGLDELETEPROGRAMPROC, DeleteProgram) \
    X(PFNGLUSEPROGRAMPROC, UseProgram) \
    X(PFNGLGETUNIFORMLOCATIONPROC, GetUniformLocation) \
    X(PFNGLPROGRAMUNIFORMMATRIX3FVPROC, ProgramUniformMatrix3fv) \
    X(PFNGLPROGRAMUNIFORM3FVPROC, ProgramUniform3fv) \
    X(PFNGLGENBUFFERSPROC, GenBuffers) \
    X(PFNGLDELETEBUFFERSPROC, DeleteBuffers) \
    X(PFNGLBINDBUFFERPROC, BindBuffer) \
    X(PFNGLBUFFERDATAPROC, BufferData) \
    X(PFNGLBINDBUFFERBASEPROC, BindBufferBase) \
    X(PFNGLGENSAMPLERSPROC, GenSamplers) \
    X(PFNGLDELETESAMPLERSPROC, DeleteSamplers) \
    X(PFNGLSAMPLERPARAMETERIPROC, SamplerParameteri) \
    X(PFNGLBINDSAMPLERPROC, BindSampler) \
    X(PFNGLBINDIMAGETEXTUREPROC, BindImageTexture) \
    X(PFNGLDISPATCHCOMPUTEPROC, DispatchCompute) \
    X(PFNGLMEMORYBARRIERPROC, MemoryBarrier) \
    X(PFNGLGENQUERIESPROC, GenQueries) \
    X(PFNGLDELETEQUERIESPROC, DeleteQueries) \
    X(PFNGLBEGINQUERYPROC, BeginQuery) \
    X(PFNGLENDQUERYPROC, EndQuery) \
    X(PFNGLGETQUERYOBJECTIVPROC, GetQueryObjectiv) \
    X(PFNGLGETQUERYOBJECTUI64VPROC, GetQueryObjectui64v)

struct sc_upscaler_gl {
#define SC_DECLARE(type, name) type name;
    SC_UPSCALER_GL_FUNCTIONS(SC_DECLARE)
#undef SC_DECLARE

    GLuint program;
    GLint yuv_matrix_loc;
    GLint yuv_offset_loc;
    GLuint coef_textures[2]; // scaler, USM
    GLuint sampler;
    GLuint ubo;

    // the sizes and color space the uniforms were last set for
    struct sc_size config_input;
    struct sc_size config_output;
    SDL_Colorspace config_colorspace;
    bool colorspace_set;

    // NV12 to RGBA conversion for VSR (created on first use)
    GLuint convert_program;
    GLint convert_matrix_loc;
    GLint convert_offset_loc;
    SDL_Colorspace convert_colorspace;
    bool convert_colorspace_set;
    GLuint rgba_texture; // the frame in RGBA, input of VSR
    struct sc_size rgba_size;

    GLuint queries[SC_UPSCALER_QUERY_COUNT];
    unsigned query_next; // index of the next query to start
    unsigned query_pending; // started, result not read yet
    uint64_t gpu_time_ns; // sum over gpu_time_samples
    unsigned gpu_time_samples;
};

void
sc_upscaler_init(struct sc_upscaler *up, SDL_Renderer *renderer,
                 uint8_t sharpness, uint8_t vsr_quality) {
    assert(sharpness <= 100);
    up->renderer = renderer;
    up->sharpness = sharpness / 100.f;
    up->vsr_quality = vsr_quality;
    up->gl = NULL;
    up->failed = false;
    up->unsupported_logged = false;
    up->vsr = NULL;
    up->vsr_failed = false;
    up->output = NULL;
    up->output_id = 0;
    up->output_prepared = false;
    up->input_size = (struct sc_size) {0, 0};
    up->output_size = (struct sc_size) {0, 0};
    up->colorspace = SDL_COLORSPACE_UNKNOWN;
}

static void
sc_upscaler_gl_delete(struct sc_upscaler_gl *gl) {
    if (gl->DeleteQueries) {
        gl->DeleteQueries(SC_UPSCALER_QUERY_COUNT, gl->queries);
    }
    if (gl->ubo) {
        gl->DeleteBuffers(1, &gl->ubo);
    }
    if (gl->sampler) {
        gl->DeleteSamplers(1, &gl->sampler);
    }
    if (gl->coef_textures[0]) {
        gl->DeleteTextures(2, gl->coef_textures);
    }
    if (gl->program) {
        gl->DeleteProgram(gl->program);
    }
    if (gl->convert_program) {
        gl->DeleteProgram(gl->convert_program);
    }
    if (gl->rgba_texture) {
        gl->DeleteTextures(1, &gl->rgba_texture);
    }
    free(gl);
}

void
sc_upscaler_destroy(struct sc_upscaler *up) {
#ifdef HAVE_VSR
    // Before the textures it uses are deleted
    if (up->vsr) {
        sc_vsr_destroy(up->vsr);
    }
#endif
    if (up->output) {
        SDL_DestroyTexture(up->output);
    }
    if (up->gl) {
        sc_upscaler_gl_delete(up->gl);
    }
}

static bool
sc_upscaler_gl_load(struct sc_upscaler_gl *gl) {
#define SC_LOAD(type, name) \
    gl->name = (type) SDL_GL_GetProcAddress("gl" #name); \
    if (!gl->name) { \
        LOGW("Upscaling unavailable: gl" #name " not found"); \
        return false; \
    }
    SC_UPSCALER_GL_FUNCTIONS(SC_LOAD)
#undef SC_LOAD
    return true;
}

static bool
sc_upscaler_gl_check_version(struct sc_upscaler_gl *gl) {
    const char *version = (const char *) gl->GetString(GL_VERSION);
    if (!version) {
        LOGW("Upscaling unavailable: unknown OpenGL version");
        return false;
    }

    int major;
    int minor;
    // Compute shaders: OpenGL 4.3 (OpenGL ES is not supported)
    bool ok = strncmp(version, "OpenGL ES", 9)
           && sscanf(version, "%d.%d", &major, &minor) == 2
           && (major > 4 || (major == 4 && minor >= 3));
    if (!ok) {
        LOGW("Upscaling unavailable: OpenGL 4.3+ required (got %s)", version);
        return false;
    }

    return true;
}

static GLuint
sc_upscaler_gl_build_program(struct sc_upscaler_gl *gl,
                             const GLchar *const *sources, GLsizei count,
                             const char *name) {
    GLuint shader = gl->CreateShader(GL_COMPUTE_SHADER);
    if (!shader) {
        LOGE("Upscaling: could not create the compute shader");
        return 0;
    }

    gl->ShaderSource(shader, count, sources, NULL);
    gl->CompileShader(shader);

    char info[4096];
    GLint status;
    gl->GetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        gl->GetShaderInfoLog(shader, sizeof(info), NULL, info);
        LOGE("Upscaling: could not compile the %s shader:\n%s", name, info);
        gl->DeleteShader(shader);
        return 0;
    }

    GLuint program = gl->CreateProgram();
    if (!program) {
        LOGE("Upscaling: could not create the shader program");
        gl->DeleteShader(shader);
        return 0;
    }

    gl->AttachShader(program, shader);
    gl->LinkProgram(program);
    gl->DeleteShader(shader); // flagged, deleted with the program

    gl->GetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        gl->GetProgramInfoLog(program, sizeof(info), NULL, info);
        LOGE("Upscaling: could not link the %s shader:\n%s", name, info);
        gl->DeleteProgram(program);
        return 0;
    }

    return program;
}

static void
sc_upscaler_gl_create_coef_textures(struct sc_upscaler_gl *gl) {
    // 8 taps per phase = 2 RGBA texels per row, one row per phase
    static_assert(SC_NIS_FILTER_SIZE == 8, "unexpected NIS filter size");
    const float *data[2] = {
        &sc_nis_coef_scale[0][0],
        &sc_nis_coef_usm[0][0],
    };

    // SDL leaves its own unpack row length set after texture uploads
    GLint row_length;
    GLint alignment;
    gl->GetIntegerv(GL_UNPACK_ROW_LENGTH, &row_length);
    gl->GetIntegerv(GL_UNPACK_ALIGNMENT, &alignment);
    gl->PixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 4);

    gl->GenTextures(2, gl->coef_textures);
    for (unsigned i = 0; i < 2; ++i) {
        gl->BindTexture(GL_TEXTURE_2D, gl->coef_textures[i]);
        // Read with texelFetch() only, but a texture with the default
        // (mipmapped) minification filter and one level is incomplete
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SC_NIS_FILTER_SIZE / 4,
                       SC_NIS_PHASE_COUNT, 0, GL_RGBA, GL_FLOAT, data[i]);
    }
    gl->BindTexture(GL_TEXTURE_2D, 0);

    gl->PixelStorei(GL_UNPACK_ROW_LENGTH, row_length);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, alignment);
}

static bool
sc_upscaler_init_gl(struct sc_upscaler *up) {
    struct sc_upscaler_gl *gl = calloc(1, sizeof(*gl));
    if (!gl) {
        LOG_OOM();
        return false;
    }

    if (!sc_upscaler_gl_load(gl)) {
        free(gl);
        return false;
    }

    if (!sc_upscaler_gl_check_version(gl)) {
        free(gl);
        return false;
    }

    while (gl->GetError()); // clear errors left by others

    const GLchar *sources[] = {
        sc_upscaler_shader_prefix,
        sc_nis_scaler_glsl,
        sc_upscaler_shader_suffix,
    };
    gl->program = sc_upscaler_gl_build_program(gl, sources, 3, "NIS");
    if (!gl->program) {
        sc_upscaler_gl_delete(gl);
        return false;
    }

    gl->yuv_matrix_loc = gl->GetUniformLocation(gl->program, "sc_yuv_matrix");
    gl->yuv_offset_loc = gl->GetUniformLocation(gl->program, "sc_yuv_offset");
    if (gl->yuv_matrix_loc < 0 || gl->yuv_offset_loc < 0) {
        LOGE("Upscaling: color conversion uniforms not found");
        sc_upscaler_gl_delete(gl);
        return false;
    }

    sc_upscaler_gl_create_coef_textures(gl);

    // The frame planes are sampled bilinearly at arbitrary positions,
    // clamped to the edges, whatever the scale mode of the SDL texture
    gl->GenSamplers(1, &gl->sampler);
    gl->SamplerParameteri(gl->sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->SamplerParameteri(gl->sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl->SamplerParameteri(gl->sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl->SamplerParameteri(gl->sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    gl->GenBuffers(1, &gl->ubo);
    gl->BindBuffer(GL_UNIFORM_BUFFER, gl->ubo);
    gl->BufferData(GL_UNIFORM_BUFFER, sizeof(struct sc_nis_config), NULL,
                   GL_DYNAMIC_DRAW);
    gl->BindBuffer(GL_UNIFORM_BUFFER, 0);

    gl->GenQueries(SC_UPSCALER_QUERY_COUNT, gl->queries);

    GLenum error = gl->GetError();
    if (error) {
        LOGE("Upscaling: could not create the OpenGL objects (GL error %#x)",
             error);
        sc_upscaler_gl_delete(gl);
        return false;
    }

    LOGI("Upscaling: NVIDIA Image Scaling ready (sharpness %d%%)",
         (int) (up->sharpness * 100 + 0.5f));
    up->gl = gl;
    return true;
}

static inline float
sc_max_f(float a, float b) {
    return a > b ? a : b;
}

// Port of NVScalerUpdateConfig() from NIS_Config.h (SDR, whole textures)
static void
sc_nis_config_init(struct sc_nis_config *c, float sharpness,
                   struct sc_size in, struct sc_size out) {
    float sharpen_slider = sharpness - 0.5f; // -0.5 to 0.5

    // 0% sharpness means no sharpening, 100% must not oversharpen
    float max_scale = sharpen_slider >= 0.f ? 1.25f : 1.75f;
    float min_scale = sharpen_slider >= 0.f ? 1.25f : 1.0f;
    float limit_scale = sharpen_slider >= 0.f ? 1.25f : 1.0f;

    float min_contrast_ratio = 2.0f;
    float max_contrast_ratio = 10.0f;
    float sharp_start_y = 0.45f;
    float sharp_end_y = 0.9f;
    float sharp_strength_min =
        sc_max_f(0.0f, 0.4f + sharpen_slider * min_scale * 1.2f);
    float sharp_strength_max = 1.6f + sharpen_slider * max_scale * 1.8f;
    float sharp_limit_min =
        sc_max_f(0.1f, 0.14f + sharpen_slider * limit_scale * 0.32f);
    float sharp_limit_max = 0.5f + sharpen_slider * limit_scale * 0.6f;

    memset(c, 0, sizeof(*c));
    c->kDetectRatio = 2 * 1127.f / 1024.f;
    c->kDetectThres = 64.0f / 1024.0f;
    c->kMinContrastRatio = min_contrast_ratio;
    c->kRatioNorm = 1.0f / (max_contrast_ratio - min_contrast_ratio);
    c->kContrastBoost = 1.0f;
    c->kEps = 1.0f / 255.0f;
    c->kSharpStartY = sharp_start_y;
    c->kSharpScaleY = 1.0f / (sharp_end_y - sharp_start_y);
    c->kSharpStrengthMin = sharp_strength_min;
    c->kSharpStrengthScale = sharp_strength_max - sharp_strength_min;
    c->kSharpLimitMin = sharp_limit_min;
    c->kSharpLimitScale = sharp_limit_max - sharp_limit_min;
    c->kScaleX = (float) in.width / out.width;
    c->kScaleY = (float) in.height / out.height;
    c->kDstNormX = 1.0f / out.width;
    c->kDstNormY = 1.0f / out.height;
    c->kSrcNormX = 1.0f / in.width;
    c->kSrcNormY = 1.0f / in.height;
    c->kInputViewportWidth = in.width;
    c->kInputViewportHeight = in.height;
    c->kOutputViewportWidth = out.width;
    c->kOutputViewportHeight = out.height;
}

// Set the YUV to RGB conversion of a program, the same as SDL's for this color
// space
static void
sc_upscaler_gl_set_colorspace(struct sc_upscaler_gl *gl, GLuint program,
                              GLint matrix_loc, GLint offset_loc,
                              SDL_Colorspace colorspace) {
    float kr;
    float kb;
    if (SDL_ISCOLORSPACE_MATRIX_BT709(colorspace)) {
        kr = 0.2126f;
        kb = 0.0722f;
    } else if (SDL_ISCOLORSPACE_MATRIX_BT2020_NCL(colorspace)) {
        kr = 0.2627f;
        kb = 0.0593f;
    } else {
        // BT.601, also by default
        kr = 0.299f;
        kb = 0.114f;
    }
    float kg = 1.f - kr - kb;

    bool full_range = SDL_ISCOLORSPACE_FULL_RANGE(colorspace);
    float ys = full_range ? 1.f : 255.f / 219.f;
    float cs = full_range ? 1.f : 255.f / 224.f;

    // rgb = matrix * (yuv - offset), column-major
    const float matrix[9] = {
        ys, ys, ys,
        0.f, -2.f * kb * (1.f - kb) / kg * cs, 2.f * (1.f - kb) * cs,
        2.f * (1.f - kr) * cs, -2.f * kr * (1.f - kr) / kg * cs, 0.f,
    };
    const float offset[3] = {
        full_range ? 0.f : 16.f / 255.f,
        128.f / 255.f,
        128.f / 255.f,
    };

    gl->ProgramUniformMatrix3fv(program, matrix_loc, 1, GL_FALSE, matrix);
    gl->ProgramUniform3fv(program, offset_loc, 1, offset);
}

static bool
sc_upscaler_create_output(struct sc_upscaler *up, struct sc_size size) {
    if (up->output) {
#ifdef HAVE_VSR
        if (up->vsr) {
            // CUDA must forget the texture before it is deleted
            sc_vsr_release_textures(up->vsr);
        }
#endif
        SDL_DestroyTexture(up->output);
        up->output = NULL;
    }

    SDL_Texture *texture =
        SDL_CreateTexture(up->renderer, SDL_PIXELFORMAT_ABGR8888,
                          SDL_TEXTUREACCESS_STATIC, size.width, size.height);
    if (!texture) {
        LOGE("Upscaling: could not create the output texture: %s",
             SDL_GetError());
        return false;
    }

    SDL_PropertiesID props = SDL_GetTextureProperties(texture);
    int64_t id =
        SDL_GetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, 0);
    int64_t target =
        SDL_GetNumberProperty(props,
                              SDL_PROP_TEXTURE_OPENGL_TEXTURE_TARGET_NUMBER, 0);
    if (!id || target != GL_TEXTURE_2D) {
        LOGE("Upscaling: unexpected OpenGL output texture (target %#" PRIx64
             ")", target);
        SDL_DestroyTexture(texture);
        return false;
    }

    // Drawn 1:1 and opaque
    SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_NONE);

    assert(!(id & ~0xFFFFFFFF));
    up->output = texture;
    up->output_id = id;
    up->output_prepared = false;
    up->output_size = size;
    return true;
}

// Read the GPU time of finished passes (never waits)
static void
sc_upscaler_collect_timings(struct sc_upscaler_gl *gl) {
    while (gl->query_pending) {
        unsigned idx = (gl->query_next + SC_UPSCALER_QUERY_COUNT
                                       - gl->query_pending)
                     % SC_UPSCALER_QUERY_COUNT;
        GLint available = 0;
        gl->GetQueryObjectiv(gl->queries[idx], GL_QUERY_RESULT_AVAILABLE,
                             &available);
        if (!available) {
            return;
        }

        GLuint64 ns = 0;
        gl->GetQueryObjectui64v(gl->queries[idx], GL_QUERY_RESULT, &ns);
        --gl->query_pending;

        gl->gpu_time_ns += ns;
        if (++gl->gpu_time_samples == SC_UPSCALER_TIMING_SAMPLES) {
            LOGD("Upscaling: %.3f ms GPU time per frame (average of %u)",
                 gl->gpu_time_ns / 1e6 / gl->gpu_time_samples,
                 gl->gpu_time_samples);
            gl->gpu_time_ns = 0;
            gl->gpu_time_samples = 0;
        }
    }
}

static inline bool
sc_size_equals(struct sc_size a, struct sc_size b) {
    return a.width == b.width && a.height == b.height;
}

static void
sc_upscaler_run_nis(struct sc_upscaler *up, GLuint y_id, GLuint uv_id,
                    SDL_Colorspace colorspace, struct sc_size input_size,
                    struct sc_size output_size) {
    struct sc_upscaler_gl *gl = up->gl;

    if (!gl->colorspace_set || gl->config_colorspace != colorspace) {
        sc_upscaler_gl_set_colorspace(gl, gl->program, gl->yuv_matrix_loc,
                                      gl->yuv_offset_loc, colorspace);
        gl->config_colorspace = colorspace;
        gl->colorspace_set = true;
    }

    if (!sc_size_equals(gl->config_input, input_size)
            || !sc_size_equals(gl->config_output, output_size)) {
        struct sc_nis_config config;
        sc_nis_config_init(&config, up->sharpness, input_size, output_size);
        gl->BindBuffer(GL_UNIFORM_BUFFER, gl->ubo);
        gl->BufferData(GL_UNIFORM_BUFFER, sizeof(config), &config,
                       GL_DYNAMIC_DRAW);
        gl->BindBuffer(GL_UNIFORM_BUFFER, 0);
        gl->config_input = input_size;
        gl->config_output = output_size;
        LOGD("Upscaling: %" PRIu16 "x%" PRIu16 " -> %" PRIu16 "x%" PRIu16,
             input_size.width, input_size.height,
             output_size.width, output_size.height);
    }

    const GLuint textures[4] = {
        y_id,
        uv_id,
        gl->coef_textures[0],
        gl->coef_textures[1],
    };
    for (unsigned i = 0; i < 4; ++i) {
        gl->ActiveTexture(GL_TEXTURE0 + i);
        gl->BindTexture(GL_TEXTURE_2D, textures[i]);
        gl->BindSampler(i, i < 2 ? gl->sampler : 0);
    }
    gl->BindImageTexture(0, up->output_id, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                         GL_RGBA8);
    gl->BindBufferBase(GL_UNIFORM_BUFFER, 0, gl->ubo);
    gl->UseProgram(gl->program);

    sc_upscaler_collect_timings(gl);
    bool timed = gl->query_pending < SC_UPSCALER_QUERY_COUNT;
    if (timed) {
        gl->BeginQuery(GL_TIME_ELAPSED, gl->queries[gl->query_next]);
    }

    GLuint groups_x =
        (output_size.width + SC_NIS_BLOCK_WIDTH - 1) / SC_NIS_BLOCK_WIDTH;
    GLuint groups_y =
        (output_size.height + SC_NIS_BLOCK_HEIGHT - 1) / SC_NIS_BLOCK_HEIGHT;
    gl->DispatchCompute(groups_x, groups_y, 1);

    if (timed) {
        gl->EndQuery(GL_TIME_ELAPSED);
        gl->query_next = (gl->query_next + 1) % SC_UPSCALER_QUERY_COUNT;
        ++gl->query_pending;
    }

    // The output is sampled by SDL right after
    gl->MemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

// Convert the frame to RGBA, then upscale it with VSR into the output texture.
// On failure, set vsr_failed (NIS is used instead from now on).
static bool
sc_upscaler_run_vsr(struct sc_upscaler *up, GLuint y_id, GLuint uv_id,
                    SDL_Colorspace colorspace, struct sc_size input_size,
                    struct sc_size output_size) {
#ifdef HAVE_VSR
    struct sc_upscaler_gl *gl = up->gl;

    if (!up->vsr) {
        up->vsr = sc_vsr_new(up->vsr_quality);
        if (!up->vsr) {
            up->vsr_failed = true;
            return false;
        }
    }

    if (!gl->convert_program) {
        const GLchar *sources[] = {sc_upscaler_convert_shader};
        gl->convert_program =
            sc_upscaler_gl_build_program(gl, sources, 1, "NV12 to RGBA");
        if (!gl->convert_program) {
            up->vsr_failed = true;
            return false;
        }
        gl->convert_matrix_loc =
            gl->GetUniformLocation(gl->convert_program, "sc_yuv_matrix");
        gl->convert_offset_loc =
            gl->GetUniformLocation(gl->convert_program, "sc_yuv_offset");
        if (gl->convert_matrix_loc < 0 || gl->convert_offset_loc < 0) {
            LOGE("Upscaling: color conversion uniforms not found");
            up->vsr_failed = true;
            return false;
        }
    }

    if (!gl->rgba_texture || !sc_size_equals(gl->rgba_size, input_size)) {
        // CUDA must forget the texture before it is deleted
        sc_vsr_release_textures(up->vsr);
        if (gl->rgba_texture) {
            gl->DeleteTextures(1, &gl->rgba_texture);
        }
        gl->GenTextures(1, &gl->rgba_texture);
        gl->BindTexture(GL_TEXTURE_2D, gl->rgba_texture);
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, input_size.width,
                       input_size.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        gl->BindTexture(GL_TEXTURE_2D, 0);
        gl->rgba_size = input_size;
    }

    if (!gl->convert_colorspace_set || gl->convert_colorspace != colorspace) {
        sc_upscaler_gl_set_colorspace(gl, gl->convert_program,
                                      gl->convert_matrix_loc,
                                      gl->convert_offset_loc, colorspace);
        gl->convert_colorspace = colorspace;
        gl->convert_colorspace_set = true;
    }

    const GLuint planes[2] = {y_id, uv_id};
    for (unsigned i = 0; i < 2; ++i) {
        gl->ActiveTexture(GL_TEXTURE0 + i);
        gl->BindTexture(GL_TEXTURE_2D, planes[i]);
        gl->BindSampler(i, gl->sampler);
    }
    gl->BindImageTexture(0, gl->rgba_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                         GL_RGBA8);
    gl->UseProgram(gl->convert_program);
    gl->DispatchCompute(
        (input_size.width + SC_CONVERT_GROUP_SIZE - 1) / SC_CONVERT_GROUP_SIZE,
        (input_size.height + SC_CONVERT_GROUP_SIZE - 1) / SC_CONVERT_GROUP_SIZE,
        1);
    gl->MemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT
                    | GL_TEXTURE_FETCH_BARRIER_BIT);
    // CUDA maps the texture: it must not be bound to an image unit
    gl->UseProgram(0);
    gl->BindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    GLenum error = gl->GetError();
    if (error) {
        LOGE("Upscaling: NV12 to RGBA conversion failed (GL error %#x)",
             error);
        up->vsr_failed = true;
        return false;
    }

    if (!sc_vsr_process(up->vsr, gl->rgba_texture, input_size, up->output_id,
                        output_size)) {
        up->vsr_failed = true;
        return false;
    }
    return true;
#else
    (void) up;
    (void) y_id;
    (void) uv_id;
    (void) colorspace;
    (void) input_size;
    (void) output_size;
    LOGW("VSR support is not compiled in (ffnvcodec headers not found)");
    up->vsr_failed = true;
    return false;
#endif
}

SDL_Texture *
sc_upscaler_process(struct sc_upscaler *up, enum sc_upscale mode,
                    SDL_Texture *input, struct sc_size input_size,
                    struct sc_size output_size) {
    assert(mode == SC_UPSCALE_NIS || mode == SC_UPSCALE_VSR);
    if (up->failed) {
        return NULL;
    }

    if (output_size.width <= input_size.width
            || output_size.height <= input_size.height) {
        // NIS and VSR only upscale; the usual rendering (with mipmaps if
        // enabled) is better at downscaling
        return NULL;
    }

    // The chroma plane of an NV12 texture is a two-channel (.rg) texture only
    // when imported from VA-API (see interop_vaapi.c)
    SDL_PropertiesID props = SDL_GetTextureProperties(input);
    int64_t y_id =
        SDL_GetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, 0);
    int64_t uv_id =
        SDL_GetNumberProperty(props, SDL_PROP_TEXTURE_OPENGL_TEXTURE_UV_NUMBER,
                              0);
    bool supported = input->format == SDL_PIXELFORMAT_NV12
                  && SDL_GetHintBoolean("SDL_RENDER_OPENGL_NV12_RG_SHADER",
                                        false)
                  && y_id && uv_id;
    if (!supported) {
        if (!up->unsupported_logged) {
            LOGW("Upscaling needs VA-API decoding with the OpenGL renderer, "
                 "drawing frames without it");
            up->unsupported_logged = true;
        }
        return NULL;
    }

    SDL_Colorspace colorspace =
        SDL_GetNumberProperty(props, SDL_PROP_TEXTURE_COLORSPACE_NUMBER,
                              SDL_COLORSPACE_BT601_LIMITED);

    if (!up->output || !sc_size_equals(up->output_size, output_size)) {
        if (!sc_upscaler_create_output(up, output_size)) {
            up->failed = true;
            return NULL;
        }
    }

    // Run SDL's pending commands, and make SDL forget its cached OpenGL state:
    // it restores its own bindings on the next draw
    if (!SDL_FlushRenderer(up->renderer)) {
        LOGW("Upscaling: could not flush the renderer: %s", SDL_GetError());
        return NULL;
    }

    if (!up->gl && !sc_upscaler_init_gl(up)) {
        up->failed = true;
        return NULL;
    }

    struct sc_upscaler_gl *gl = up->gl;
    while (gl->GetError()); // clear errors left by others

    if (!up->output_prepared) {
        // SDL sets the filters of a texture only when it draws it. Until
        // then, the default minification filter uses mipmaps, so the texture
        // (a single level) is incomplete and image stores are ignored.
        gl->BindTexture(GL_TEXTURE_2D, up->output_id);
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl->BindTexture(GL_TEXTURE_2D, 0);
        up->output_prepared = true;
    }

    bool done = false;
    if (mode == SC_UPSCALE_VSR && !up->vsr_failed) {
        done = sc_upscaler_run_vsr(up, (GLuint) y_id, (GLuint) uv_id,
                                   colorspace, input_size, output_size);
        if (up->vsr_failed) {
            LOGW("Upscaling: VSR unavailable, using NIS instead");
        }
    }
    if (!done) {
        sc_upscaler_run_nis(up, (GLuint) y_id, (GLuint) uv_id, colorspace,
                            input_size, output_size);
    }

    // Unbind everything (SDL expects texture unit 0 to be active)
    gl->UseProgram(0);
    gl->BindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
    gl->BindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    for (int i = 3; i >= 0; --i) {
        gl->ActiveTexture(GL_TEXTURE0 + i);
        gl->BindTexture(GL_TEXTURE_2D, 0);
        gl->BindSampler(i, 0);
    }

    GLenum error = gl->GetError();
    if (error) {
        LOGE("Upscaling failed (GL error %#x), disabled", error);
        up->failed = true;
        return NULL;
    }

    up->input_size = input_size;
    up->colorspace = colorspace;
    return up->output;
}
