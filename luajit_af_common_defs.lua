--------------------------------------------------------------------------------
-- luajit_af_common_defs.lua (Refactored cdefs for ArrayFire 3.9.0)
--------------------------------------------------------------------------------
local ffi = require("ffi")

ffi.cdef[[

    // --------------------------------------------------------------------------
    //                           ArrayFire Enums
    // --------------------------------------------------------------------------
    typedef enum {
        AF_COLORMAP_DEFAULT = 0,
        AF_COLORMAP_SPECTRUM,
        AF_COLORMAP_COLORS,
        AF_COLORMAP_RED,
        AF_COLORMAP_MOOD,
        AF_COLORMAP_HEAT,
        AF_COLORMAP_BLUE,
        AF_COLORMAP_INFERNO,
        AF_COLORMAP_MAGMA,
        AF_COLORMAP_PLASMA,
        AF_COLORMAP_VIRIDIS,
        AF_COLORMAP_CIVIDIS,
        AF_COLORMAP_TWILIGHT,
        AF_COLORMAP_TWILIGHT_SHIFTED,
        AF_COLORMAP_TURBO
    } af_colormap;
    
    typedef enum {
        AF_MARKER_NONE = 0,
        AF_MARKER_POINT,
        AF_MARKER_CIRCLE,
        AF_MARKER_SQUARE,
        AF_MARKER_TRIANGLE,
        AF_MARKER_CROSS,
        AF_MARKER_PLUS,
        AF_MARKER_STAR
    } af_marker_type;

    typedef enum {
        AF_SUCCESS            =   0,
        AF_ERR_NO_MEM         = 101,
        AF_ERR_DRIVER         = 102,
        AF_ERR_RUNTIME        = 103,
        AF_ERR_INVALID_ARRAY  = 201,
        AF_ERR_ARG            = 202,
        AF_ERR_SIZE           = 203,
        AF_ERR_TYPE           = 204,
        AF_ERR_DIFF_TYPE      = 205,
        AF_ERR_BATCH          = 207,
        AF_ERR_DEVICE         = 208,
        AF_ERR_NOT_SUPPORTED  = 301,
        AF_ERR_NOT_CONFIGURED = 302,
        AF_ERR_NONFREE        = 303,
        AF_ERR_NO_DBL         = 401,
        AF_ERR_NO_GFX         = 402,
        AF_ERR_NO_HALF        = 403,
        AF_ERR_LOAD_LIB       = 501,
        AF_ERR_LOAD_SYM       = 502,
        AF_ERR_ARR_BKND_MISMATCH = 503,
        AF_ERR_INTERNAL       = 998,
        AF_ERR_UNKNOWN        = 999
    } af_err;

    typedef enum {
        AF_BINARY_ADD,
        AF_BINARY_MUL,
        AF_BINARY_MIN,
        AF_BINARY_MAX
    } af_binary_op;
    
    typedef enum {
        AF_CONV_DEFAULT,
        AF_CONV_EXPAND
    } af_conv_mode;

    typedef enum {afDevice, afHost} af_source;

    typedef enum {
        AF_CONV_AUTO,
        AF_CONV_SPATIAL,
        AF_CONV_FREQ
    } af_conv_domain;
    
    typedef enum {
        AF_RANDOM_ENGINE_PHILOX_4X32_10,
        AF_RANDOM_ENGINE_THREEFRY_2X32_16,
        AF_RANDOM_ENGINE_MERSENNE_GP11213,
        AF_RANDOM_ENGINE_PHILOX,
        AF_RANDOM_ENGINE_THREEFRY,
        AF_RANDOM_ENGINE_DEFAULT
    } af_random_engine_type;

    typedef enum {
        AF_CONNECTIVITY_FOUR = 4,
        AF_CONNECTIVITY_EIGHT = 8
    } af_connectivity;
    
    typedef enum {
        AF_CANNY_THRESHOLD_AUTO_OTSU,
        AF_CANNY_THRESHOLD_MANUAL
    } af_canny_threshold;
    
    typedef enum {
        AF_CSPACE_RGB,
        AF_CSPACE_GRAY,
        AF_CSPACE_HSV,
        AF_CSPACE_YCbCr
    } af_cspace_t;

    typedef enum {
        AF_HOMOGRAPHY_RANSAC,
        AF_HOMOGRAPHY_LMEDS
    } af_homography_type;

    typedef enum {
        AF_MAT_NONE,
        AF_MAT_TRANS,
        AF_MAT_CTRANS,
        AF_MAT_UPPER,
        AF_MAT_LOWER
    } af_mat_prop;
    
    typedef enum {
        AF_INVERSE_DECONV_LUCY,
        AF_INVERSE_DECONV_TIKHONOV
    } af_inverse_deconv_algo;
    
    typedef enum {
        AF_MATCH_SAD,
        AF_MATCH_ZSAD,
        AF_MATCH_LSAD,
        AF_MATCH_SSD,
        AF_MATCH_ZSSD,
        AF_MATCH_LSSD,
        AF_MATCH_NCC,
        AF_MATCH_ZNCC,
        AF_MATCH_SHD
    } af_match_type;

    typedef enum {
        AF_PAD_ZERO,
        AF_PAD_SYM
    } af_border_type;
    
    typedef enum {
        AF_MOMENT_M00,
        AF_MOMENT_M01,
        AF_MOMENT_M10,
        AF_MOMENT_M11
    } af_moment_type;

    typedef enum {
        AF_NORM_VECTOR_1,
        AF_NORM_VECTOR_INF,
        AF_NORM_VECTOR_2,
        AF_NORM_VECTOR_P,
        AF_NORM_MATRIX_1,
        AF_NORM_MATRIX_INF,
        AF_NORM_MATRIX_2,
        AF_NORM_MATRIX_L_PQ,
        AF_NORM_EUCLID
    } af_norm_type;

    typedef enum {
        AF_INTERP_NEAREST,
        AF_INTERP_LINEAR,
        AF_INTERP_BILINEAR,
        AF_INTERP_CUBIC,
        AF_INTERP_LOWER
    } af_interp_type;

    typedef enum {
        AF_YCC_601,
        AF_YCC_709,
        AF_YCC_2020
    } af_ycc_std;

    typedef enum {
        AF_IMAGE_PNG,
        AF_IMAGE_JPG,
        AF_IMAGE_BMP,
        AF_IMAGE_TIFF
    } af_image_format;

    typedef enum {
        AF_STORAGE_DENSE,
        AF_STORAGE_CSR,
        AF_STORAGE_CSC,
        AF_STORAGE_COO
    } af_storage;

    typedef enum {
        AF_TOPK_MIN,
        AF_TOPK_MAX,
        AF_TOPK_DEFAULT
    } af_topk_function;

    // --------------------------------------------------------------------------
    //                         ArrayFire Basic Types
    // --------------------------------------------------------------------------
    typedef long long dim_t;
    typedef struct af_array_t* af_array;
    typedef struct af_seq {double begin, end, step; unsigned is_gfor;} af_seq;
    typedef void* af_event;
    typedef void* af_random_engine;
    typedef void* af_features;
    typedef unsigned int af_backend;
    typedef void* af_window;
    typedef enum {f32, f64, c32, c64, b8, s32, u32, u8, s64, u64, s16, u16, f16} af_dtype;

    // --------------------------------------------------------------------------
    //                          Indexing Structure
    // --------------------------------------------------------------------------
    typedef struct af_index_t {
        union {
            af_array arr;
            af_seq   seq;
        } idx;
        bool isSeq;
        bool isBatch;
    } af_index_t;
    
    typedef struct af_cell {
        int row;
        int col;
        const char* title;
        af_colormap cmap;
    } af_cell;

    typedef struct {af_array arr;} array;
    typedef struct {dim_t dims[4];} dim4;

    // --------------------------------------------------------------------------
    //                    dim4 Helpers
    // --------------------------------------------------------------------------
    dim_t dim4_elements(dim4* self);
    dim_t dim4_ndims(dim4* self);
    dim4 dim4_operator_add(const dim4* first, const dim4* second);
    dim4 dim4_operator_multiply(const dim4* first, const dim4* second);
    dim4 dim4_operator_subtract(const dim4* first, const dim4* second);
    bool dim4_operator_equals(const dim4* self, const dim4* other);
    bool dim4_operator_not_equals(const dim4* self, const dim4* other);
    void dim4_operator_multiply_assign(dim4* self, const dim4* other);
    void dim4_operator_add_assign(dim4* self, const dim4* other);
    void dim4_operator_subtract_assign(dim4* self, const dim4* other);
    dim_t* dim4_operator_index(dim4* self, unsigned int dim);
    const dim_t* dim4_operator_index_const(const dim4* self, unsigned int dim);

    // --------------------------------------------------------------------------
    //                    Event Helpers
    // --------------------------------------------------------------------------
    void af_event_free(void* self);
    void af_event_mark(void* self);
    void af_event_enqueue(void* self);
    void af_event_block(const void* self);

    // --------------------------------------------------------------------------
    //                    Additional Feature Helpers
    // --------------------------------------------------------------------------
    size_t af_features_getNumFeatures(const void* self);

    // --------------------------------------------------------------------------
    //                    Random Engine Helpers
    // --------------------------------------------------------------------------
    void af_random_engine_free(void* self);
    void af_random_engine_set_type(void* self, const af_random_engine_type type);
    unsigned long long af_random_engine_get_seed(const void* self);
    void af_random_engine_set_seed(void* self, const unsigned long long seed);

    // --------------------------------------------------------------------------
    //                    API Declarations
    // --------------------------------------------------------------------------
    // Error Handling
    void af_get_last_error(char **msg, dim_t *len);
    const char *af_err_to_string(const af_err err);

    // Create and release arrays
    af_err af_create_array(
        af_array *out,
        const void *data,
        const unsigned ndims,
        const dim_t * const dims,
        const af_dtype types
    );
    af_err af_release_array(af_array arr);
    af_err af_retain_array(af_array *out, const af_array in);

    // Evaluate
    af_err af_eval(const af_array arr);
    af_err af_eval_multiple(const int num, af_array *arrs);

    // Write
    af_err af_write_array(
        af_array arr,
        const void *data,
        const size_t bytes,
        af_source src
    );

    // Copy
    af_err af_copy_array(af_array *out, const af_array in);

    // Array info
    af_err af_get_elements(dim_t *elems, const af_array arr);
    af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3, const af_array arr);
    af_err af_get_numdims(unsigned *result, const af_array arr);
    af_err af_get_data_ptr(void *data, const af_array arr);

    // Type
    af_err af_get_type(af_dtype *out, const af_array arr);
    af_err af_get_scalar(double *val, const af_array arr);

    // Query
    af_err af_is_empty(bool *result, const af_array arr);
    af_err af_is_scalar(bool *result, const af_array arr);
    af_err af_is_row(bool *result, const af_array arr);
    af_err af_is_column(bool *result, const af_array arr);
    af_err af_is_vector(bool *result, const af_array arr);
    af_err af_is_complex(bool *result, const af_array arr);
    af_err af_is_real(bool *result, const af_array arr);
    af_err af_is_double(bool *result, const af_array arr);
    af_err af_is_single(bool *result, const af_array arr);
    af_err af_is_half(bool *result, const af_array arr);
    af_err af_is_realfloating(bool *result, const af_array arr);
    af_err af_is_floating(bool *result, const af_array arr);
    af_err af_is_integer(bool *result, const af_array arr);
    af_err af_is_bool(bool *result, const af_array arr);
    af_err af_is_sparse(bool *result, const af_array arr);

    // Features
    af_err af_create_features(af_features *feat, dim_t num);
    af_err af_release_features(af_features feat);
    af_err af_retain_features(af_features *out, af_features in);
    af_err af_get_features_num(dim_t *out, const af_features in);
    af_err af_get_features_xpos(af_array *out, const af_features in);
    af_err af_get_features_ypos(af_array *out, const af_features in);
    af_err af_get_features_score(af_array *out, const af_features in);
    af_err af_get_features_orientation(af_array *out, const af_features in);
    af_err af_get_features_size(af_array *out, const af_features in);

    // Events
    af_err af_create_event(af_event *out);
    af_err af_delete_event(af_event eventHandle);
    af_err af_mark_event(const af_event eventHandle);
    af_err af_enqueue_wait_event(const af_event eventHandle);
    af_err af_block_event(const af_event eventHandle);

    // Indexing
    af_err af_lookup(
        af_array *out,
        const af_array in,
        const af_array indices,
        const unsigned dim
    );
    // Some older versions had af_index, plus official af_index_gen/af_assign_gen.
    af_err af_index(af_array *out, const af_array in, const af_index_t *index, const unsigned ndims);
    af_err af_assign_seq(af_array *out, const af_array lhs, const unsigned ndims,
                         const af_seq *seqs, const af_array rhs);

    af_err af_index_gen(
        af_array *out,
        const af_array in,
        const unsigned ndims,
        const af_index_t *indices
    );
    af_err af_assign_gen(
        af_array *out,
        const af_array lhs,
        const unsigned ndims,
        const af_index_t *indices,
        const af_array rhs
    );

    af_err af_create_indexers(af_index_t** indexers);
    af_err af_release_indexers(af_index_t* indexers);
    af_err af_set_array_indexer(af_index_t* indexers, const af_array idx, dim_t dim);
    af_err af_set_seq_indexer(af_index_t* indexers, const af_seq *seq, dim_t dim, bool is_batch);
    af_err af_set_seq_param_indexer(af_index_t* indexers, double begin, double end,
                                    double step, dim_t dim, bool is_batch);

    // Setting seed
    af_err af_set_seed(const unsigned long long seed);
    af_err af_get_seed(unsigned long long *seed);
    af_err af_randu(af_array *out, const unsigned ndims,
                    const dim_t * const dims, const af_dtype type);
    af_err af_randn(af_array *out, const unsigned ndims,
                    const dim_t * const dims, const af_dtype type);

    // set/get manual eval
    af_err af_set_manual_eval_flag(const bool flag);
    af_err af_get_manual_eval_flag(bool *out);

    // Example Ops
    af_err af_abs(af_array *out, const af_array in);
    af_err af_accum(af_array *out, const af_array in, const int dim);

    // Matrix multiply
    af_err af_matmul(
        af_array *out,
        const af_array lhs,
        const af_array rhs,
        const int optLhs,
        const int optRhs
    );

    // Transpose
    af_err af_transpose(
        af_array *out,
        const af_array in,
        const bool conjugate
    );

    // Basic elementwise ops
    af_err af_add(af_array *out, const af_array lhs, const af_array rhs, bool batch);
    af_err af_sub(af_array *out, const af_array lhs, const af_array rhs, bool batch);
    af_err af_mul(af_array *out, const af_array lhs, const af_array rhs, bool batch);
    af_err af_div(af_array *out, const af_array lhs, const af_array rhs, bool batch);

    // Scalar compare
    af_err af_gt_scalar(af_array *out, const af_array lhs, const double rhs, bool batch);
    af_err af_lt_scalar(af_array *out, const af_array lhs, const double rhs, bool batch);

    // Casting
    af_err af_cast(af_array *out, const af_array in, const af_dtype type);

    // Summation over all elements
    af_err af_sum_all(double *real, double *imag, const af_array in);

    // Max over all elements
    af_err af_max_all(double *real, double *imag, const af_array in);

    // Exponential
    af_err af_exp(af_array *out, const af_array in);

    // Create constant array
    af_err af_constant(
        af_array *out,
        const double val,
        const unsigned ndims,
        const dim_t *dims,
        const af_dtype type
    );

    // Sqrt
    af_err af_sqrt(af_array *out, const af_array in);

    // Elementwise min/max
    af_err af_minof(af_array *out, const af_array lhs, const af_array rhs, bool batch);
    af_err af_maxof(af_array *out, const af_array lhs, const af_array rhs, bool batch);
]]

--------------------------------------------------------------------------------
-- Example: you can store version info or other constants here if desired.
--------------------------------------------------------------------------------
local M = {}
M.AF_VERSION_STRING = "3.9.0 (Refactored)"

return M
