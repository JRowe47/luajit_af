local ffi = require 'ffi'
local luajit_af = {}
local libaf = ffi.load('af')


-- Define ArrayFire functions and types
ffi.cdef[[

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



typedef struct af_array_t* af_array;
typedef struct af_seq {double begin, end, step; unsigned is_gfor;} af_seq;
typedef long long dim_t;
typedef void* af_event;
typedef void* af_random_engine;
typedef void* af_features;
typedef unsigned int af_backend;
typedef void* af_window;

typedef enum {f32, f64, c32, c64, b8, s32, u32, u8, s64, u64, s16, u16, f16} af_dtype;

typedef enum {AF_BINARY_ADD, AF_BINARY_MUL, AF_BINARY_MIN, AF_BINARY_MAX} af_binary_op;
typedef enum {AF_CONV_DEFAULT, AF_CONV_EXPAND} af_conv_mode;
typedef enum {afDevice, afHost} af_source;
typedef enum {AF_CONV_AUTO, AF_CONV_SPATIAL, AF_CONV_FREQ} af_conv_domain;
typedef enum {
    AF_SUCCESS            =   0 
    , AF_ERR_NO_MEM         = 101
    , AF_ERR_DRIVER         = 102
    , AF_ERR_RUNTIME        = 103
    , AF_ERR_INVALID_ARRAY  = 201
    , AF_ERR_ARG            = 202
    , AF_ERR_SIZE           = 203
    , AF_ERR_TYPE           = 204
    , AF_ERR_DIFF_TYPE      = 205
    , AF_ERR_BATCH          = 207
    , AF_ERR_DEVICE         = 208
    , AF_ERR_NOT_SUPPORTED  = 301
    , AF_ERR_NOT_CONFIGURED = 302
    , AF_ERR_NONFREE        = 303
    , AF_ERR_NO_DBL         = 401
    , AF_ERR_NO_GFX         = 402
    , AF_ERR_NO_HALF        = 403
    , AF_ERR_LOAD_LIB       = 501
    , AF_ERR_LOAD_SYM       = 502
    , AF_ERR_ARR_BKND_MISMATCH    = 503
    , AF_ERR_INTERNAL       = 998
    , AF_ERR_UNKNOWN        = 999
} af_err;

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

typedef struct af_index_t {
    union {
        af_array arr;   
        af_seq   seq;   
    } idx;
    bool isSeq;     
    bool isBatch;   
} af_index_t;

typedef struct {int row; int col; const char* title; af_colormap cmap;} af_cell;
typedef struct array {af_array arr;} array;
typedef struct {dim_t dims[4];} dim4;
typedef struct array_proxy_impl array_proxy_impl;
typedef struct {af_array arr;} af_array_struct;
typedef struct {af_features feat;} af_features_struct;
typedef struct {af_index_t impl;} af_index_struct;
typedef struct {af_random_engine engine;} af_random_engine_struct;
typedef struct {af_seq s; size_t size; bool m_gfor;} af_seq_struct;
typedef struct af_event_struct {af_event e;} af_event_struct;
typedef struct {char m_msg[1024]; af_err m_err;} af_exception;

typedef struct array_proxy {
    array_proxy_impl* impl;
    af_array_struct (*operator_array)(struct array_proxy const *self);
    void (*set)(af_array tmp);
    void (*eval)(const struct array_proxy *self);
    af_array_struct (*copy)(const struct array_proxy *self);
    dim_t (*elements)(const struct array_proxy *self);
    af_dtype (*type)(const struct array_proxy *self);
    dim4 (*dims)(const struct array_proxy *self);
    dim_t (*dims_dim)(const struct array_proxy *self, unsigned dim);
    unsigned (*numdims)(const struct array_proxy *self);
    size_t (*bytes)(const struct array_proxy *self);
    size_t (*allocated)(const struct array_proxy *self);
    bool (*isempty)(const struct array_proxy *self);
    bool (*isscalar)(const struct array_proxy *self);
    bool (*isvector)(const struct array_proxy *self);
    bool (*isrow)(const struct array_proxy *self);
    bool (*iscolumn)(const struct array_proxy *self);
    bool (*iscomplex)(const struct array_proxy *self);
    bool (*isreal)(const struct array_proxy *self);
    bool (*isdouble)(const struct array_proxy *self);
    bool (*issingle)(const struct array_proxy *self);
    bool (*ishalf)(const struct array_proxy *self);
    bool (*isrealfloating)(const struct array_proxy *self);
    bool (*isfloating)(const struct array_proxy *self);
    bool (*isinteger)(const struct array_proxy *self);
    bool (*isbool)(const struct array_proxy *self);
    bool (*issparse)(const struct array_proxy *self);
    void (*unlock)(const struct array_proxy *self);
    void (*lock)(const struct array_proxy *self);
    bool (*isLocked)(const struct array_proxy *self);
} array_proxy;

dim_t dim4_elements(dim4* self);
dim_t dim4_ndims(dim4* self);
bool dim4_operator_equals(const dim4* self, const dim4* other);
bool dim4_operator_not_equals(const dim4* self, const dim4* other);
void dim4_operator_multiply_assign(dim4* self, const dim4* other);
void dim4_operator_add_assign(dim4* self, const dim4* other);
void dim4_operator_subtract_assign(dim4* self, const dim4* other);
dim_t* dim4_operator_index(dim4* self, unsigned int dim);
const dim_t* dim4_operator_index_const(const dim4* self, unsigned int dim);
void af_event_free(af_event_struct* self);
void af_event_mark(af_event_struct* self);
void af_event_enqueue(af_event_struct* self);
void af_event_block(const af_event_struct* self);
void af_get_last_error(char **msg, dim_t *len);
const char *af_err_to_string(const af_err err);
void af_exception_free(af_exception* self);
const char* af_exception_what(const af_exception* self);
size_t af_features_getNumFeatures(const af_features_struct* self);
void af_index_free(af_index_struct* self);
bool af_index_isspan(const af_index_struct* self);
void af_random_engine_free(af_random_engine_struct* self);
void af_random_engine_set_type(af_random_engine_struct* self, const af_random_engine_type type);
unsigned long long af_random_engine_get_seed(const af_random_engine_struct* self);
void af_random_engine_set_seed(af_random_engine_struct* self, const unsigned long long seed);

void af_seq_free(af_seq_struct* self);

af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_t * const dims, const af_dtype type);
af_err af_create_handle(af_array *arr, const unsigned ndims, const dim_t * const dims, const af_dtype type);
af_err af_copy_array(af_array *arr, const af_array arrayIn);
af_err af_write_array(af_array arr, const void *data, const size_t bytes, af_source src);
af_err af_get_data_ptr(void *data, const af_array arr);
af_err af_release_array(af_array arr);
af_err af_retain_array(af_array *out, const af_array arrayIn);
af_err af_eval(af_array arrayIn);
af_err af_eval_multiple(const int num, af_array *arrays);
af_err af_set_manual_eval_flag(bool flag);
af_err af_get_manual_eval_flag(bool *flag);
af_err af_get_elements(dim_t *elems, const af_array arr);
af_err af_get_type(af_dtype *type, const af_array arr);
af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3, const af_array arr);
af_err af_get_numdims(unsigned *result, const af_array arr);
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
af_err af_get_scalar(void* output_value, const af_array arr);
dim4 dim4_operator_add(const dim4* first, const dim4* second);
dim4 dim4_operator_subtract(const dim4* first, const dim4* second);
dim4 dim4_operator_multiply(const dim4* first, const dim4* second);
af_err af_create_event(af_event* eventHandle);
af_err af_delete_event(af_event eventHandle);
af_err af_mark_event(const af_event eventHandle);
af_err af_enqueue_wait_event(const af_event eventHandle);
af_err af_block_event(const af_event eventHandle);
af_event_struct* af_event_new();
af_event af_event_get(const af_event_struct* self);
af_exception* af_exception_new();
af_err af_exception_err(af_exception* self);
af_err af_create_features(af_features *feat, dim_t num);
af_err af_retain_features(af_features *out, const af_features feat);
af_err af_get_features_num(dim_t *num, const af_features feat);
af_err af_get_features_xpos(af_array *out, const af_features feat);
af_err af_get_features_ypos(af_array *out, const af_features feat);
af_err af_get_features_score(af_array *score, const af_features feat);
af_err af_get_features_orientation(af_array *orientation, const af_features feat);
af_err af_get_features_size(af_array *size, const af_features feat);
af_err af_release_features(af_features feat);
af_features_struct* af_features_new();
af_features_struct* af_features_new_from_size(size_t n);
af_features_struct* af_features_new_from_af_features(af_features f);
af_array af_features_getX(const af_features_struct* self);
af_array af_features_getY(const af_features_struct* self);
af_array af_features_getScore(const af_features_struct* self);
af_array af_features_getOrientation(const af_features_struct* self);
af_array af_features_getSize(const af_features_struct* self);
af_features af_features_get(const af_features_struct* self);
af_err af_index(af_array *out, const af_array in, const unsigned ndims, const af_seq* const index);
af_err af_lookup(af_array *out, const af_array in, const af_array indices, const unsigned dim);
af_err af_assign_seq(af_array *out, const af_array lhs, const unsigned ndims, const af_seq* const indices, const af_array rhs);
af_err af_index_gen(af_array *out, const af_array in, const dim_t ndims, const af_index_t* indices);
af_err af_assign_gen(af_array *out, const af_array lhs, const dim_t ndims, const af_index_t* indices, const af_array rhs);
af_err af_create_indexers(af_index_t** indexers);
af_err af_set_array_indexer(af_index_t* indexer, const af_array idx, const dim_t dim);
af_err af_set_seq_indexer(af_index_t* indexer, const af_seq* idx, const dim_t dim, const bool is_batch);
af_err af_set_seq_param_indexer(af_index_t* indexer, const double begin, const double end, const double step, const dim_t dim, const bool is_batch);
af_err af_release_indexers(af_index_t* indexers);
af_index_struct* af_index_new();
af_index_struct* af_index_new_from_int(int idx);
af_index_struct* af_index_new_from_seq(const af_seq* s0);
af_index_struct* af_index_new_from_array(const af_array idx0);
af_index_t af_index_get(const af_index_struct* self);
af_err af_create_random_engine(af_random_engine *engine, af_random_engine_type rtype, unsigned long long seed);
af_err af_retain_random_engine(af_random_engine *out, const af_random_engine engine);
af_err af_random_engine_set_type(af_random_engine *engine, const af_random_engine_type rtype);
af_err af_random_engine_get_type(af_random_engine_type *rtype, const af_random_engine engine);
af_err af_random_uniform(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine);
af_err af_random_normal(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine);
af_err af_random_engine_set_seed(af_random_engine *engine, const unsigned long long seed);
af_err af_get_default_random_engine(af_random_engine *engine);
af_err af_set_default_random_engine_type(const af_random_engine_type rtype);
af_err af_random_engine_get_seed(unsigned long long * const seed, af_random_engine engine);
af_err af_release_random_engine(af_random_engine engine);
af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);
af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);
af_err af_set_seed(const unsigned long long seed);
af_err af_get_seed(unsigned long long *seed);
af_random_engine_struct* af_random_engine_new(af_random_engine_type type, unsigned long long seed);
af_random_engine_struct* af_random_engine_new_from_engine(af_random_engine engine);
af_random_engine_type af_random_engine_get_type(const af_random_engine_struct* self);
af_random_engine af_random_engine_get(const af_random_engine_struct* self);
af_seq af_make_seq(double begin, double end, double step);
af_seq_struct* af_seq_new(double begin, double end, double step);
af_seq_struct* af_seq_new_from_seq(af_seq_struct* other, bool is_gfor);
af_seq_struct* af_seq_new_from_af_seq(const af_seq* s);
af_seq af_seq_get(const af_seq_struct* self);
af_seq_struct* af_seq_negate(const af_seq_struct* self);
af_seq_struct* af_seq_add_double(const af_seq_struct* self, double x);
af_seq_struct* af_seq_sub_double(const af_seq_struct* self, double x);
af_seq_struct* af_seq_mul_double(const af_seq_struct* self, double x);
af_err af_abs(af_array *out, const af_array arrayIn);
af_err af_accum(af_array *out, const af_array in, const int dim);
af_err af_acos(af_array *out, const af_array arrayIn);
af_err af_acosh(af_array *out, const af_array arrayIn);
af_err af_add(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_all_true(af_array *out, const af_array in, const int dim);
af_err af_all_true_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_alloc_host(void **ptr, const size_t bytes);
af_err af_alloc_v2(void **ptr, const size_t bytes);
af_err af_and(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_anisotropic_diffusion(af_array *out, const af_array in, const float dt, const float K, const unsigned iterations);
af_err af_any_true_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_anytrue(af_array *out, const af_array in, const int dim);
af_err af_approx1(af_array *out, const af_array in, const af_array pos, const int interp, const float off_grid);
af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1, const int interp, const float off_grid);
af_err af_arg(af_array *out, const af_array arrayIn);
af_err af_asin(af_array *out, const af_array arrayIn);
af_err af_asinh(af_array *out, const af_array arrayIn);
af_err af_assign_seq(af_array *out, const af_array lhs, const af_seq* const indices, const af_array rhs);
af_err af_atan(af_array *out, const af_array arrayIn);
af_err af_atan2(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_atanh(af_array *out, const af_array arrayIn);
af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma);
af_err af_bitand(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_bitnot(af_array *out, const af_array arrayIn);
af_err af_bitor(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_bitshiftl(af_array *out, const af_array lhs, const af_array rhs);
af_err af_bitshiftr(af_array *out, const af_array lhs, const af_array rhs);
af_err af_bitxor(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_cast(af_array *out, const af_array in, const af_dtype type);
af_err af_cbrt(af_array *out, const af_array arrayIn);
af_err af_ceil(af_array *out, const af_array arrayIn);
af_err af_cholesky(af_array *out, int *info, const af_array in, const bool is_upper);
af_err af_clamp(af_array *out, const af_array in, const af_array lo, const af_array hi);
af_err af_col(af_array *out, const af_array in, const dim_t col);
af_err af_cols(af_array *out, const af_array in, const dim_t first, const dim_t last);
af_err af_complex(af_array *out, const af_array real, const af_array imag);
af_err af_confidence_cc(af_array *out, const af_array in, const af_array template, const af_array disparity, const af_array mean, const float radius);
af_err af_conjg(af_array *out, const af_array arrayIn);
af_err af_constant(af_array *out, const double val, const dim_t d0, const af_dtype type);
af_err af_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, const af_conv_domain domain);
af_err af_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, const af_conv_domain domain);
af_err af_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, const af_conv_domain domain);
af_err af_corrcoef(double *out, const af_array X, const af_array Y);
af_err af_cos(af_array *out, const af_array arrayIn);
af_err af_cosh(af_array *out, const af_array arrayIn);
af_err af_count(af_array *out, const af_array in, const int dim);
af_err af_count_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_cov(af_array *out, const af_array X, const af_array Y, const bool is_biased);
af_err af_delete_image_mem(af_array arrayIn);
af_err af_dense(af_array *out, const af_array sparse);
af_err af_det(double *out, const af_array arrayIn);
af_err af_device_info(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);
af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers, size_t *lock_bytes, size_t *lock_buffers);
af_err af_diag(af_array *out, const af_array in, const int num);
af_err af_diff1(af_array *out, const af_array in, const int dim);
af_err af_diff2(af_array *out, const af_array in, const int dim);
af_err af_dilate(af_array *out, const af_array in, const af_array mask);
af_err af_dilate3d(af_array *out, const af_array in, const af_array mask);
af_err af_div(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_dog(af_array *out, const af_array in, const int radius1, const int radius2);
af_err af_dot(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_eq(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_erf(af_array *out, const af_array arrayIn);
af_err af_erfc(af_array *out, const af_array arrayIn);
af_err af_erode(af_array *out, const af_array in, const af_array mask);
af_err af_erode3d(af_array *out, const af_array in, const af_array mask);
af_err af_exp(af_array *out, const af_array arrayIn);
af_err af_expm1(af_array *out, const af_array arrayIn);
af_err af_factorial(af_array *out, const af_array arrayIn);
af_err af_fast(af_array *out, const af_array in, const float thr, const unsigned arc_length, const bool non_max, const float feature_ratio, const unsigned edge);
af_err af_features(af_array *out, const af_array arrayIn);
af_err af_fft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);
af_err af_fft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);
af_err af_fft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);
af_err af_fft_c2r(af_array *out, const af_array in, const double norm_factor, const bool is_odd);
af_err af_fft_r2c(af_array *out, const af_array in, const double norm_factor);
af_err af_fir(af_array *out, const af_array b, const af_array x);
af_err af_flat(af_array *out, const af_array arrayIn);
af_err af_flip(af_array *out, const af_array in, const unsigned dim);
af_err af_floor(af_array *out, const af_array arrayIn);
af_err af_free_host(void *ptr);
af_err af_free_pinned(void *ptr);
af_err af_free_v2(void *ptr);
af_err af_gaussian_kernel(af_array *out, const int rows, const int cols, const double sigma_r, const double sigma_c);
af_err af_ge(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_get_active_backend(af_backend *backend);
af_err af_get_available_backends(unsigned *backends);
af_err af_get_backend_count(unsigned *num_backends);
af_err af_get_backend_id(af_backend *backend, const af_array arrayIn);
af_err af_get_default_random_engine(af_array *engine);
af_err af_get_device(int *device);
af_err af_get_device_count(int *nDevices);
af_err af_get_device_id(int *device, const af_array arrayIn);
af_err af_get_seed(unsigned long long *seed);
af_err af_gloh(af_array *out, const af_array arrayIn);
af_err af_grad(af_array *dx, af_array *dy, const af_array arrayIn);
af_err af_gray2rgb(af_array *out, const af_array in, const float r, const float g, const float b);
af_err af_gt(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_hamming_matcher(af_array *out, const af_array query, const af_array train, const dim_t dist_dim, const unsigned n_dist);
af_err af_harris(af_array *out, const af_array in, const unsigned max_corners, const double min_response, const double sigma, const unsigned filter_len, const double k_thr);
af_err af_histequal(af_array *out, const af_array in, const af_array hist);
af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval);
af_err af_hsv2rgb(af_array *out, const af_array arrayIn);
af_err af_hypot(af_array *out, const af_array lhs, const af_array rhs);
af_err af_identity(af_array *out, const dim_t dim0, const af_dtype type);
af_err af_ifft(af_array *out, const af_array in, const double norm_factor, const dim_t odim0);
af_err af_ifft2(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1);
af_err af_ifft3(af_array *out, const af_array in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2);
af_err af_iir(af_array *y, af_array *x, const af_array b, const af_array a);
af_err af_imag(af_array *out, const af_array arrayIn);
af_err af_index(af_array *out, const af_array in, const af_index_t* indices, const unsigned ndims);
af_err af_info(char* info_string);
af_err af_info_string(char** str, const bool verbose);
af_err af_iota(af_array *out, const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const af_dtype type);
af_err af_is_double_available(int *out, const int device);
af_err af_is_half_available(int *out, const int device);
af_err af_is_image_io_available(bool *out);
af_err af_is_lapack_available(bool *out);
af_err af_isinf(af_array *out, const af_array arrayIn);
af_err af_isnan(af_array *out, const af_array arrayIn);
af_err af_iszero(af_array *out, const af_array arrayIn);
af_err af_iterative_deconv(af_array *out, const af_array in, const af_array ker, const float gamma, const unsigned max_iter);
af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);
af_err af_le(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_lgamma(af_array *out, const af_array arrayIn);
af_err af_load_image(af_array *out, const char *filename, const bool is_color);
af_err af_load_image_memory(af_array *out, const void *ptr);
af_err af_log(af_array *out, const af_array arrayIn);
af_err af_log10(af_array *out, const af_array arrayIn);
af_err af_log1p(af_array *out, const af_array arrayIn);
af_err af_log2(af_array *out, const af_array arrayIn);
af_err af_lookup(af_array *out, const af_array in, const af_array indices, const unsigned dim);
af_err af_lower(af_array *out, const af_array in, const bool is_unit_diag);
af_err af_lt(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_lu(af_array *lower, af_array *upper, af_array *pivot, const af_array arrayIn);
af_err af_max(af_array *out, const af_array in, const int dim);
af_err af_max_all(double *real, double *imag, const af_array arrayIn);
af_err af_max_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_maxfilt(af_array *out, const af_array in, const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const bool is_border);
af_err af_mean(af_array *out, const af_array in, const int dim);
af_err af_meanshift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);
af_err af_median(af_array *out, const af_array in, const int dim);
af_err af_min(af_array *out, const af_array in, const int dim);
af_err af_min_all(double *real, double *imag, const af_array arrayIn);
af_err af_min_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_minfilt(af_array *out, const af_array in, const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const bool is_border);
af_err af_mod(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_t * const dims);
af_err af_mul(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_neg(af_array *out, const af_array arrayIn);
af_err af_neq(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_not(af_array *out, const af_array arrayIn);
af_err af_or(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_orb(af_array *out, af_array *desc, const af_array in, const af_array mask, const float fast_thr, const unsigned max_feat, const float scl_fctr, const unsigned levels);
af_err af_pinned(af_array *out, const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const af_dtype type);
af_err af_pinverse(af_array *out, const af_array in, const double tol);
af_err af_pow(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_pow2(af_array *out, const af_array arrayIn);
af_err af_print_array(const af_array arr);
af_err af_product(af_array *out, const af_array in, const int dim);
af_err af_product_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array arrayIn);
af_err af_randn(af_array *out, const dim_t d0, const af_dtype type);
af_err af_random_engine(af_array *out, const af_random_engine_type type, const unsigned long long seed);
af_err af_randu(af_array *out, const dim_t d0, const af_dtype type);
af_err af_range(af_array *out, const dim_t d0, const int seq_dim, const double begin, const double step, const af_dtype type);
af_err af_rank(unsigned *rank, const af_array in, const double tol);
af_err af_read_array(af_array *out, const char *filename, const bool is_column);
af_err af_real(af_array *out, const af_array arrayIn);
af_err af_rem(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_reorder(af_array *out, const af_array in, const unsigned x, const unsigned y, const unsigned z, const unsigned w);
af_err af_replace(af_array a, const af_array cond, const af_array b);
af_err af_rgb2gray(af_array *out, const af_array in, const float rWeight, const float gWeight, const float bWeight);
af_err af_rgb2hsv(af_array *out, const af_array arrayIn);
af_err af_root(af_array *out, const af_array in, const double n);
af_err af_round(af_array *out, const af_array arrayIn);
af_err af_row(af_array *out, const af_array in, const dim_t row);
af_err af_rows(af_array *out, const af_array in, const dim_t first, const dim_t last);
af_err af_rsqrt(af_array *out, const af_array arrayIn);
af_err af_sat(af_array *out, const af_array arrayIn);
af_err af_save_array(const char *filename, const af_array arrayIn);
af_err af_save_image(const char *filename, const af_array arrayIn);
af_err af_scan(af_array *out, const af_array in, const int dim, const af_binary_op op, const bool inclusive_scan);
af_err af_scan_by_key(af_array *out, const af_array keys, const af_array in, const int dim, const af_binary_op op, const bool inclusive_scan);
af_err af_select(af_array *out, const af_array cond, const af_array a, const af_array b);
af_err af_set_backend(const af_backend backend);
af_err af_set_default_random_engine_type(const af_random_engine_type type);
af_err af_set_device(const int device);
af_err af_set_seed(const unsigned long long seed);
af_err af_setintersect(af_array *out, const af_array a, const af_array b, const bool is_unique);
af_err af_setunion(af_array *out, const af_array a, const af_array b, const bool is_unique);
af_err af_setunique(af_array *out, const af_array in, const bool is_sorted);
af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);
af_err af_sift(af_array *out, const af_array in, const unsigned n_layers, const float contrast_thr, const float edge_thr, const float init_sigma);
af_err af_sigmoid(af_array *out, const af_array arrayIn);
af_err af_sign(af_array *out, const af_array arrayIn);
af_err af_sin(af_array *out, const af_array arrayIn);
af_err af_sinh(af_array *out, const af_array arrayIn);
af_err af_skew(af_array *out, const af_array in, const int dim, const bool bias);
af_err af_slice(af_array *out, const af_array in, const dim_t idx);
af_err af_slices(af_array *out, const af_array in, const dim_t first, const dim_t last);
af_err af_sobel(af_array *dx, af_array *dy, const af_array in, const unsigned w_size);
af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool ascending);
af_err af_sort_by_key(af_array *out_keys, af_array *out_values, const af_array keys, const af_array values, const unsigned dim, const bool ascending);
af_err af_sort_index(af_array *out_indices, const af_array in, const unsigned dim, const bool ascending);
af_err af_sparse_get_coldx(af_array *out, const af_array arrayIn);
af_err af_sparse_get_nnz(dim_t *out, const af_array arrayIn);
af_err af_sparse_get_row_idx(af_array *out, const af_array arrayIn);
af_err af_sparse_get_values(af_array *out, const af_array arrayIn);
af_err af_sqrt(af_array *out, const af_array arrayIn);
af_err af_stdev(af_array *out, const af_array in, const int dim);
af_err af_sub(af_array *out, const af_array lhs, const af_array rhs, const unsigned dim);
af_err af_sum(af_array *out, const af_array in, const int dim);
af_err af_sum_by_key(af_array *out, af_array *keys_out, const af_array in, const af_array key);
af_err af_susan(af_array *out, const af_array in, const unsigned radius, const float diff_thr, const float geom_thr, const float feature_ratio, const unsigned edge);
af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array arrayIn);
af_err af_sync(const int device);
af_err af_tan(af_array *out, const af_array arrayIn);
af_err af_tanh(af_array *out, const af_array arrayIn);
af_err af_tgamma(af_array *out, const af_array arrayIn);
af_err af_tile(af_array *out, const af_array in, const dim_t x, const dim_t y, const dim_t z, const dim_t w);
af_err af_to_string(char **output, const char *exp, const af_array arr);
af_err af_transform_coordinates(af_array *out, const af_array tf, const float d0, const float d1);
af_err af_transpose(af_array *out, const af_array in, const bool conjugate);
af_err af_trunc(af_array *out, const af_array arrayIn);
af_err af_unwrap(af_array *out, const af_array in, const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const bool is_column);
af_err af_upper(af_array *out, const af_array in, const bool is_unit_diag);
af_err af_var(af_array *out, const af_array in, const bool is_biased, const int dim);
af_err af_where(af_array *idx, const af_array arrayIn);
af_err af_wrap(af_array *out, const af_array in, const dim_t ox, const dim_t oy, const dim_t wx, dim_t wy, const dim_t sx,  dim_t sy, const dim_t px,  dim_t py, const bool is_column);
af_err af_canny(af_array *out, const af_array in, const af_canny_threshold thresholdType, const float low, const float high, const unsigned isfGaussian, const float sigma);
af_err af_color_space(af_array *out, const af_array in, const af_cspace_t to, const af_cspace_t from);
af_err af_homography(af_array *out, float *inliers, const af_array x_src, const af_array y_src, const af_array x_dst, const af_array y_dst, const af_homography_type htype, const float inlier_thr, const unsigned iterations);
af_err af_inverse(af_array *out, const af_array in, const af_mat_prop options);
af_err af_inverse_deconv(af_array *out, const af_array in, const af_array psf, const double gamma, const af_inverse_deconv_algo algo);
af_err af_match_template(af_array *out, const af_array search_img, const af_array template_img, const af_match_type mtype);
af_err af_matmul(af_array *out, const af_array lhs, const af_array rhs, const af_mat_prop opt_lhs, const af_mat_prop opt_rhs);
af_err af_medfilt(af_array *out, const af_array in, const dim_t wx, const dim_t wy, const af_border_type edge_pad);
af_err af_moments(af_array *out, const af_array in, const af_moment_type moment);
af_err af_nearest_neighbour(af_array *idx, const af_array query, const af_array train, const dim_t dist_dim, const unsigned k, const af_match_type dist_type);
af_err af_norm(af_array *out, const af_array in, const af_norm_type type, const double p, const double q, const int dim);
af_err af_pad(af_array *out, const af_array in, const dim_t begin_padding, const dim_t end_padding, const af_border_type border_type);
af_err af_regions(af_array *out, const af_array in, const af_connectivity connectivity, const af_dtype type);
af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1, const dim_t odim0, const dim_t odim1, const af_interp_type method);
af_err af_transform(af_array *out, const af_array in, const af_array tf, const dim_t odim0, const dim_t odim1, const af_interp_type method, const bool inverse);
af_err af_resize(af_array *out, const af_array in, const dim_t od0, const dim_t od1, const af_interp_type method);
af_err af_rotate(af_array *out, const af_array in, const float theta, const bool crop, const af_interp_type method);
af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1, const dim_t od0, const dim_t od1, const af_interp_type method);
af_err af_ycbcr2rgb(af_array *out, const af_array in, const af_ycc_std standard);
af_err af_rgb2ycbcr(af_array *out, const af_array in, const af_ycc_std standard);
af_err af_save_image_memory(void **out, const af_array in, const af_image_format format);
af_err af_solve(af_array *out, const af_array a, const af_array b, const af_mat_prop options);
af_err af_solve_lu(af_array *out, const af_array a, const af_array piv, const af_array b, const af_mat_prop options);
af_err af_sparse_get_storage(af_storage *out, const af_array arrayIn);
af_err af_sparse_get_info(af_array *out_values, af_array *row_idx, af_array *col_idx, af_array *dims, af_storage *stype, const af_array arrayIn);
af_err af_sparse_convert_to(af_array *out, const af_array in, const af_storage stype);
af_err af_sparse(af_array *out, const af_array values, const af_array row_idx, const af_array col_idx, const af_array dims, const af_storage stype);
af_err af_topk(af_array *values, af_array *indices, const af_array in, const unsigned k, const int dim, const af_topk_function function);
af_err af_create_window(af_window *out, const int width, const int height, const char* const title);
af_err af_set_position(const af_window wind, const unsigned x, const unsigned y);
af_err af_set_title(const af_window wind, const char* const title);
af_err af_set_size(const af_window wind, const unsigned w, const unsigned h);
af_err af_draw_image(const af_window wind, const af_array in, const af_cell* const props);
af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y, const af_marker_type marker, const af_cell* const props);
af_err af_draw_hist(const af_window wind, const af_array X, const double minval, const double maxval, const af_cell* const props);
af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell* const props);
af_err af_draw_vector_field_2d(const af_window wind, const af_array xPoints, const af_array yPoints, const af_array xDirs, const af_array yDirs, const af_cell* const props);
af_err af_grid(const af_window wind, const int rows, const int cols);
af_err af_show(const af_window wind);
af_err af_is_window_closed(bool *out, const af_window wind);
af_err af_set_visibility(const af_window wind, const bool is_visible);
af_err af_destroy_window(const af_window wind);
af_err af_set_axes_limits_2d(const af_window wind, const float xmin, const float xmax, const float ymin, const float ymax, const bool exact, const af_cell* const props);
af_err af_set_axes_titles(const af_window wind, const char * const xtitle, const char * const ytitle, const char * const ztitle, const af_cell* const props);
af_err af_set_axes_label_format(const af_window wind, const char *const xformat, const char *const yformat, const char *const zformat, const af_cell *const props);
af_err af_draw_image(const af_window wind, const af_array in, const af_cell *const props);
af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y, const af_cell *const props);
af_err af_draw_plot3(const af_window wind, const af_array P, const af_cell *const props);
af_err af_draw_plot_nd(const af_window wind, const af_array P, const af_cell *const props);
af_err af_draw_plot_2d(const af_window wind, const af_array X, const af_array Y, const af_cell *const props);
af_err af_draw_plot_3d(const af_window wind, const af_array X, const af_array Y, const af_array Z, const af_cell *const props);
af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y, const af_marker_type marker, const af_cell *const props);
af_err af_draw_scatter3(const af_window wind, const af_array P, const af_marker_type marker, const af_cell *const props);
af_err af_draw_scatter_nd(const af_window wind, const af_array P, const af_marker_type marker, const af_cell *const props);
af_err af_draw_scatter_2d(const af_window wind, const af_array X, const af_array Y, const af_marker_type marker, const af_cell *const props);
af_err af_draw_scatter_3d(const af_window wind, const af_array X, const af_array Y, const af_array Z, af_marker_type marker, af_cell *const props);
af_err af_draw_hist(const af_window wind, const af_array X, const double minval, const double maxval, const af_cell *const props);
af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell *const props);
af_err af_draw_vector_field_nd(const af_window wind, af_array points, af_array directions, af_cell *const props);
af_err af_draw_vector_field_3d(const af_window wind, af_array xPoints, af_array yPoints, af_array zPoints, af_array xDirs, af_array yDirs, af_array zDirs, af_cell *const props);
af_err af_draw_vector_field_2d(const af_window wind, af_array xPoints, af_array yPoints, af_array xDirs, af_array yDirs, af_cell *const props);

]]


-- Function implementations
function luajit_af.af_create_array(arr, data, ndims, dims, dtype)
    local err = libaf.af_create_array(arr, data, ndims, dims, dtype)
    if err ~= 0 then
        error("Error in af_create_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_create_handle(arr, ndims, dims, dtype)
    local err = libaf.af_create_handle(arr, ndims, dims, dtype)
    if err ~= 0 then
        error("Error in af_create_handle: " .. tostring(err))
    end
    return err
end

function luajit_af.af_copy_array(arr, arrayIn)
    local err = libaf.af_copy_array(arr, arrayIn)
    if err ~= 0 then
        error("Error in af_copy_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_write_array(arr, data, bytes, src)
    local err = libaf.af_write_array(arr, data, bytes, src)
    if err ~= 0 then
        error("Error in af_write_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_data_ptr(data, arr)
    local err = libaf.af_get_data_ptr(data, arr)
    if err ~= 0 then
        error("Error in af_get_data_ptr: " .. tostring(err))
    end
    return err
end

function luajit_af.af_release_array(arr)
    local err = libaf.af_release_array(arr)
    if err ~= 0 then
        print(err)
        error("Error in af_release_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_retain_array(out, arrayIn)
    local err = libaf.af_retain_array(out, arrayIn)
    if err ~= 0 then
        error("Error in af_retain_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_eval(arrayIn)
    local err = libaf.af_eval(arrayIn)
    if err ~= 0 then
        error("Error in af_eval: " .. tostring(err))
    end
    return err
end

function luajit_af.af_eval_multiple(num, arrays)
    local err = libaf.af_eval_multiple(num, arrays)
    if err ~= 0 then
        error("Error in af_eval_multiple: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_manual_eval_flag(flag)
    local err = libaf.af_set_manual_eval_flag(flag)
    if err ~= 0 then
        error("Error in af_set_manual_eval_flag: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_manual_eval_flag()
    local flag = ffi.new("bool[1]")
    local err = libaf.af_get_manual_eval_flag(flag)
    if err ~= 0 then
        error("Error in af_get_manual_eval_flag: " .. tostring(err))
    end
    return flag[0], err
end

function luajit_af.af_get_elements(arr)
    local elems = ffi.new("dim_t[1]")
    local err = libaf.af_get_elements(elems, arr)
    if err ~= 0 then
        error("Error in af_get_elements: " .. tostring(err))
    end
    return tonumber(elems[0]), err
end

function luajit_af.af_get_type(arr)
    local type = ffi.new("af_dtype[1]")
    local err = libaf.af_get_type(type, arr)
    if err ~= 0 then
        error("Error in af_get_type: " .. tostring(err))
    end
    return type[0], err
end

function luajit_af.af_get_dims(arr)
    local d0 = ffi.new("dim_t[1]")
    local d1 = ffi.new("dim_t[1]")
    local d2 = ffi.new("dim_t[1]")
    local d3 = ffi.new("dim_t[1]")
    local err = libaf.af_get_dims(d0, d1, d2, d3, arr)
    if err ~= 0 then
        error("Error in af_get_dims: " .. tostring(err))
    end
    return tonumber(d0[0]), tonumber(d1[0]), tonumber(d2[0]), tonumber(d3[0]), err
end

function luajit_af.af_get_numdims(arr)
    local result = ffi.new("unsigned[1]")
    local err = libaf.af_get_numdims(result, arr)
    if err ~= 0 then
        error("Error in af_get_numdims: " .. tostring(err))
    end
    return tonumber(result[0]), err
end

function luajit_af.af_is_empty(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_empty(result, arr)
    if err ~= 0 then
        error("Error in af_is_empty: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_scalar(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_scalar(result, arr)
    if err ~= 0 then
        error("Error in af_is_scalar: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_row(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_row(result, arr)
    if err ~= 0 then
        error("Error in af_is_row: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_column(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_column(result, arr)
    if err ~= 0 then
        error("Error in af_is_column: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_vector(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_vector(result, arr)
    if err ~= 0 then
        error("Error in af_is_vector: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_complex(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_complex(result, arr)
    if err ~= 0 then
        error("Error in af_is_complex: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_real(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_real(result, arr)
    if err ~= 0 then
        error("Error in af_is_real: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_double(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_double(result, arr)
    if err ~= 0 then
        error("Error in af_is_double: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_single(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_single(result, arr)
    if err ~= 0 then
        error("Error in af_is_single: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_half(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_half(result, arr)
    if err ~= 0 then
        error("Error in af_is_half: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_realfloating(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_realfloating(result, arr)
    if err ~= 0 then
        error("Error in af_is_realfloating: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_floating(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_floating(result, arr)
    if err ~= 0 then
        error("Error in af_is_floating: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_integer(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_integer(result, arr)
    if err ~= 0 then
        error("Error in af_is_integer: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_bool(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_bool(result, arr)
    if err ~= 0 then
        error("Error in af_is_bool: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_is_sparse(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_sparse(result, arr)
    if err ~= 0 then
        error("Error in af_is_sparse: " .. tostring(err))
    end
    return result[0], err
end

function luajit_af.af_get_scalar(output_value, arr)
    local err = libaf.af_get_scalar(output_value, arr)
    if err ~= 0 then
        error("Error in af_get_scalar: " .. tostring(err))
    end
    return err
end

function luajit_af.dim4_operator_add(first, second)
    return libaf.dim4_operator_add(first, second)
end

function luajit_af.dim4_operator_subtract(first, second)
    return libaf.dim4_operator_subtract(first, second)
end

function luajit_af.dim4_operator_multiply(first, second)
    return libaf.dim4_operator_multiply(first, second)
end

function luajit_af.af_create_event(eventHandle)
    local err = libaf.af_create_event(eventHandle)
    if err ~= 0 then
        error("Error in af_create_event: " .. tostring(err))
    end
    return err
end

function luajit_af.af_delete_event(eventHandle)
    local err = libaf.af_delete_event(eventHandle)
    if err ~= 0 then
        error("Error in af_delete_event: " .. tostring(err))
    end
    return err
end

function luajit_af.af_mark_event(eventHandle)
    local err = libaf.af_mark_event(eventHandle)
    if err ~= 0 then
        error("Error in af_mark_event: " .. tostring(err))
    end
    return err
end

function luajit_af.af_enqueue_wait_event(eventHandle)
    local err = libaf.af_enqueue_wait_event(eventHandle)
    if err ~= 0 then
        error("Error in af_enqueue_wait_event: " .. tostring(err))
    end
    return err
end

function luajit_af.af_block_event(eventHandle)
    local err = libaf.af_block_event(eventHandle)
    if err ~= 0 then
        error("Error in af_block_event: " .. tostring(err))
    end
    return err
end

function luajit_af.af_event_new()
    return libaf.af_event_new()
end

function luajit_af.af_event_get(self)
    return libaf.af_event_get(self)
end

function luajit_af.af_exception_new()
    return libaf.af_exception_new()
end

function luajit_af.af_exception_err(self)
    local err = libaf.af_exception_err(self)
    if err ~= 0 then
        error("Error in af_exception_err: " .. tostring(err))
    end
    return err
end

function luajit_af.af_create_features(feat, num)
    local err = libaf.af_create_features(feat, num)
    if err ~= 0 then
        error("Error in af_create_features: " .. tostring(err))
    end
    return err
end

function luajit_af.af_retain_features(out, feat)
    local err = libaf.af_retain_features(out, feat)
    if err ~= 0 then
        error("Error in af_retain_features: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_features_num(feat)
    local num = ffi.new("dim_t[1]")
    local err = libaf.af_get_features_num(num, feat)
    if err ~= 0 then
        error("Error in af_get_features_num: " .. tostring(err))
    end
    return tonumber(num[0]), err
end

function luajit_af.af_get_features_xpos(out, feat)
    local err = libaf.af_get_features_xpos(out, feat)
    if err ~= 0 then
        error("Error in af_get_features_xpos: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_features_ypos(out, feat)
    local err = libaf.af_get_features_ypos(out, feat)
    if err ~= 0 then
        error("Error in af_get_features_ypos: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_features_score(score, feat)
    local err = libaf.af_get_features_score(score, feat)
    if err ~= 0 then
        error("Error in af_get_features_score: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_features_orientation(orientation, feat)
    local err = libaf.af_get_features_orientation(orientation, feat)
    if err ~= 0 then
        error("Error in af_get_features_orientation: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_features_size(size, feat)
    local err = libaf.af_get_features_size(size, feat)
    if err ~= 0 then
        error("Error in af_get_features_size: " .. tostring(err))
    end
    return err
end

function luajit_af.af_release_features(feat)
    local err = libaf.af_release_features(feat)
    if err ~= 0 then
        error("Error in af_release_features: " .. tostring(err))
    end
    return err
end

function luajit_af.af_features_getX(self)
    return libaf.af_features_getX(self)
end

function luajit_af.af_features_getY(self)
    return libaf.af_features_getY(self)
end

function luajit_af.af_features_getScore(self)
    return libaf.af_features_getScore(self)
end

function luajit_af.af_features_getOrientation(self)
    return libaf.af_features_getOrientation(self)
end

function luajit_af.af_features_getSize(self)
    return libaf.af_features_getSize(self)
end

function luajit_af.af_features_get(self)
    return libaf.af_features_get(self)
end

function luajit_af.af_index(out, in_array, ndims, index)
    local err = libaf.af_index(out, in_array, ndims, index)
    if err ~= 0 then
        error("Error in af_index: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lookup(out, in_array, indices, dim)
    local err = libaf.af_lookup(out, in_array, indices, dim)
    if err ~= 0 then
        error("Error in af_lookup: " .. tostring(err))
    end
    return err
end

function luajit_af.af_assign_seq(out, lhs, ndims, indices, rhs)
    local err = libaf.af_assign_seq(out, lhs, ndims, indices, rhs)
    if err ~= 0 then
        error("Error in af_assign_seq: " .. tostring(err))
    end
    return err
end

function luajit_af.af_index_gen(out, in_array, ndims, indices)
    local err = libaf.af_index_gen(out, in_array, ndims, indices)
    if err ~= 0 then
        error("Error in af_index_gen: " .. tostring(err))
    end
    return err
end

function luajit_af.af_assign_gen(out, lhs, ndims, indices, rhs)
    local err = libaf.af_assign_gen(out, lhs, ndims, indices, rhs)
    if err ~= 0 then
        error("Error in af_assign_gen: " .. tostring(err))
    end
    return err
end

function luajit_af.af_create_indexers(indexers)
    local err = libaf.af_create_indexers(indexers)
    if err ~= 0 then
        error("Error in af_create_indexers: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_array_indexer(indexer, idx, dim)
    local err = libaf.af_set_array_indexer(indexer, idx, dim)
    if err ~= 0 then
        error("Error in af_set_array_indexer: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_seq_indexer(indexer, idx, dim, is_batch)
    local err = libaf.af_set_seq_indexer(indexer, idx, dim, is_batch)
    if err ~= 0 then
        error("Error in af_set_seq_indexer: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_seq_param_indexer(indexer, begin, seq_end, step, dim, is_batch)
    local err = libaf.af_set_seq_param_indexer(indexer, begin, seq_end, step, dim, is_batch)
    if err ~= 0 then
        error("Error in af_set_seq_param_indexer: " .. tostring(err))
    end
    return err
end

function luajit_af.af_release_indexers(indexers)
    local err = libaf.af_release_indexers(indexers)
    if err ~= 0 then
        error("Error in af_release_indexers: " .. tostring(err))
    end
    return err
end

function luajit_af.af_index_get(self)
    return libaf.af_index_get(self)
end

function luajit_af.af_create_random_engine(engine, rtype, seed)
    local err = libaf.af_create_random_engine(engine, rtype, seed)
    if err ~= 0 then
        error("Error in af_create_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_retain_random_engine(out, engine)
    local err = libaf.af_retain_random_engine(out, engine)
    if err ~= 0 then
        error("Error in af_retain_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_engine_set_type(engine, rtype)
    local err = libaf.af_random_engine_set_type(engine, rtype)
    if err ~= 0 then
        error("Error in af_random_engine_set_type: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_engine_get_type(engine)
    local rtype = ffi.new("af_random_engine_type[1]")
    local err = libaf.af_random_engine_get_type(rtype, engine)
    if err ~= 0 then
        error("Error in af_random_engine_get_type: " .. tostring(err))
    end
    return rtype[0], err
end

function luajit_af.af_random_uniform(out, ndims, dims, type, engine)
    local err = libaf.af_random_uniform(out, ndims, dims, type, engine)
    if err ~= 0 then
        error("Error in af_random_uniform: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_normal(out, ndims, dims, type, engine)
    local err = libaf.af_random_normal(out, ndims, dims, type, engine)
    if err ~= 0 then
        error("Error in af_random_normal: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_engine_set_seed(engine, seed)
    local err = libaf.af_random_engine_set_seed(engine, seed)
    if err ~= 0 then
        error("Error in af_random_engine_set_seed: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_default_random_engine(engine)
    local err = libaf.af_get_default_random_engine(engine)
    if err ~= 0 then
        error("Error in af_get_default_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_default_random_engine_type(rtype)
    local err = libaf.af_set_default_random_engine_type(rtype)
    if err ~= 0 then
        error("Error in af_set_default_random_engine_type: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_engine_get_seed(engine)
    local seed = ffi.new("unsigned long long[1]")
    local err = libaf.af_random_engine_get_seed(seed, engine)
    if err ~= 0 then
        error("Error in af_random_engine_get_seed: " .. tostring(err))
    end
    return seed[0], err
end

function luajit_af.af_release_random_engine(engine)
    local err = libaf.af_release_random_engine(engine)
    if err ~= 0 then
        error("Error in af_release_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_randu(out, ndims, dims, type)
    local err = libaf.af_randu(out, ndims, dims, type)
    if err ~= 0 then
        error("Error in af_randu: " .. tostring(err))
    end
    return err
end

function luajit_af.af_randn(out, ndims, dims, type)
    local err = libaf.af_randn(out, ndims, dims, type)
    if err ~= 0 then
        error("Error in af_randn: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_seed(seed)
    local err = libaf.af_set_seed(seed)
    if err ~= 0 then
        error("Error in af_set_seed: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_seed()
    local seed = ffi.new("unsigned long long[1]")
    local err = libaf.af_get_seed(seed)
    if err ~= 0 then
        error("Error in af_get_seed: " .. tostring(err))
    end
    return seed[0], err
end

function luajit_af.af_random_engine_new(type, seed)
    return libaf.af_random_engine_new(type, seed)
end

function luajit_af.af_random_engine_new_from_engine(engine)
    return libaf.af_random_engine_new_from_engine(engine)
end

function luajit_af.af_random_engine_get_type(self)
    return libaf.af_random_engine_get_type(self)
end

function luajit_af.af_random_engine_get(self)
    return libaf.af_random_engine_get(self)
end

function luajit_af.af_make_seq(begin, seq_end, step)
    return libaf.af_make_seq(begin, seq_end, step)
end

function luajit_af.af_seq_new(begin, seq_end, step)
    return libaf.af_seq_new(begin, seq_end, step)
end

function luajit_af.af_seq_new_from_seq(other, is_gfor)
    return libaf.af_seq_new_from_seq(other, is_gfor)
end

function luajit_af.af_seq_new_from_af_seq(s)
    return libaf.af_seq_new_from_af_seq(s)
end

function luajit_af.af_seq_get(self)
    return libaf.af_seq_get(self)
end

function luajit_af.af_seq_negate(self)
    return libaf.af_seq_negate(self)
end

function luajit_af.af_seq_add_double(self, x)
    return libaf.af_seq_add_double(self, x)
end

function luajit_af.af_seq_sub_double(self, x)
    return libaf.af_seq_sub_double(self, x)
end

function luajit_af.af_seq_mul_double(self, x)
    return libaf.af_seq_mul_double(self, x)
end

function luajit_af.af_abs(out, arrayIn)
    local err = libaf.af_abs(out, arrayIn)
    if err ~= 0 then
        error("Error in af_abs: " .. tostring(err))
    end
    return err
end

function luajit_af.af_accum(out, in_array, dim)
    local err = libaf.af_accum(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_accum: " .. tostring(err))
    end
    return err
end

function luajit_af.af_acos(out, arrayIn)
    local err = libaf.af_acos(out, arrayIn)
    if err ~= 0 then
        error("Error in af_acos: " .. tostring(err))
    end
    return err
end

function luajit_af.af_acosh(out, arrayIn)
    local err = libaf.af_acosh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_acosh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_add(out, lhs, rhs, dim)
    local err = libaf.af_add(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_add: " .. tostring(err))
    end
    return err
end

function luajit_af.af_all_true(out, in_array, dim)
    local err = libaf.af_all_true(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_all_true: " .. tostring(err))
    end
    return err
end

function luajit_af.af_all_true_by_key(out, keys_out, in_array, key)
    local err = libaf.af_all_true_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_all_true_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_alloc_host(ptr, bytes)
    local err = libaf.af_alloc_host(ptr, bytes)
    if err ~= 0 then
        error("Error in af_alloc_host: " .. tostring(err))
    end
    return err
end

function luajit_af.af_alloc_v2(ptr, bytes)
    local err = libaf.af_alloc_v2(ptr, bytes)
    if err ~= 0 then
        error("Error in af_alloc_v2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_and(out, lhs, rhs, dim)
    local err = libaf.af_and(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_and: " .. tostring(err))
    end
    return err
end

function luajit_af.af_anisotropic_diffusion(out, in_array, dt, K, iterations)
    local err = libaf.af_anisotropic_diffusion(out, in_array, dt, K, iterations)
    if err ~= 0 then
        error("Error in af_anisotropic_diffusion: " .. tostring(err))
    end
    return err
end

function luajit_af.af_any_true_by_key(out, keys_out, in_array, key)
    local err = libaf.af_any_true_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_any_true_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_anytrue(out, in_array, dim)
    local err = libaf.af_anytrue(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_anytrue: " .. tostring(err))
    end
    return err
end

function luajit_af.af_approx1(out, in_array, pos, interp, off_grid)
    local err = libaf.af_approx1(out, in_array, pos, interp, off_grid)
    if err ~= 0 then
        error("Error in af_approx1: " .. tostring(err))
    end
    return err
end

function luajit_af.af_approx2(out, in_array, pos0, pos1, interp, off_grid)
    local err = libaf.af_approx2(out, in_array, pos0, pos1, interp, off_grid)
    if err ~= 0 then
        error("Error in af_approx2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_arg(out, arrayIn)
    local err = libaf.af_arg(out, arrayIn)
    if err ~= 0 then
        error("Error in af_arg: " .. tostring(err))
    end
    return err
end

function luajit_af.af_asin(out, arrayIn)
    local err = libaf.af_asin(out, arrayIn)
    if err ~= 0 then
        error("Error in af_asin: " .. tostring(err))
    end
    return err
end

function luajit_af.af_asinh(out, arrayIn)
    local err = libaf.af_asinh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_asinh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_atan(out, arrayIn)
    local err = libaf.af_atan(out, arrayIn)
    if err ~= 0 then
        error("Error in af_atan: " .. tostring(err))
    end
    return err
end

function luajit_af.af_atan2(out, lhs, rhs, dim)
    local err = libaf.af_atan2(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_atan2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_atanh(out, arrayIn)
    local err = libaf.af_atanh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_atanh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bilateral(out, in_array, spatial_sigma, chromatic_sigma)
    local err = libaf.af_bilateral(out, in_array, spatial_sigma, chromatic_sigma)
    if err ~= 0 then
        error("Error in af_bilateral: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitand(out, lhs, rhs, dim)
    local err = libaf.af_bitand(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_bitand: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitnot(out, arrayIn)
    local err = libaf.af_bitnot(out, arrayIn)
    if err ~= 0 then
        error("Error in af_bitnot: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitor(out, lhs, rhs, dim)
    local err = libaf.af_bitor(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_bitor: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitshiftl(out, lhs, rhs)
    local err = libaf.af_bitshiftl(out, lhs, rhs)
    if err ~= 0 then
        error("Error in af_bitshiftl: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitshiftr(out, lhs, rhs)
    local err = libaf.af_bitshiftr(out, lhs, rhs)
    if err ~= 0 then
        error("Error in af_bitshiftr: " .. tostring(err))
    end
    return err
end

function luajit_af.af_bitxor(out, lhs, rhs, dim)
    local err = libaf.af_bitxor(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_bitxor: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cast(out, in_array, type)
    local err = libaf.af_cast(out, in_array, type)
    if err ~= 0 then
        error("Error in af_cast: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cbrt(out, arrayIn)
    local err = libaf.af_cbrt(out, arrayIn)
    if err ~= 0 then
        error("Error in af_cbrt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ceil(out, arrayIn)
    local err = libaf.af_ceil(out, arrayIn)
    if err ~= 0 then
        error("Error in af_ceil: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cholesky(out, info, in_array, is_upper)
    local err = libaf.af_cholesky(out, info, in_array, is_upper)
    if err ~= 0 then
        error("Error in af_cholesky: " .. tostring(err))
    end
    return err
end

function luajit_af.af_clamp(out, in_array, lo, hi)
    local err = libaf.af_clamp(out, in_array, lo, hi)
    if err ~= 0 then
        error("Error in af_clamp: " .. tostring(err))
    end
    return err
end

function luajit_af.af_col(out, in_array, col)
    local err = libaf.af_col(out, in_array, col)
    if err ~= 0 then
        error("Error in af_col: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cols(out, in_array, first, last)
    local err = libaf.af_cols(out, in_array, first, last)
    if err ~= 0 then
        error("Error in af_cols: " .. tostring(err))
    end
    return err
end

function luajit_af.af_complex(out, real, imag)
    local err = libaf.af_complex(out, real, imag)
    if err ~= 0 then
        error("Error in af_complex: " .. tostring(err))
    end
    return err
end

function luajit_af.af_confidence_cc(out, in_array, template, disparity, mean, radius)
    local err = libaf.af_confidence_cc(out, in_array, template, disparity, mean, radius)
    if err ~= 0 then
        error("Error in af_confidence_cc: " .. tostring(err))
    end
    return err
end

function luajit_af.af_conjg(out, arrayIn)
    local err = libaf.af_conjg(out, arrayIn)
    if err ~= 0 then
        error("Error in af_conjg: " .. tostring(err))
    end
    return err
end

function luajit_af.af_constant(out, val, d0, type)
    local err = libaf.af_constant(out, val, d0, type)
    if err ~= 0 then
        error("Error in af_constant: " .. tostring(err))
    end
    return err
end

function luajit_af.af_convolve1(out, signal, filter, mode, domain)
    local err = libaf.af_convolve1(out, signal, filter, mode, domain)
    if err ~= 0 then
        error("Error in af_convolve1: " .. tostring(err))
    end
    return err
end

function luajit_af.af_convolve2(out, signal, filter, mode, domain)
    local err = libaf.af_convolve2(out, signal, filter, mode, domain)
    if err ~= 0 then
        error("Error in af_convolve2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_convolve3(out, signal, filter, mode, domain)
    local err = libaf.af_convolve3(out, signal, filter, mode, domain)
    if err ~= 0 then
        error("Error in af_convolve3: " .. tostring(err))
    end
    return err
end

function luajit_af.af_corrcoef(out, X, Y)
    local err = libaf.af_corrcoef(out, X, Y)
    if err ~= 0 then
        error("Error in af_corrcoef: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cos(out, arrayIn)
    local err = libaf.af_cos(out, arrayIn)
    if err ~= 0 then
        error("Error in af_cos: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cosh(out, arrayIn)
    local err = libaf.af_cosh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_cosh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_count(out, in_array, dim)
    local err = libaf.af_count(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_count: " .. tostring(err))
    end
    return err
end

function luajit_af.af_count_by_key(out, keys_out, in_array, key)
    local err = libaf.af_count_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_count_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_cov(out, X, Y, is_biased)
    local err = libaf.af_cov(out, X, Y, is_biased)
    if err ~= 0 then
        error("Error in af_cov: " .. tostring(err))
    end
    return err
end

function luajit_af.af_delete_image_mem(arrayIn)
    local err = libaf.af_delete_image_mem(arrayIn)
    if err ~= 0 then
        error("Error in af_delete_image_mem: " .. tostring(err))
    end
    return err
end

function luajit_af.af_dense(out, sparse)
    local err = libaf.af_dense(out, sparse)
    if err ~= 0 then
        error("Error in af_dense: " .. tostring(err))
    end
    return err
end

function luajit_af.af_det(out, arrayIn)
    local err = libaf.af_det(out, arrayIn)
    if err ~= 0 then
        error("Error in af_det: " .. tostring(err))
    end
    return err
end

function luajit_af.af_device_info(d_name, d_platform, d_toolkit, d_compute)
    local err = libaf.af_device_info(d_name, d_platform, d_toolkit, d_compute)
    if err ~= 0 then
        error("Error in af_device_info: " .. tostring(err))
    end
    return err
end

function luajit_af.af_device_mem_info(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers)
    local err = libaf.af_device_mem_info(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers)
    if err ~= 0 then
        error("Error in af_device_mem_info: " .. tostring(err))
    end
    return err
end

function luajit_af.af_diag(out, in_array, num)
    local err = libaf.af_diag(out, in_array, num)
    if err ~= 0 then
        error("Error in af_diag: " .. tostring(err))
    end
    return err
end

function luajit_af.af_diff1(out, in_array, dim)
    local err = libaf.af_diff1(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_diff1: " .. tostring(err))
    end
    return err
end

function luajit_af.af_diff2(out, in_array, dim)
    local err = libaf.af_diff2(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_diff2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_dilate(out, in_array, mask)
    local err = libaf.af_dilate(out, in_array, mask)
    if err ~= 0 then
        error("Error in af_dilate: " .. tostring(err))
    end
    return err
end

function luajit_af.af_dilate3d(out, in_array, mask)
    local err = libaf.af_dilate3d(out, in_array, mask)
    if err ~= 0 then
        error("Error in af_dilate3d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_div(out, lhs, rhs, dim)
    local err = libaf.af_div(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_div: " .. tostring(err))
    end
    return err
end

function luajit_af.af_dog(out, in_array, radius1, radius2)
    local err = libaf.af_dog(out, in_array, radius1, radius2)
    if err ~= 0 then
        error("Error in af_dog: " .. tostring(err))
    end
    return err
end

function luajit_af.af_dot(out, lhs, rhs, dim)
    local err = libaf.af_dot(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_dot: " .. tostring(err))
    end
    return err
end

function luajit_af.af_eq(out, lhs, rhs, dim)
    local err = libaf.af_eq(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_eq: " .. tostring(err))
    end
    return err
end

function luajit_af.af_erf(out, arrayIn)
    local err = libaf.af_erf(out, arrayIn)
    if err ~= 0 then
        error("Error in af_erf: " .. tostring(err))
    end
    return err
end

function luajit_af.af_erfc(out, arrayIn)
    local err = libaf.af_erfc(out, arrayIn)
    if err ~= 0 then
        error("Error in af_erfc: " .. tostring(err))
    end
    return err
end

function luajit_af.af_erode(out, in_array, mask)
    local err = libaf.af_erode(out, in_array, mask)
    if err ~= 0 then
        error("Error in af_erode: " .. tostring(err))
    end
    return err
end

function luajit_af.af_erode3d(out, in_array, mask)
    local err = libaf.af_erode3d(out, in_array, mask)
    if err ~= 0 then
        error("Error in af_erode3d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_exp(out, arrayIn)
    local err = libaf.af_exp(out, arrayIn)
    if err ~= 0 then
        error("Error in af_exp: " .. tostring(err))
    end
    return err
end

function luajit_af.af_expm1(out, arrayIn)
    local err = libaf.af_expm1(out, arrayIn)
    if err ~= 0 then
        error("Error in af_expm1: " .. tostring(err))
    end
    return err
end

function luajit_af.af_factorial(out, arrayIn)
    local err = libaf.af_factorial(out, arrayIn)
    if err ~= 0 then
        error("Error in af_factorial: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fast(out, in_array, thr, arc_length, non_max, feature_ratio, edge)
    local err = libaf.af_fast(out, in_array, thr, arc_length, non_max, feature_ratio, edge)
    if err ~= 0 then
        error("Error in af_fast: " .. tostring(err))
    end
    return err
end

function luajit_af.af_features(out, arrayIn)
    local err = libaf.af_features(out, arrayIn)
    if err ~= 0 then
        error("Error in af_features: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fft(out, in_array, norm_factor, odim0)
    local err = libaf.af_fft(out, in_array, norm_factor, odim0)
    if err ~= 0 then
        error("Error in af_fft: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fft2(out, in_array, norm_factor, odim0, odim1)
    local err = libaf.af_fft2(out, in_array, norm_factor, odim0, odim1)
    if err ~= 0 then
        error("Error in af_fft2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fft3(out, in_array, norm_factor, odim0, odim1, odim2)
    local err = libaf.af_fft3(out, in_array, norm_factor, odim0, odim1, odim2)
    if err ~= 0 then
        error("Error in af_fft3: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fft_c2r(out, in_array, norm_factor, is_odd)
    local err = libaf.af_fft_c2r(out, in_array, norm_factor, is_odd)
    if err ~= 0 then
        error("Error in af_fft_c2r: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fft_r2c(out, in_array, norm_factor)
    local err = libaf.af_fft_r2c(out, in_array, norm_factor)
    if err ~= 0 then
        error("Error in af_fft_r2c: " .. tostring(err))
    end
    return err
end

function luajit_af.af_fir(out, b, x)
    local err = libaf.af_fir(out, b, x)
    if err ~= 0 then
        error("Error in af_fir: " .. tostring(err))
    end
    return err
end

function luajit_af.af_flat(out, arrayIn)
    local err = libaf.af_flat(out, arrayIn)
    if err ~= 0 then
        error("Error in af_flat: " .. tostring(err))
    end
    return err
end

function luajit_af.af_flip(out, in_array, dim)
    local err = libaf.af_flip(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_flip: " .. tostring(err))
    end
    return err
end

function luajit_af.af_floor(out, arrayIn)
    local err = libaf.af_floor(out, arrayIn)
    if err ~= 0 then
        error("Error in af_floor: " .. tostring(err))
    end
    return err
end

function luajit_af.af_free_host(ptr)
    local err = libaf.af_free_host(ptr)
    if err ~= 0 then
        error("Error in af_free_host: " .. tostring(err))
    end
    return err
end

function luajit_af.af_free_pinned(ptr)
    local err = libaf.af_free_pinned(ptr)
    if err ~= 0 then
        error("Error in af_free_pinned: " .. tostring(err))
    end
    return err
end

function luajit_af.af_free_v2(ptr)
    local err = libaf.af_free_v2(ptr)
    if err ~= 0 then
        error("Error in af_free_v2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_gaussian_kernel(out, rows, cols, sigma_r, sigma_c)
    local err = libaf.af_gaussian_kernel(out, rows, cols, sigma_r, sigma_c)
    if err ~= 0 then
        error("Error in af_gaussian_kernel: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ge(out, lhs, rhs, dim)
    local err = libaf.af_ge(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_ge: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_active_backend(backend)
    local err = libaf.af_get_active_backend(backend)
    if err ~= 0 then
        error("Error in af_get_active_backend: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_available_backends(backends)
    local err = libaf.af_get_available_backends(backends)
    if err ~= 0 then
        error("Error in af_get_available_backends: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_backend_count(num_backends)
    local err = libaf.af_get_backend_count(num_backends)
    if err ~= 0 then
        error("Error in af_get_backend_count: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_backend_id(backend, arrayIn)
    local err = libaf.af_get_backend_id(backend, arrayIn)
    if err ~= 0 then
        error("Error in af_get_backend_id: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_default_random_engine(engine)
    local err = libaf.af_get_default_random_engine(engine)
    if err ~= 0 then
        error("Error in af_get_default_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_device(device)
    local err = libaf.af_get_device(device)
    if err ~= 0 then
        error("Error in af_get_device: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_device_count(nDevices)
    local err = libaf.af_get_device_count(nDevices)
    if err ~= 0 then
        error("Error in af_get_device_count: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_device_id(device, arrayIn)
    local err = libaf.af_get_device_id(device, arrayIn)
    if err ~= 0 then
        error("Error in af_get_device_id: " .. tostring(err))
    end
    return err
end

function luajit_af.af_get_seed(seed)
    local err = libaf.af_get_seed(seed)
    if err ~= 0 then
        error("Error in af_get_seed: " .. tostring(err))
    end
    return err
end

function luajit_af.af_gloh(out, arrayIn)
    local err = libaf.af_gloh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_gloh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_grad(dx, dy, arrayIn)
    local err = libaf.af_grad(dx, dy, arrayIn)
    if err ~= 0 then
        error("Error in af_grad: " .. tostring(err))
    end
    return err
end

function luajit_af.af_gray2rgb(out, in_array, r, g, b)
    local err = libaf.af_gray2rgb(out, in_array, r, g, b)
    if err ~= 0 then
        error("Error in af_gray2rgb: " .. tostring(err))
    end
    return err
end

function luajit_af.af_gt(out, lhs, rhs, dim)
    local err = libaf.af_gt(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_gt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_hamming_matcher(out, query, train, dist_dim, n_dist)
    local err = libaf.af_hamming_matcher(out, query, train, dist_dim, n_dist)
    if err ~= 0 then
        error("Error in af_hamming_matcher: " .. tostring(err))
    end
    return err
end

function luajit_af.af_harris(out, in_array, max_corners, min_response, sigma, filter_len, k_thr)
    local err = libaf.af_harris(out, in_array, max_corners, min_response, sigma, filter_len, k_thr)
    if err ~= 0 then
        error("Error in af_harris: " .. tostring(err))
    end
    return err
end

function luajit_af.af_histequal(out, in_array, hist)
    local err = libaf.af_histequal(out, in_array, hist)
    if err ~= 0 then
        error("Error in af_histequal: " .. tostring(err))
    end
    return err
end

function luajit_af.af_histogram(out, in_array, nbins, minval, maxval)
    local err = libaf.af_histogram(out, in_array, nbins, minval, maxval)
    if err ~= 0 then
        error("Error in af_histogram: " .. tostring(err))
    end
    return err
end

function luajit_af.af_hsv2rgb(out, arrayIn)
    local err = libaf.af_hsv2rgb(out, arrayIn)
    if err ~= 0 then
        error("Error in af_hsv2rgb: " .. tostring(err))
    end
    return err
end

function luajit_af.af_hypot(out, lhs, rhs)
    local err = libaf.af_hypot(out, lhs, rhs)
    if err ~= 0 then
        error("Error in af_hypot: " .. tostring(err))
    end
    return err
end

function luajit_af.af_identity(out, dim0, type)
    local err = libaf.af_identity(out, dim0, type)
    if err ~= 0 then
        error("Error in af_identity: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ifft(out, in_array, norm_factor, odim0)
    local err = libaf.af_ifft(out, in_array, norm_factor, odim0)
    if err ~= 0 then
        error("Error in af_ifft: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ifft2(out, in_array, norm_factor, odim0, odim1)
    local err = libaf.af_ifft2(out, in_array, norm_factor, odim0, odim1)
    if err ~= 0 then
        error("Error in af_ifft2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ifft3(out, in_array, norm_factor, odim0, odim1, odim2)
    local err = libaf.af_ifft3(out, in_array, norm_factor, odim0, odim1, odim2)
    if err ~= 0 then
        error("Error in af_ifft3: " .. tostring(err))
    end
    return err
end

function luajit_af.af_iir(y, x, b, a)
    local err = libaf.af_iir(y, x, b, a)
    if err ~= 0 then
        error("Error in af_iir: " .. tostring(err))
    end
    return err
end

function luajit_af.af_imag(out, arrayIn)
    local err = libaf.af_imag(out, arrayIn)
    if err ~= 0 then
        error("Error in af_imag: " .. tostring(err))
    end
    return err
end

function luajit_af.af_index(out, in_array, indices, ndims)
    local err = libaf.af_index(out, in_array, indices, ndims)
    if err ~= 0 then
        error("Error in af_index: " .. tostring(err))
    end
    return err
end

function luajit_af.af_info(info_string)
    local err = libaf.af_info(info_string)
    if err ~= 0 then
        error("Error in af_info: " .. tostring(err))
    end
    return err
end

function luajit_af.af_info_string(str, verbose)
    local err = libaf.af_info_string(str, verbose)
    if err ~= 0 then
        error("Error in af_info_string: " .. tostring(err))
    end
    return err
end

function luajit_af.af_iota(out, dim0, dim1, dim2, dim3, type)
    local err = libaf.af_iota(out, dim0, dim1, dim2, dim3, type)
    if err ~= 0 then
        error("Error in af_iota: " .. tostring(err))
    end
    return err
end

function luajit_af.af_is_double_available(out, device)
    local err = libaf.af_is_double_available(out, device)
    if err ~= 0 then
        error("Error in af_is_double_available: " .. tostring(err))
    end
    return err
end

function luajit_af.af_is_half_available(out, device)
    local err = libaf.af_is_half_available(out, device)
    if err ~= 0 then
        error("Error in af_is_half_available: " .. tostring(err))
    end
    return err
end

function luajit_af.af_is_image_io_available(out)
    local err = libaf.af_is_image_io_available(out)
    if err ~= 0 then
        error("Error in af_is_image_io_available: " .. tostring(err))
    end
    return err
end

function luajit_af.af_is_lapack_available(out)
    local err = libaf.af_is_lapack_available(out)
    if err ~= 0 then
        error("Error in af_is_lapack_available: " .. tostring(err))
    end
    return err
end

function luajit_af.af_isinf(out, arrayIn)
    local err = libaf.af_isinf(out, arrayIn)
    if err ~= 0 then
        error("Error in af_isinf: " .. tostring(err))
    end
    return err
end

function luajit_af.af_isnan(out, arrayIn)
    local err = libaf.af_isnan(out, arrayIn)
    if err ~= 0 then
        error("Error in af_isnan: " .. tostring(err))
    end
    return err
end

function luajit_af.af_iszero(out, arrayIn)
    local err = libaf.af_iszero(out, arrayIn)
    if err ~= 0 then
        error("Error in af_iszero: " .. tostring(err))
    end
    return err
end

function luajit_af.af_iterative_deconv(out, in_array, ker, gamma, max_iter)
    local err = libaf.af_iterative_deconv(out, in_array, ker, gamma, max_iter)
    if err ~= 0 then
        error("Error in af_iterative_deconv: " .. tostring(err))
    end
    return err
end

function luajit_af.af_join(out, dim, first, second)
    local err = libaf.af_join(out, dim, first, second)
    if err ~= 0 then
        error("Error in af_join: " .. tostring(err))
    end
    return err
end

function luajit_af.af_le(out, lhs, rhs, dim)
    local err = libaf.af_le(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_le: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lgamma(out, arrayIn)
    local err = libaf.af_lgamma(out, arrayIn)
    if err ~= 0 then
        error("Error in af_lgamma: " .. tostring(err))
    end
    return err
end

function luajit_af.af_load_image(out, filename, is_color)
    local err = libaf.af_load_image(out, filename, is_color)
    if err ~= 0 then
        error("Error in af_load_image: " .. tostring(err))
    end
    return err
end

function luajit_af.af_load_image_memory(out, ptr)
    local err = libaf.af_load_image_memory(out, ptr)
    if err ~= 0 then
        error("Error in af_load_image_memory: " .. tostring(err))
    end
    return err
end

function luajit_af.af_log(out, arrayIn)
    local err = libaf.af_log(out, arrayIn)
    if err ~= 0 then
        error("Error in af_log: " .. tostring(err))
    end
    return err
end

function luajit_af.af_log10(out, arrayIn)
    local err = libaf.af_log10(out, arrayIn)
    if err ~= 0 then
        error("Error in af_log10: " .. tostring(err))
    end
    return err
end

function luajit_af.af_log1p(out, arrayIn)
    local err = libaf.af_log1p(out, arrayIn)
    if err ~= 0 then
        error("Error in af_log1p: " .. tostring(err))
    end
    return err
end

function luajit_af.af_log2(out, arrayIn)
    local err = libaf.af_log2(out, arrayIn)
    if err ~= 0 then
        error("Error in af_log2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lookup(out, in_array, indices, dim)
    local err = libaf.af_lookup(out, in_array, indices, dim)
    if err ~= 0 then
        error("Error in af_lookup: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lower(out, in_array, is_unit_diag)
    local err = libaf.af_lower(out, in_array, is_unit_diag)
    if err ~= 0 then
        error("Error in af_lower: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lt(out, lhs, rhs, dim)
    local err = libaf.af_lt(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_lt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_lu(lower, upper, pivot, arrayIn)
    local err = libaf.af_lu(lower, upper, pivot, arrayIn)
    if err ~= 0 then
        error("Error in af_lu: " .. tostring(err))
    end
    return err
end

function luajit_af.af_max(out, in_array, dim)
    local err = libaf.af_max(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_max: " .. tostring(err))
    end
    return err
end

function luajit_af.af_max_all(real, imag, arrayIn)
    local err = libaf.af_max_all(real, imag, arrayIn)
    if err ~= 0 then
        error("Error in af_max_all: " .. tostring(err))
    end
    return err
end

function luajit_af.af_max_by_key(out, keys_out, in_array, key)
    local err = libaf.af_max_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_max_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_maxfilt(out, in_array, wx, wy, sx, sy, px, py, is_border)
    local err = libaf.af_maxfilt(out, in_array, wx, wy, sx, sy, px, py, is_border)
    if err ~= 0 then
        error("Error in af_maxfilt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_mean(out, in_array, dim)
    local err = libaf.af_mean(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_mean: " .. tostring(err))
    end
    return err
end

function luajit_af.af_meanshift(out, in_array, spatial_sigma, chromatic_sigma, iter, is_color)
    local err = libaf.af_meanshift(out, in_array, spatial_sigma, chromatic_sigma, iter, is_color)
    if err ~= 0 then
        error("Error in af_meanshift: " .. tostring(err))
    end
    return err
end

function luajit_af.af_median(out, in_array, dim)
    local err = libaf.af_median(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_median: " .. tostring(err))
    end
    return err
end

function luajit_af.af_min(out, in_array, dim)
    local err = libaf.af_min(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_min: " .. tostring(err))
    end
    return err
end

function luajit_af.af_min_all(real, imag, arrayIn)
    local err = libaf.af_min_all(real, imag, arrayIn)
    if err ~= 0 then
        error("Error in af_min_all: " .. tostring(err))
    end
    return err
end

function luajit_af.af_min_by_key(out, keys_out, in_array, key)
    local err = libaf.af_min_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_min_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_minfilt(out, in_array, wx, wy, sx, sy, px, py, is_border)
    local err = libaf.af_minfilt(out, in_array, wx, wy, sx, sy, px, py, is_border)
    if err ~= 0 then
        error("Error in af_minfilt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_mod(out, lhs, rhs, dim)
    local err = libaf.af_mod(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_mod: " .. tostring(err))
    end
    return err
end

function luajit_af.af_moddims(out, in_array, ndims, dims)
    local err = libaf.af_moddims(out, in_array, ndims, dims)
    if err ~= 0 then
        error("Error in af_moddims: " .. tostring(err))
    end
    return err
end

function luajit_af.af_mul(out, lhs, rhs, dim)
    local err = libaf.af_mul(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_mul: " .. tostring(err))
    end
    return err
end

function luajit_af.af_neg(out, arrayIn)
    local err = libaf.af_neg(out, arrayIn)
    if err ~= 0 then
        error("Error in af_neg: " .. tostring(err))
    end
    return err
end

function luajit_af.af_neq(out, lhs, rhs, dim)
    local err = libaf.af_neq(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_neq: " .. tostring(err))
    end
    return err
end

function luajit_af.af_not(out, arrayIn)
    local err = libaf.af_not(out, arrayIn)
    if err ~= 0 then
        error("Error in af_not: " .. tostring(err))
    end
    return err
end

function luajit_af.af_or(out, lhs, rhs, dim)
    local err = libaf.af_or(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_or: " .. tostring(err))
    end
    return err
end

function luajit_af.af_orb(out, desc, in_array, mask, fast_thr, max_feat, scl_fctr, levels)
    local err = libaf.af_orb(out, desc, in_array, mask, fast_thr, max_feat, scl_fctr, levels)
    if err ~= 0 then
        error("Error in af_orb: " .. tostring(err))
    end
    return err
end

function luajit_af.af_pinned(out, dim0, dim1, dim2, dim3, type)
    local err = libaf.af_pinned(out, dim0, dim1, dim2, dim3, type)
    if err ~= 0 then
        error("Error in af_pinned: " .. tostring(err))
    end
    return err
end

function luajit_af.af_pinverse(out, in_array, tol)
    local err = libaf.af_pinverse(out, in_array, tol)
    if err ~= 0 then
        error("Error in af_pinverse: " .. tostring(err))
    end
    return err
end

function luajit_af.af_pow(out, lhs, rhs, dim)
    local err = libaf.af_pow(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_pow: " .. tostring(err))
    end
    return err
end

function luajit_af.af_pow2(out, arrayIn)
    local err = libaf.af_pow2(out, arrayIn)
    if err ~= 0 then
        error("Error in af_pow2: " .. tostring(err))
    end
    return err
end

function luajit_af.af_print_array(arr)
    local err = libaf.af_print_array(arr)
    if err ~= 0 then
        error("Error in af_print_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_product(out, in_array, dim)
    local err = libaf.af_product(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_product: " .. tostring(err))
    end
    return err
end

function luajit_af.af_product_by_key(out, keys_out, in_array, key)
    local err = libaf.af_product_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_product_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_qr(q, r, tau, arrayIn)
    local err = libaf.af_qr(q, r, tau, arrayIn)
    if err ~= 0 then
        error("Error in af_qr: " .. tostring(err))
    end
    return err
end

function luajit_af.af_randn(out, d0, type)
    local err = libaf.af_randn(out, d0, type)
    if err ~= 0 then
        error("Error in af_randn: " .. tostring(err))
    end
    return err
end

function luajit_af.af_random_engine(out, type, seed)
    local err = libaf.af_random_engine(out, type, seed)
    if err ~= 0 then
        error("Error in af_random_engine: " .. tostring(err))
    end
    return err
end

function luajit_af.af_randu(out, d0, type)
    local err = libaf.af_randu(out, d0, type)
    if err ~= 0 then
        error("Error in af_randu: " .. tostring(err))
    end
    return err
end

function luajit_af.af_range(out, d0, seq_dim, begin, step, type)
    local err = libaf.af_range(out, d0, seq_dim, begin, step, type)
    if err ~= 0 then
        error("Error in af_range: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rank(rank, in_array, tol)
    local err = libaf.af_rank(rank, in_array, tol)
    if err ~= 0 then
        error("Error in af_rank: " .. tostring(err))
    end
    return err
end

function luajit_af.af_read_array(out, filename, is_column)
    local err = libaf.af_read_array(out, filename, is_column)
    if err ~= 0 then
        error("Error in af_read_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_real(out, arrayIn)
    local err = libaf.af_real(out, arrayIn)
    if err ~= 0 then
        error("Error in af_real: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rem(out, lhs, rhs, dim)
    local err = libaf.af_rem(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_rem: " .. tostring(err))
    end
    return err
end

function luajit_af.af_reorder(out, in_array, x, y, z, w)
    local err = libaf.af_reorder(out, in_array, x, y, z, w)
    if err ~= 0 then
        error("Error in af_reorder: " .. tostring(err))
    end
    return err
end

function luajit_af.af_replace(a, cond, b)
    local err = libaf.af_replace(a, cond, b)
    if err ~= 0 then
        error("Error in af_replace: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rgb2gray(out, in_array, rWeight, gWeight, bWeight)
    local err = libaf.af_rgb2gray(out, in_array, rWeight, gWeight, bWeight)
    if err ~= 0 then
        error("Error in af_rgb2gray: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rgb2hsv(out, arrayIn)
    local err = libaf.af_rgb2hsv(out, arrayIn)
    if err ~= 0 then
        error("Error in af_rgb2hsv: " .. tostring(err))
    end
    return err
end

function luajit_af.af_root(out, in_array, n)
    local err = libaf.af_root(out, in_array, n)
    if err ~= 0 then
        error("Error in af_root: " .. tostring(err))
    end
    return err
end

function luajit_af.af_round(out, arrayIn)
    local err = libaf.af_round(out, arrayIn)
    if err ~= 0 then
        error("Error in af_round: " .. tostring(err))
    end
    return err
end

function luajit_af.af_row(out, in_array, row)
    local err = libaf.af_row(out, in_array, row)
    if err ~= 0 then
        error("Error in af_row: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rows(out, in_array, first, last)
    local err = libaf.af_rows(out, in_array, first, last)
    if err ~= 0 then
        error("Error in af_rows: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rsqrt(out, arrayIn)
    local err = libaf.af_rsqrt(out, arrayIn)
    if err ~= 0 then
        error("Error in af_rsqrt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sat(out, arrayIn)
    local err = libaf.af_sat(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sat: " .. tostring(err))
    end
    return err
end

function luajit_af.af_save_array(filename, arrayIn)
    local err = libaf.af_save_array(filename, arrayIn)
    if err ~= 0 then
        error("Error in af_save_array: " .. tostring(err))
    end
    return err
end

function luajit_af.af_save_image(filename, arrayIn)
    local err = libaf.af_save_image(filename, arrayIn)
    if err ~= 0 then
        error("Error in af_save_image: " .. tostring(err))
    end
    return err
end

function luajit_af.af_scan(out, in_array, dim, op, inclusive_scan)
    local err = libaf.af_scan(out, in_array, dim, op, inclusive_scan)
    if err ~= 0 then
        error("Error in af_scan: " .. tostring(err))
    end
    return err
end

function luajit_af.af_scan_by_key(out, keys, in_array, dim, op, inclusive_scan)
    local err = libaf.af_scan_by_key(out, keys, in_array, dim, op, inclusive_scan)
    if err ~= 0 then
        error("Error in af_scan_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_select(out, cond, a, b)
    local err = libaf.af_select(out, cond, a, b)
    if err ~= 0 then
        error("Error in af_select: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_backend(backend)
    local err = libaf.af_set_backend(backend)
    if err ~= 0 then
        error("Error in af_set_backend: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_default_random_engine_type(type)
    local err = libaf.af_set_default_random_engine_type(type)
    if err ~= 0 then
        error("Error in af_set_default_random_engine_type: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_device(device)
    local err = libaf.af_set_device(device)
    if err ~= 0 then
        error("Error in af_set_device: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_seed(seed)
    local err = libaf.af_set_seed(seed)
    if err ~= 0 then
        error("Error in af_set_seed: " .. tostring(err))
    end
    return err
end

function luajit_af.af_setintersect(out, a, b, is_unique)
    local err = libaf.af_setintersect(out, a, b, is_unique)
    if err ~= 0 then
        error("Error in af_setintersect: " .. tostring(err))
    end
    return err
end

function luajit_af.af_setunion(out, a, b, is_unique)
    local err = libaf.af_setunion(out, a, b, is_unique)
    if err ~= 0 then
        error("Error in af_setunion: " .. tostring(err))
    end
    return err
end

function luajit_af.af_setunique(out, in_array, is_sorted)
    local err = libaf.af_setunique(out, in_array, is_sorted)
    if err ~= 0 then
        error("Error in af_setunique: " .. tostring(err))
    end
    return err
end

function luajit_af.af_shift(out, in_array, x, y, z, w)
    local err = libaf.af_shift(out, in_array, x, y, z, w)
    if err ~= 0 then
        error("Error in af_shift: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sift(out, in_array, n_layers, contrast_thr, edge_thr, init_sigma)
    local err = libaf.af_sift(out, in_array, n_layers, contrast_thr, edge_thr, init_sigma)
    if err ~= 0 then
        error("Error in af_sift: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sigmoid(out, arrayIn)
    local err = libaf.af_sigmoid(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sigmoid: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sign(out, arrayIn)
    local err = libaf.af_sign(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sign: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sin(out, arrayIn)
    local err = libaf.af_sin(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sin: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sinh(out, arrayIn)
    local err = libaf.af_sinh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sinh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_skew(out, in_array, dim, bias)
    local err = libaf.af_skew(out, in_array, dim, bias)
    if err ~= 0 then
        error("Error in af_skew: " .. tostring(err))
    end
    return err
end

function luajit_af.af_slice(out, in_array, idx)
    local err = libaf.af_slice(out, in_array, idx)
    if err ~= 0 then
        error("Error in af_slice: " .. tostring(err))
    end
    return err
end

function luajit_af.af_slices(out, in_array, first, last)
    local err = libaf.af_slices(out, in_array, first, last)
    if err ~= 0 then
        error("Error in af_slices: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sobel(dx, dy, in_array, w_size)
    local err = libaf.af_sobel(dx, dy, in_array, w_size)
    if err ~= 0 then
        error("Error in af_sobel: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sort(out, in_array, dim, ascending)
    local err = libaf.af_sort(out, in_array, dim, ascending)
    if err ~= 0 then
        error("Error in af_sort: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sort_by_key(out_keys, out_values, keys, values, dim, ascending)
    local err = libaf.af_sort_by_key(out_keys, out_values, keys, values, dim, ascending)
    if err ~= 0 then
        error("Error in af_sort_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sort_index(out_indices, in_array, dim, ascending)
    local err = libaf.af_sort_index(out_indices, in_array, dim, ascending)
    if err ~= 0 then
        error("Error in af_sort_index: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_coldx(out, arrayIn)
    local err = libaf.af_sparse_get_coldx(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_coldx: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_nnz(out, arrayIn)
    local err = libaf.af_sparse_get_nnz(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_nnz: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_row_idx(out, arrayIn)
    local err = libaf.af_sparse_get_row_idx(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_row_idx: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_values(out, arrayIn)
    local err = libaf.af_sparse_get_values(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_values: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sqrt(out, arrayIn)
    local err = libaf.af_sqrt(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sqrt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_stdev(out, in_array, dim)
    local err = libaf.af_stdev(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_stdev: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sub(out, lhs, rhs, dim)
    local err = libaf.af_sub(out, lhs, rhs, dim)
    if err ~= 0 then
        error("Error in af_sub: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sum(out, in_array, dim)
    local err = libaf.af_sum(out, in_array, dim)
    if err ~= 0 then
        error("Error in af_sum: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sum_by_key(out, keys_out, in_array, key)
    local err = libaf.af_sum_by_key(out, keys_out, in_array, key)
    if err ~= 0 then
        error("Error in af_sum_by_key: " .. tostring(err))
    end
    return err
end

function luajit_af.af_susan(out, in_array, radius, diff_thr, geom_thr, feature_ratio, edge)
    local err = libaf.af_susan(out, in_array, radius, diff_thr, geom_thr, feature_ratio, edge)
    if err ~= 0 then
        error("Error in af_susan: " .. tostring(err))
    end
    return err
end

function luajit_af.af_svd(u, s, vt, arrayIn)
    local err = libaf.af_svd(u, s, vt, arrayIn)
    if err ~= 0 then
        error("Error in af_svd: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sync(device)
    local err = libaf.af_sync(device)
    if err ~= 0 then
        error("Error in af_sync: " .. tostring(err))
    end
    return err
end

function luajit_af.af_tan(out, arrayIn)
    local err = libaf.af_tan(out, arrayIn)
    if err ~= 0 then
        error("Error in af_tan: " .. tostring(err))
    end
    return err
end

function luajit_af.af_tanh(out, arrayIn)
    local err = libaf.af_tanh(out, arrayIn)
    if err ~= 0 then
        error("Error in af_tanh: " .. tostring(err))
    end
    return err
end

function luajit_af.af_tgamma(out, arrayIn)
    local err = libaf.af_tgamma(out, arrayIn)
    if err ~= 0 then
        error("Error in af_tgamma: " .. tostring(err))
    end
    return err
end

function luajit_af.af_tile(out, in_array, x, y, z, w)
    local err = libaf.af_tile(out, in_array, x, y, z, w)
    if err ~= 0 then
        error("Error in af_tile: " .. tostring(err))
    end
    return err
end

function luajit_af.af_to_string(output, exp, arr)
    local err = libaf.af_to_string(output, exp, arr)
    if err ~= 0 then
        error("Error in af_to_string: " .. tostring(err))
    end
    return err
end

function luajit_af.af_transform_coordinates(out, tf, d0, d1)
    local err = libaf.af_transform_coordinates(out, tf, d0, d1)
    if err ~= 0 then
        error("Error in af_transform_coordinates: " .. tostring(err))
    end
    return err
end

function luajit_af.af_transpose(out, in_array, conjugate)
    local err = libaf.af_transpose(out, in_array, conjugate)
    if err ~= 0 then
        error("Error in af_transpose: " .. tostring(err))
    end
    return err
end

function luajit_af.af_trunc(out, arrayIn)
    local err = libaf.af_trunc(out, arrayIn)
    if err ~= 0 then
        error("Error in af_trunc: " .. tostring(err))
    end
    return err
end

function luajit_af.af_unwrap(out, in_array, wx, wy, sx, sy, px, py, is_column)
    local err = libaf.af_unwrap(out, in_array, wx, wy, sx, sy, px, py, is_column)
    if err ~= 0 then
        error("Error in af_unwrap: " .. tostring(err))
    end
    return err
end

function luajit_af.af_upper(out, in_array, is_unit_diag)
    local err = libaf.af_upper(out, in_array, is_unit_diag)
    if err ~= 0 then
        error("Error in af_upper: " .. tostring(err))
    end
    return err
end

function luajit_af.af_var(out, in_array, is_biased, dim)
    local err = libaf.af_var(out, in_array, is_biased, dim)
    if err ~= 0 then
        error("Error in af_var: " .. tostring(err))
    end
    return err
end

function luajit_af.af_where(idx, arrayIn)
    local err = libaf.af_where(idx, arrayIn)
    if err ~= 0 then
        error("Error in af_where: " .. tostring(err))
    end
    return err
end

function luajit_af.af_wrap(out, in_array, ox, oy, wx, wy, sx, sy, px, py, is_column)
    local err = libaf.af_wrap(out, in_array, ox, oy, wx, wy, sx, sy, px, py, is_column)
    if err ~= 0 then
        error("Error in af_wrap: " .. tostring(err))
    end
    return err
end

function luajit_af.af_canny(out, in_array, thresholdType, low, high, isfGaussian, sigma)
    local err = libaf.af_canny(out, in_array, thresholdType, low, high, isfGaussian, sigma)
    if err ~= 0 then
        error("Error in af_canny: " .. tostring(err))
    end
    return err
end

function luajit_af.af_color_space(out, in_array, to, from)
    local err = libaf.af_color_space(out, in_array, to, from)
    if err ~= 0 then
        error("Error in af_color_space: " .. tostring(err))
    end
    return err
end

function luajit_af.af_homography(out, inliers, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations)
    local err = libaf.af_homography(out, inliers, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations)
    if err ~= 0 then
        error("Error in af_homography: " .. tostring(err))
    end
    return err
end

function luajit_af.af_inverse(out, in_array, options)
    local err = libaf.af_inverse(out, in_array, options)
    if err ~= 0 then
        error("Error in af_inverse: " .. tostring(err))
    end
    return err
end

function luajit_af.af_inverse_deconv(out, in_array, psf, gamma, algo)
    local err = libaf.af_inverse_deconv(out, in_array, psf, gamma, algo)
    if err ~= 0 then
        error("Error in af_inverse_deconv: " .. tostring(err))
    end
    return err
end

function luajit_af.af_match_template(out, search_img, template_img, mtype)
    local err = libaf.af_match_template(out, search_img, template_img, mtype)
    if err ~= 0 then
        error("Error in af_match_template: " .. tostring(err))
    end
    return err
end

function luajit_af.af_matmul(out, lhs, rhs, opt_lhs, opt_rhs)
    local err = libaf.af_matmul(out, lhs, rhs, opt_lhs, opt_rhs)
    if err ~= 0 then
        error("Error in af_matmul: " .. tostring(err))
    end
    return err
end

function luajit_af.af_medfilt(out, in_array, wx, wy, edge_pad)
    local err = libaf.af_medfilt(out, in_array, wx, wy, edge_pad)
    if err ~= 0 then
        error("Error in af_medfilt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_moments(out, in_array, moment)
    local err = libaf.af_moments(out, in_array, moment)
    if err ~= 0 then
        error("Error in af_moments: " .. tostring(err))
    end
    return err
end

function luajit_af.af_nearest_neighbour(idx, query, train, dist_dim, k, dist_type)
    local err = libaf.af_nearest_neighbour(idx, query, train, dist_dim, k, dist_type)
    if err ~= 0 then
        error("Error in af_nearest_neighbour: " .. tostring(err))
    end
    return err
end

function luajit_af.af_norm(out, in_array, type, p, q, dim)
    local err = libaf.af_norm(out, in_array, type, p, q, dim)
    if err ~= 0 then
        error("Error in af_norm: " .. tostring(err))
    end
    return err
end

function luajit_af.af_pad(out, in_array, begin_padding, end_padding, border_type)
    local err = libaf.af_pad(out, in_array, begin_padding, end_padding, border_type)
    if err ~= 0 then
        error("Error in af_pad: " .. tostring(err))
    end
    return err
end

function luajit_af.af_regions(out, in_array, connectivity, type)
    local err = libaf.af_regions(out, in_array, connectivity, type)
    if err ~= 0 then
        error("Error in af_regions: " .. tostring(err))
    end
    return err
end

function luajit_af.af_medfilt(out, in_array, trans0, trans1, odim0, odim1, method)
    local err = libaf.af_medfilt(out, in_array, trans0, trans1, odim0, odim1, method)
    if err ~= 0 then
        error("Error in af_medfilt: " .. tostring(err))
    end
    return err
end

function luajit_af.af_transform(out, in_array, tf, odim0, odim1, method, inverse)
    local err = libaf.af_transform(out, in_array, tf, odim0, odim1, method, inverse)
    if err ~= 0 then
        error("Error in af_transform: " .. tostring(err))
    end
    return err
end

function luajit_af.af_resize(out, in_array, od0, od1, method)
    local err = libaf.af_resize(out, in_array, od0, od1, method)
    if err ~= 0 then
        error("Error in af_resize: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rotate(out, in_array, theta, crop, method)
    local err = libaf.af_rotate(out, in_array, theta, crop, method)
    if err ~= 0 then
        error("Error in af_rotate: " .. tostring(err))
    end
    return err
end

function luajit_af.af_scale(out, in_array, scale0, scale1, od0, od1, method)
    local err = libaf.af_scale(out, in_array, scale0, scale1, od0, od1, method)
    if err ~= 0 then
        error("Error in af_scale: " .. tostring(err))
    end
    return err
end

function luajit_af.af_ycbcr2rgb(out, in_array, standard)
    local err = libaf.af_ycbcr2rgb(out, in_array, standard)
    if err ~= 0 then
        error("Error in af_ycbcr2rgb: " .. tostring(err))
    end
    return err
end

function luajit_af.af_rgb2ycbcr(out, in_array, standard)
    local err = libaf.af_rgb2ycbcr(out, in_array, standard)
    if err ~= 0 then
        error("Error in af_rgb2ycbcr: " .. tostring(err))
    end
    return err
end

function luajit_af.af_save_image_memory(out, in_array, format)
    local err = libaf.af_save_image_memory(out, in_array, format)
    if err ~= 0 then
        error("Error in af_save_image_memory: " .. tostring(err))
    end
    return err
end

function luajit_af.af_solve(out, a, b, options)
    local err = libaf.af_solve(out, a, b, options)
    if err ~= 0 then
        error("Error in af_solve: " .. tostring(err))
    end
    return err
end

function luajit_af.af_solve_lu(out, a, piv, b, options)
    local err = libaf.af_solve_lu(out, a, piv, b, options)
    if err ~= 0 then
        error("Error in af_solve_lu: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_storage(out, arrayIn)
    local err = libaf.af_sparse_get_storage(out, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_storage: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_get_info(out_values, row_idx, col_idx, dims, stype, arrayIn)
    local err = libaf.af_sparse_get_info(out_values, row_idx, col_idx, dims, stype, arrayIn)
    if err ~= 0 then
        error("Error in af_sparse_get_info: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse_convert_to(out, in_array, stype)
    local err = libaf.af_sparse_convert_to(out, in_array, stype)
    if err ~= 0 then
        error("Error in af_sparse_convert_to: " .. tostring(err))
    end
    return err
end

function luajit_af.af_sparse(out, values, row_idx, col_idx, dims, stype)
    local err = libaf.af_sparse(out, values, row_idx, col_idx, dims, stype)
    if err ~= 0 then
        error("Error in af_sparse: " .. tostring(err))
    end
    return err
end

function luajit_af.af_topk(values, indices, in_array, k, dim, func)
    local err = libaf.af_topk(values, indices, in_array, k, dim, func)
    if err ~= 0 then
        error("Error in af_topk: " .. tostring(err))
    end
    return err
end

function luajit_af.af_create_window(out, width, height, title)
    local err = libaf.af_create_window(out, width, height, title)
    if err ~= 0 then
        error("Error in af_create_window: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_position(wind, x, y)
    local err = libaf.af_set_position(wind, x, y)
    if err ~= 0 then
        error("Error in af_set_position: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_title(wind, title)
    local err = libaf.af_set_title(wind, title)
    if err ~= 0 then
        error("Error in af_set_title: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_size(wind, w, h)
    local err = libaf.af_set_size(wind, w, h)
    if err ~= 0 then
        error("Error in af_set_size: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_image(wind, in_array, props)
    local err = libaf.af_draw_image(wind, in_array, props)
    if err ~= 0 then
        error("Error in af_draw_image: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter(wind, X, Y, marker, props)
    local err = libaf.af_draw_scatter(wind, X, Y, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_hist(wind, X, minval, maxval, props)
    local err = libaf.af_draw_hist(wind, X, minval, maxval, props)
    if err ~= 0 then
        error("Error in af_draw_hist: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_surface(wind, xVals, yVals, S, props)
    local err = libaf.af_draw_surface(wind, xVals, yVals, S, props)
    if err ~= 0 then
        error("Error in af_draw_surface: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_vector_field_2d(wind, xPoints, yPoints, xDirs, yDirs, props)
    local err = libaf.af_draw_vector_field_2d(wind, xPoints, yPoints, xDirs, yDirs, props)
    if err ~= 0 then
        error("Error in af_draw_vector_field_2d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_grid(wind, rows, cols)
    local err = libaf.af_grid(wind, rows, cols)
    if err ~= 0 then
        error("Error in af_grid: " .. tostring(err))
    end
    return err
end

function luajit_af.af_show(wind)
    local err = libaf.af_show(wind)
    if err ~= 0 then
        error("Error in af_show: " .. tostring(err))
    end
    return err
end

function luajit_af.af_is_window_closed(out, wind)
    local err = libaf.af_is_window_closed(out, wind)
    if err ~= 0 then
        error("Error in af_is_window_closed: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_visibility(wind, is_visible)
    local err = libaf.af_set_visibility(wind, is_visible)
    if err ~= 0 then
        error("Error in af_set_visibility: " .. tostring(err))
    end
    return err
end

function luajit_af.af_destroy_window(wind)
    local err = libaf.af_destroy_window(wind)
    if err ~= 0 then
        error("Error in af_destroy_window: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_axes_limits_2d(wind, xmin, xmax, ymin, ymax, exact, props)
    local err = libaf.af_set_axes_limits_2d(wind, xmin, xmax, ymin, ymax, exact, props)
    if err ~= 0 then
        error("Error in af_set_axes_limits_2d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_axes_titles(wind, xtitle, ytitle, ztitle, props)
    local err = libaf.af_set_axes_titles(wind, xtitle, ytitle, ztitle, props)
    if err ~= 0 then
        error("Error in af_set_axes_titles: " .. tostring(err))
    end
    return err
end

function luajit_af.af_set_axes_label_format(wind, xformat, yformat, zformat, props)
    local err = libaf.af_set_axes_label_format(wind, xformat, yformat, zformat, props)
    if err ~= 0 then
        error("Error in af_set_axes_label_format: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_image(wind, in_array, props)
    local err = libaf.af_draw_image(wind, in_array, props)
    if err ~= 0 then
        error("Error in af_draw_image: " .. tostring(err))
    end
    return err
end
function luajit_af.af_draw_plot(wind, X, Y, props)
    local err = libaf.af_draw_plot(wind, X, Y, props)
    if err ~= 0 then
        error("Error in af_draw_plot: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_plot3(wind, P, props)
    local err = libaf.af_draw_plot3(wind, P, props)
    if err ~= 0 then
        error("Error in af_draw_plot3: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_plot_nd(wind, P, props)
    local err = libaf.af_draw_plot_nd(wind, P, props)
    if err ~= 0 then
        error("Error in af_draw_plot_nd: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_plot_2d(wind, X, Y, props)
    local err = libaf.af_draw_plot_2d(wind, X, Y, props)
    if err ~= 0 then
        error("Error in af_draw_plot_2d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_plot_3d(wind, X, Y, Z, props)
    local err = libaf.af_draw_plot_3d(wind, X, Y, Z, props)
    if err ~= 0 then
        error("Error in af_draw_plot_3d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter(wind, X, Y, marker, props)
    local err = libaf.af_draw_scatter(wind, X, Y, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter3(wind, P, marker, props)
    local err = libaf.af_draw_scatter3(wind, P, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter3: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter_nd(wind, P, marker, props)
    local err = libaf.af_draw_scatter_nd(wind, P, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter_nd: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter_2d(wind, X, Y, marker, props)
    local err = libaf.af_draw_scatter_2d(wind, X, Y, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter_2d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_scatter_3d(wind, X, Y, Z, marker, props)
    local err = libaf.af_draw_scatter_3d(wind, X, Y, Z, marker, props)
    if err ~= 0 then
        error("Error in af_draw_scatter_3d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_hist(wind, X, minval, maxval, props)
    local err = libaf.af_draw_hist(wind, X, minval, maxval, props)
    if err ~= 0 then
        error("Error in af_draw_hist: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_surface(wind, xVals, yVals, S, props)
    local err = libaf.af_draw_surface(wind, xVals, yVals, S, props)
    if err ~= 0 then
        error("Error in af_draw_surface: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_vector_field_nd(wind, points, directions, props)
    local err = libaf.af_draw_vector_field_nd(wind, points, directions, props)
    if err ~= 0 then
        error("Error in af_draw_vector_field_nd: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_vector_field_3d(wind, xPoints, yPoints, zPoints, xDirs, yDirs, zDirs, props)
    local err = libaf.af_draw_vector_field_3d(wind, xPoints, yPoints, zPoints, xDirs, yDirs, zDirs, props)
    if err ~= 0 then
        error("Error in af_draw_vector_field_3d: " .. tostring(err))
    end
    return err
end

function luajit_af.af_draw_vector_field_2d(wind, xPoints, yPoints, xDirs, yDirs, props)
    local err = libaf.af_draw_vector_field_2d(wind, xPoints, yPoints, xDirs, yDirs, props)
    if err ~= 0 then
        error("Error in af_draw_vector_field_2d: " .. tostring(err))
    end
    return err
end



return luajit_af
