--------------------------------------------------------------------------------
-- luajit_af.lua (Refactored for ArrayFire 3.9.0)
--------------------------------------------------------------------------------

local ffi = require("ffi")

--------------------------------------------------------------------------------
-- 1) Load an appropriate ArrayFire DLL on Windows (or shared library on Linux)
--------------------------------------------------------------------------------
local possibleLibs = {
    "afcuda",  -- if you have CUDA installed
    "afcpu",   -- CPU fallback
    "afopencl",
    "af",      -- final fallback if it's named just "af"
}

local libaf
local found = false
for _, libname in ipairs(possibleLibs) do
    local ok, loaded = pcall(function() return ffi.load(libname) end)
    if ok and loaded then
        libaf = loaded
        found = true
        print("luajit_af: Successfully loaded ArrayFire library: " .. libname)
        break
    end
end

if not found then
    error("luajit_af: Could not find a suitable ArrayFire DLL among: " ..
          table.concat(possibleLibs, ", "))
end

--------------------------------------------------------------------------------
-- 2) Require the cdefs for all the AF API
--------------------------------------------------------------------------------
local defs = require("luajit_af_common_defs")

--------------------------------------------------------------------------------
-- 3) Build the luajit_af wrapper table
--------------------------------------------------------------------------------
local luajit_af = {}

----------------------------------------
-- 3a) Cached FFI types
----------------------------------------
local cached_types = {
    dim_t_arr4   = ffi.typeof("dim_t[4]"),
    af_array_ptr = ffi.typeof("af_array[1]")
}

----------------------------------------
-- 4) Error Handling Helper
----------------------------------------
local function check_err(errval, funcname)
    local e = tonumber(errval) or -9999  -- cast cdata enum to number
    if e ~= 0 then
        -- Convert error code to string
        local err_str = libaf.af_err_to_string(e)
        error(string.format("Error in %s: %d (%s)", funcname, e, err_str))
    end
end

----------------------------------------
-- 5) Automatic Cleanup Metatable
----------------------------------------
local array_mt = {
    __gc = function(self)
        if self.handle then
            -- Release the underlying ArrayFire handle
            luajit_af.af_release_array(self.handle)
        end
    end
}

--- Create a Lua table that manages an `af_array` handle automatically.
-- When the table is GC-ed, the underlying array will be released.
-- @param data The source data (Lua table, pointer, or other FFI-acceptable type)
-- @param ndims Number of dimensions (1-4)
-- @param dims Table or FFI array of dimension sizes
-- @param dtype ArrayFire data type (e.g., f32, f64, etc.)
-- @return Lua table with `.handle` referencing the allocated `af_array`.
function luajit_af.create_managed_array(data, ndims, dims, dtype)
    local arr = setmetatable({}, array_mt)
    arr.handle = luajit_af.af_create_array(data, ndims, dims, dtype)
    return arr
end

----------------------------------------
--    Creating & Releasing Arrays
----------------------------------------

--- Creates an ArrayFire array from data
-- @param data The source data (must be compatible with ArrayFire types)
-- @param ndims Number of dimensions (1-4)
-- @param dims Table or ffi array of dimension sizes
-- @param dtype ArrayFire data type (e.g., f32, f64)
-- @return af_array handle and error code
function luajit_af.af_create_array(data, ndims, dims, dtype)
    -- Parameter checks: convert Lua table -> cdata if needed
    if type(dims) == "table" then
        -- We'll allocate a temporary cdata array of size 4, then fill it
        local dimsC = cached_types.dim_t_arr4()
        for i = 1, ndims do
            dimsC[i - 1] = dims[i]
        end
        dims = dimsC
    else
        -- Otherwise, ensure dims is already an FFI cdata array of type dim_t[?]
        assert(ffi.istype("dim_t[?]", dims), "dims must be a table or dim_t array")
    end

    local arr = cached_types.af_array_ptr()
    local err = libaf.af_create_array(arr, data, ndims, dims, dtype)
    check_err(err, "af_create_array")
    return arr[0], tonumber(err)
end

function luajit_af.af_copy_array(arrayIn)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_copy_array(arr, arrayIn)
    check_err(err, "af_copy_array")
    return arr[0], tonumber(err)
end

function luajit_af.af_write_array(arr, data, bytes, src)
    local err = libaf.af_write_array(arr, data, bytes, src)
    check_err(err, "af_write_array")
    return tonumber(err)
end

function luajit_af.af_get_data_ptr(arr)
    -- We'll fetch the number of elements:
    local infoDim = ffi.new("dim_t[1]")
    local err_elems = libaf.af_get_elements(infoDim, arr)
    check_err(err_elems, "af_get_elements")

    local numElems = tonumber(infoDim[0]) or 0
    if numElems <= 0 then
        error("Cannot get data pointer for an empty or invalid array.")
    end

    -- We assume a float array here; adjust if you know the actual dtype.
    local outBuffer = ffi.new("float[?]", numElems)
    local err = libaf.af_get_data_ptr(outBuffer, arr)
    check_err(err, "af_get_data_ptr")

    return outBuffer, numElems, tonumber(err)
end

function luajit_af.af_release_array(arr)
    local err = libaf.af_release_array(arr)
    check_err(err, "af_release_array")
    return tonumber(err)
end

function luajit_af.af_retain_array(arrayIn)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_retain_array(out, arrayIn)
    check_err(err, "af_retain_array")
    return out[0], tonumber(err)
end

----------------------------------------
--            Evaluation
----------------------------------------
function luajit_af.af_eval(arr)
    local err = libaf.af_eval(arr)
    check_err(err, "af_eval")
    return tonumber(err)
end

function luajit_af.af_eval_multiple(arrays)
    local num = #arrays
    local c_arrays = ffi.new("af_array[?]", num)
    for i = 1, num do
        c_arrays[i - 1] = arrays[i]
    end

    local err = libaf.af_eval_multiple(num, c_arrays)
    check_err(err, "af_eval_multiple")
    return tonumber(err)
end

function luajit_af.af_set_manual_eval_flag(flag)
    local err = libaf.af_set_manual_eval_flag(flag)
    check_err(err, "af_set_manual_eval_flag")
    return tonumber(err)
end

function luajit_af.af_get_manual_eval_flag()
    local flagPtr = ffi.new("bool[1]")
    local err = libaf.af_get_manual_eval_flag(flagPtr)
    check_err(err, "af_get_manual_eval_flag")
    return (flagPtr[0] == true), tonumber(err)
end

----------------------------------------
--       Array Properties
----------------------------------------
function luajit_af.af_get_elements(arr)
    local elems = ffi.new("dim_t[1]")
    local err = libaf.af_get_elements(elems, arr)
    check_err(err, "af_get_elements")
    return tonumber(elems[0]), tonumber(err)
end

function luajit_af.af_get_type(arr)
    local dtype = ffi.new("af_dtype[1]")
    local err = libaf.af_get_type(dtype, arr)
    check_err(err, "af_get_type")
    return dtype[0], tonumber(err)
end

function luajit_af.af_get_dims(arr)
    local d0 = ffi.new("dim_t[1]")
    local d1 = ffi.new("dim_t[1]")
    local d2 = ffi.new("dim_t[1]")
    local d3 = ffi.new("dim_t[1]")
    local err = libaf.af_get_dims(d0, d1, d2, d3, arr)
    check_err(err, "af_get_dims")
    return tonumber(d0[0]), tonumber(d1[0]), tonumber(d2[0]), tonumber(d3[0]), tonumber(err)
end

function luajit_af.af_get_numdims(arr)
    local result = ffi.new("unsigned[1]")
    local err = libaf.af_get_numdims(result, arr)
    check_err(err, "af_get_numdims")
    return tonumber(result[0]), tonumber(err)
end

function luajit_af.af_is_empty(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_empty(result, arr)
    check_err(err, "af_is_empty")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_scalar(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_scalar(result, arr)
    check_err(err, "af_is_scalar")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_row(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_row(result, arr)
    check_err(err, "af_is_row")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_column(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_column(result, arr)
    check_err(err, "af_is_column")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_vector(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_vector(result, arr)
    check_err(err, "af_is_vector")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_complex(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_complex(result, arr)
    check_err(err, "af_is_complex")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_real(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_real(result, arr)
    check_err(err, "af_is_real")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_double(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_double(result, arr)
    check_err(err, "af_is_double")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_single(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_single(result, arr)
    check_err(err, "af_is_single")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_half(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_half(result, arr)
    check_err(err, "af_is_half")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_realfloating(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_realfloating(result, arr)
    check_err(err, "af_is_realfloating")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_floating(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_floating(result, arr)
    check_err(err, "af_is_floating")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_integer(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_integer(result, arr)
    check_err(err, "af_is_integer")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_bool(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_bool(result, arr)
    check_err(err, "af_is_bool")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_is_sparse(arr)
    local result = ffi.new("bool[1]")
    local err = libaf.af_is_sparse(result, arr)
    check_err(err, "af_is_sparse")
    return (result[0] == true), tonumber(err)
end

function luajit_af.af_get_scalar(arr)
    -- Retrieves a single scalar (e.g. if arr is 1x1)
    local storage = ffi.new("double[1]")
    local err = libaf.af_get_scalar(storage, arr)
    check_err(err, "af_get_scalar")
    return storage[0], tonumber(err)
end

----------------------------------------
--          dim4 operators
----------------------------------------
function luajit_af.dim4_operator_add(first, second)
    return libaf.dim4_operator_add(first, second)
end

function luajit_af.dim4_operator_subtract(first, second)
    return libaf.dim4_operator_subtract(first, second)
end

function luajit_af.dim4_operator_multiply(first, second)
    return libaf.dim4_operator_multiply(first, second)
end

----------------------------------------
--     Events (af_event)
----------------------------------------
function luajit_af.af_create_event()
    local evt = ffi.new("af_event[1]")
    local err = libaf.af_create_event(evt)
    check_err(err, "af_create_event")
    return evt[0], tonumber(err)
end

function luajit_af.af_delete_event(eventHandle)
    local err = libaf.af_delete_event(eventHandle)
    check_err(err, "af_delete_event")
    return tonumber(err)
end

function luajit_af.af_mark_event(eventHandle)
    local err = libaf.af_mark_event(eventHandle)
    check_err(err, "af_mark_event")
    return tonumber(err)
end

function luajit_af.af_enqueue_wait_event(eventHandle)
    local err = libaf.af_enqueue_wait_event(eventHandle)
    check_err(err, "af_enqueue_wait_event")
    return tonumber(err)
end

function luajit_af.af_block_event(eventHandle)
    local err = libaf.af_block_event(eventHandle)
    check_err(err, "af_block_event")
    return tonumber(err)
end

----------------------------------------
--       Feature Structures
----------------------------------------
function luajit_af.af_create_features(num)
    local feat = ffi.new("af_features[1]")
    local err = libaf.af_create_features(feat, num)
    check_err(err, "af_create_features")
    return feat[0], tonumber(err)
end

function luajit_af.af_retain_features(feat)
    local out = ffi.new("af_features[1]")
    local err = libaf.af_retain_features(out, feat)
    check_err(err, "af_retain_features")
    return out[0], tonumber(err)
end

function luajit_af.af_get_features_num(feat)
    local num = ffi.new("dim_t[1]")
    local err = libaf.af_get_features_num(num, feat)
    check_err(err, "af_get_features_num")
    return tonumber(num[0]), tonumber(err)
end

function luajit_af.af_get_features_xpos(feat)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_get_features_xpos(arr, feat)
    check_err(err, "af_get_features_xpos")
    return arr[0], tonumber(err)
end

function luajit_af.af_get_features_ypos(feat)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_get_features_ypos(arr, feat)
    check_err(err, "af_get_features_ypos")
    return arr[0], tonumber(err)
end

function luajit_af.af_get_features_score(feat)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_get_features_score(arr, feat)
    check_err(err, "af_get_features_score")
    return arr[0], tonumber(err)
end

function luajit_af.af_get_features_orientation(feat)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_get_features_orientation(arr, feat)
    check_err(err, "af_get_features_orientation")
    return arr[0], tonumber(err)
end

function luajit_af.af_get_features_size(feat)
    local arr = cached_types.af_array_ptr()
    local err = libaf.af_get_features_size(arr, feat)
    check_err(err, "af_get_features_size")
    return arr[0], tonumber(err)
end

function luajit_af.af_release_features(feat)
    local err = libaf.af_release_features(feat)
    check_err(err, "af_release_features")
    return tonumber(err)
end

----------------------------------------
--       Indexing
----------------------------------------
function luajit_af.af_index(in_array, seqsOrIndices, ndims)
    local out = cached_types.af_array_ptr()
    if not ndims then
        ndims = #seqsOrIndices
    end

    local cindices = ffi.new("af_index_t[?]", ndims)
    for i = 1, ndims do
        cindices[i-1] = seqsOrIndices[i]
    end

    -- Some older AF versions had 'af_index()', but official 3.9 often uses 'af_index_gen()'.
    -- If your AF has 'af_index', keep this; otherwise, use af_index_gen below:
    local err = libaf.af_index(out, in_array, cindices, ndims)
    check_err(err, "af_index")
    return out[0], tonumber(err)
end

function luajit_af.af_index_gen(in_array, indices)
    local out = cached_types.af_array_ptr()
    local ndims = #indices
    local cindices = ffi.new("af_index_t[?]", ndims)
    for i = 1, ndims do
        cindices[i-1] = indices[i]
    end

    local err = libaf.af_index_gen(out, in_array, ndims, cindices)
    check_err(err, "af_index_gen")
    return out[0], tonumber(err)
end

function luajit_af.af_assign_gen(lhs, indices, rhs)
    local out = cached_types.af_array_ptr()
    local ndims = #indices
    local cindices = ffi.new("af_index_t[?]", ndims)
    for i = 1, ndims do
        cindices[i-1] = indices[i]
    end

    local err = libaf.af_assign_gen(out, lhs, ndims, cindices, rhs)
    check_err(err, "af_assign_gen")
    return out[0], tonumber(err)
end

function luajit_af.af_assign_seq(lhs, seqs, rhs)
    local ndims = #seqs
    local cseq = ffi.new("af_seq[?]", ndims)
    for i = 1, ndims do
        cseq[i-1] = seqs[i]
    end

    local out = cached_types.af_array_ptr()
    local err = libaf.af_assign_seq(out, lhs, ndims, cseq, rhs)
    check_err(err, "af_assign_seq")
    return out[0], tonumber(err)
end

function luajit_af.af_lookup(in_array, indices, dim)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_lookup(out, in_array, indices, dim)
    check_err(err, "af_lookup")
    return out[0], tonumber(err)
end

function luajit_af.af_create_indexers()
    local indexersPtr = ffi.new("af_index_t*[1]")
    local err = libaf.af_create_indexers(indexersPtr)
    check_err(err, "af_create_indexers")
    return indexersPtr[0], tonumber(err)
end

function luajit_af.af_set_array_indexer(indexers, idx, dim)
    local err = libaf.af_set_array_indexer(indexers, idx, dim)
    check_err(err, "af_set_array_indexer")
    return tonumber(err)
end

function luajit_af.af_set_seq_indexer(indexers, seqAddr, dim, is_batch)
    local err = libaf.af_set_seq_indexer(indexers, seqAddr, dim, is_batch)
    check_err(err, "af_set_seq_indexer")
    return tonumber(err)
end

function luajit_af.af_set_seq_param_indexer(indexers, begin_, end_, step_, dim, is_batch)
    local err = libaf.af_set_seq_param_indexer(indexers, begin_, end_, step_, dim, is_batch)
    check_err(err, "af_set_seq_param_indexer")
    return tonumber(err)
end

function luajit_af.af_release_indexers(indexers)
    local err = libaf.af_release_indexers(indexers)
    check_err(err, "af_release_indexers")
    return tonumber(err)
end

function luajit_af.af_randu(ndims, dims, ctype)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_randu(out, ndims, dims, ctype)
    check_err(err, "af_randu")
    return out[0], tonumber(err)
end

function luajit_af.af_randn(ndims, dims, ctype)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_randn(out, ndims, dims, ctype)
    check_err(err, "af_randn")
    return out[0], tonumber(err)
end

function luajit_af.af_set_seed(seed)
    local err = libaf.af_set_seed(seed)
    check_err(err, "af_set_seed")
    return tonumber(err)
end

function luajit_af.af_get_seed()
    local seedPtr = ffi.new("unsigned long long[1]")
    local err = libaf.af_get_seed(seedPtr)
    check_err(err, "af_get_seed")
    return seedPtr[0], tonumber(err)
end

----------------------------------------
--           Example Ops
----------------------------------------
function luajit_af.af_abs(arrayIn)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_abs(out, arrayIn)
    check_err(err, "af_abs")
    return out[0], tonumber(err)
end

function luajit_af.af_accum(in_array, dim)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_accum(out, in_array, dim)
    check_err(err, "af_accum")
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- af_matmul
--------------------------------------------------------------------------------
function luajit_af.af_matmul(lhs, rhs, optLhs, optRhs)
    optLhs = optLhs or 0  -- AF_MAT_NONE
    optRhs = optRhs or 0  -- AF_MAT_NONE
    local out = cached_types.af_array_ptr()
    local err = libaf.af_matmul(out, lhs, rhs, optLhs, optRhs)
    check_err(err, "af_matmul")
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- af_transpose
--------------------------------------------------------------------------------
function luajit_af.af_transpose(inarr, conjugate)
    local out = cached_types.af_array_ptr()
    local conjVal = (conjugate == true)
    local err = libaf.af_transpose(out, inarr, conjVal)
    check_err(err, "af_transpose")
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Elementwise ops: af_add, af_sub, af_mul, af_div
--------------------------------------------------------------------------------
function luajit_af.af_add(lhs, rhs, batch)
    local out = cached_types.af_array_ptr()
    local b = batch and true or false
    local err = libaf.af_add(out, lhs, rhs, b)
    check_err(err, "af_add")
    return out[0], tonumber(err)
end

function luajit_af.af_sub(lhs, rhs, batch)
    local out = cached_types.af_array_ptr()
    local b = batch and true or false
    local err = libaf.af_sub(out, lhs, rhs, b)
    check_err(err, "af_sub")
    return out[0], tonumber(err)
end

function luajit_af.af_mul(lhs, rhs, batch)
    local out = cached_types.af_array_ptr()
    local b = batch and true or false
    local err = libaf.af_mul(out, lhs, rhs, b)
    check_err(err, "af_mul")
    return out[0], tonumber(err)
end

function luajit_af.af_div(lhs, rhs, batch)
    local out = cached_types.af_array_ptr()
    local b = batch and true or false
    local err = libaf.af_div(out, lhs, rhs, b)
    check_err(err, "af_div")
    return out[0], tonumber(err)
end

-------------------------------------------------------------------------------
--  Scalar ops with older ArrayFire (no direct *scalar functions)
-------------------------------------------------------------------------------
local function makeScalarArray(lhs, scalarVal)
    local d0, d1, d2, d3 = luajit_af.af_get_dims(lhs)
    local dims = cached_types.dim_t_arr4(d0, d1, d2, d3)
    local constOut = cached_types.af_array_ptr()
    local err = libaf.af_constant(constOut, scalarVal, 4, dims, ffi.C.f32)
    check_err(err, "af_constant")
    return constOut[0]
end

function luajit_af.af_add_scalar(lhs, scalarVal)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_add(out, lhs, scalarArr, false)
    check_err(err, "af_add_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

function luajit_af.af_sub_scalar(lhs, scalarVal)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_sub(out, lhs, scalarArr, false)
    check_err(err, "af_sub_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

function luajit_af.af_mul_scalar(lhs, scalarVal)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_mul(out, lhs, scalarArr, false)
    check_err(err, "af_mul_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

function luajit_af.af_div_scalar(lhs, scalarVal)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_div(out, lhs, scalarArr, false)
    check_err(err, "af_div_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

function luajit_af.af_rsub_scalar(scalarVal, lhs)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_sub(out, scalarArr, lhs, false)
    check_err(err, "af_rsub_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

function luajit_af.af_rdiv_scalar(scalarVal, lhs)
    local scalarArr = makeScalarArray(lhs, scalarVal)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_div(out, scalarArr, lhs, false)
    check_err(err, "af_rdiv_scalar")
    luajit_af.af_release_array(scalarArr)
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Casting: af_cast
--------------------------------------------------------------------------------
function luajit_af.af_cast(inarr, dtype)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_cast(out, inarr, dtype)
    check_err(err, "af_cast")
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Summation over all elements: af_sum_all
--------------------------------------------------------------------------------
function luajit_af.af_sum_all(inarr)
    local realPtr = ffi.new("double[1]")
    local imagPtr = ffi.new("double[1]")
    local err = libaf.af_sum_all(realPtr, imagPtr, inarr)
    check_err(err, "af_sum_all")
    return realPtr[0], imagPtr[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Max over all elements: af_max_all
--------------------------------------------------------------------------------
function luajit_af.af_max_all(inarr)
    local realPtr = ffi.new("double[1]")
    local imagPtr = ffi.new("double[1]")
    local err = libaf.af_max_all(realPtr, imagPtr, inarr)
    check_err(err, "af_max_all")
    return realPtr[0], imagPtr[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Exponential: af_exp
--------------------------------------------------------------------------------
function luajit_af.af_exp(inarr)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_exp(out, inarr)
    check_err(err, "af_exp")
    return out[0], tonumber(err)
end

--------------------------------------------------------------------------------
-- Sqrt: af_sqrt
--------------------------------------------------------------------------------
function luajit_af.af_sqrt(inarr)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_sqrt(out, inarr)
    check_err(err, "af_sqrt")
    return out[0], tonumber(err)  -- <== Return out[0]
end

--------------------------------------------------------------------------------
-- Min/Max-of elementwise
--------------------------------------------------------------------------------
function luajit_af.af_minof(lhs, rhs)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_minof(out, lhs, rhs, false)
    check_err(err, "af_minof")
    return out[0], tonumber(err)
end

function luajit_af.af_maxof(lhs, rhs)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_maxof(out, lhs, rhs, false)
    check_err(err, "af_maxof")
    return out[0], tonumber(err)
end

function luajit_af.af_lt_scalar(lhs, scalar)
    local out = cached_types.af_array_ptr()
    local err = libaf.af_lt_scalar(out, lhs, scalar, false)
    check_err(err, "af_lt_scalar")
    return out[0], tonumber(err)
end

----------------------------------------
-- Return the module
----------------------------------------
return luajit_af
