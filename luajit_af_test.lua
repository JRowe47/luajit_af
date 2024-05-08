local luajit_af = require 'luajit_af'
local ffi = require 'ffi'

-- Helper function to initialize test data
local function create_test_array()
    local ndims = 1
    local dims = ffi.new("dim_t[1]", {10})  -- Create an array of size 10
    local dtype = ffi.new("af_dtype", 2)    -- Assuming '2' corresponds to f32
    local arr = ffi.new("af_array[1]")

    -- Fill the data with example values
    local data = ffi.new("float[10]")
    for i = 0, 9 do
        data[i] = i
    end

    return arr, data, ndims, dims, dtype
end

-- Test for af_create_array
local function test_af_create_array()
    local arr, data, ndims, dims, dtype = create_test_array()
    assert(luajit_af.af_create_array(arr, ffi.cast("const void *", data), ndims, dims, dtype) == 0)
    luajit_af.af_release_array(arr[0])
end

-- Test for af_create_handle
local function test_af_create_handle()
    local arr, _, ndims, dims, dtype = create_test_array()
    assert(luajit_af.af_create_handle(arr, ndims, dims, dtype) == 0)
    luajit_af.af_release_array(arr[0])
end

-- Test for af_copy_array
local function test_af_copy_array()
    local arr, data, ndims, dims, dtype = create_test_array()
    luajit_af.af_create_array(arr, ffi.cast("const void *", data), ndims, dims, dtype)
    local arr_copy = ffi.new("af_array[1]")
    assert(luajit_af.af_copy_array(arr_copy, arr[0]) == 0)
    luajit_af.af_release_array(arr[0])
    luajit_af.af_release_array(arr_copy[0])
end

-- Assuming data and other variables are initialized correctly
local function test_af_write_array()
    local ndims = 1
    local dims = ffi.new("dim_t[1]", {10})  -- Creating an array of size 10
    local dtype = ffi.new("af_dtype", 0)    -- Assuming f32 corresponds to 0 in your setup
    local arr = ffi.new("af_array[1]")

    -- Allocate and initialize data matching the array type and size
    local data = ffi.new("float[10]")
    for i = 0, 9 do
        data[i] = i * 2.0  -- Arbitrary data initialization
    end

    luajit_af.af_create_array(arr, ffi.cast("const void *", data), ndims, dims, dtype)

    -- Assuming modification and then write back
    for i = 0, 9 do
        data[i] = data[i] + 1.0
    end
    local bytes = 10 * ffi.sizeof("float")  -- Size must match exactly
    local src = 1

    local err = luajit_af.af_write_array(arr[0], ffi.cast("const void *", data), bytes, src)
    assert(err == 0, "Failed to write array data")

    luajit_af.af_release_array(arr[0])
end


-- Test for af_get_data_ptr
local function test_af_get_data_ptr()
    local arr, data, ndims, dims, dtype = create_test_array()
    luajit_af.af_create_array(arr, ffi.cast("const void *", data), ndims, dims, dtype)
    local retrieved_data = ffi.new("float[10]")
    assert(luajit_af.af_get_data_ptr(ffi.cast("void *", retrieved_data), arr[0]) == 0)
    luajit_af.af_release_array(arr[0])
end

-- Test for af_release_array
local function test_af_release_array()
    local arr, data, ndims, dims, dtype = create_test_array()
    luajit_af.af_create_array(arr, ffi.cast("const void *", data), ndims, dims, dtype)
    assert(luajit_af.af_release_array(arr[0]) == 0)
end

-- Main function to run all tests
local function run_tests()
    test_af_write_array()
    test_af_create_array()
    test_af_create_handle()
    test_af_copy_array()
    test_af_get_data_ptr()
    test_af_release_array()
    print("All tests passed successfully.")
end

run_tests()
