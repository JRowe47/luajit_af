local luajit_af = require 'luajit_af'

local function test_create_and_release_array()
    local dims = {2, 2}
    local data = {1.0, 2.0, 3.0, 4.0}
    local arr = luajit_af.create_array(data, dims)
    print("Array Created:")
    luajit_af.print_array(arr)
    luajit_af.release_array(arr)
    print("Array Released Successfully")
end

local function test_randu()
    local dims = {5}
    local arr = luajit_af.randu(dims)
    print("Random Array:")
    luajit_af.print_array(arr)
    luajit_af.release_array(arr)
end

local function test_exp()
    local dims = {1}
    local data = {1.0}
    local arr = luajit_af.create_array(data, dims)
    local result = luajit_af.exp(arr)
    print("Exponential Result:")
    luajit_af.print_array(result)
    luajit_af.release_array(arr)
    luajit_af.release_array(result)
end

local function test_arithmetic_operations()
    local dims = {1}
    local data1 = {10.0}
    local data2 = {2.0}
    local arr1 = luajit_af.create_array(data1, dims)
    local arr2 = luajit_af.create_array(data2, dims)

    local add_result = luajit_af.add(arr1, arr2)
    print("Addition Result:")
    luajit_af.print_array(add_result)

    local sub_result = luajit_af.sub(arr1, arr2)
    print("Subtraction Result:")
    luajit_af.print_array(sub_result)

    local mul_result = luajit_af.mul(arr1, arr2)
    print("Multiplication Result:")
    luajit_af.print_array(mul_result)

    local div_result = luajit_af.div(arr1, arr2)
    print("Division Result:")
    luajit_af.print_array(div_result)

    luajit_af.release_array(arr1)
    luajit_af.release_array(arr2)
    luajit_af.release_array(add_result)
    luajit_af.release_array(sub_result)
    luajit_af.release_array(mul_result)
    luajit_af.release_array(div_result)
end

local function test_set_seed()
    luajit_af.set_seed(12345)
    test_randu() -- Should generate a predictable pattern if seed works
end

print("Testing create and release array...")
test_create_and_release_array()

print("\nTesting random number generation...")
test_randu()

print("\nTesting exponential function...")
test_exp()

print("\nTesting arithmetic operations...")
test_arithmetic_operations()

print("\nTesting set seed functionality...")
test_set_seed()

print("Complete")
