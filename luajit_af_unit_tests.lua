--------------------------------------------------------------------------------
-- test_luajit_af.lua (Refactored for new luajit_af interface)
--------------------------------------------------------------------------------

local luaunit = require("luaunit")
local ffi     = require("ffi")
local af      = require("luajit_af")  -- The refactored luajit_af module

--------------------------------------------------------------------------------
-- Helper: create a small managed array with given dims and data
--         We'll return a Lua object whose .handle is the af_array.
--------------------------------------------------------------------------------
local function createTestArrayFloat(values, dimsTable)
    local n = #values
    local data = ffi.new("float[?]", n)
    for i = 1, n do
        data[i - 1] = values[i]
    end

    -- IMPORTANT: Pass dimsTable directly as a Lua table to af.create_managed_array.
    local arrObj = af.create_managed_array(data, #dimsTable, dimsTable, ffi.C.f32)

    luaunit.assertNotNil(arrObj, "Managed array object should not be nil")
    luaunit.assertNotNil(arrObj.handle, "arrObj.handle should be a valid af_array")
    return arrObj
end

--------------------------------------------------------------------------------
-- 1) Array Creation, Copy, Release
--------------------------------------------------------------------------------
TestArrayCreation = {}

function TestArrayCreation:test_create_and_release()
    -- In this test, we explicitly verify the low-level release call.
    local arrObj = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arr = arrObj.handle  -- raw handle

    local elems, errEl = af.af_get_elements(arr)
    luaunit.assertEquals(errEl, 0)
    luaunit.assertEquals(elems, 4)

    -- Manually call release to ensure the function is tested:
    local errRel = af.af_release_array(arr)
    luaunit.assertEquals(errRel, 0)

    -- Optionally, set arrObj.handle to nil so the GC doesn't try again:
    arrObj.handle = nil
end

function TestArrayCreation:test_copy_array()
    local arrObj = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arr = arrObj.handle

    local arrCopy, errCpy = af.af_copy_array(arr)
    luaunit.assertEquals(errCpy, 0)
    luaunit.assertNotNil(arrCopy)

    local outBuf, n, errGet = af.af_get_data_ptr(arrCopy)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(outBuf[0]), 1)
    luaunit.assertEquals(tonumber(outBuf[1]), 3)
    luaunit.assertEquals(tonumber(outBuf[2]), 2)
    luaunit.assertEquals(tonumber(outBuf[3]), 4)

    -- We do not call af_release_array on arrObj.handle or arrCopy here,
    -- because arrObj is GC-managed, and arrCopy can be manually freed or also left
    -- to be freed if we wrap it in another managed object. For simplicity:
    af.af_release_array(arrCopy)
end

--------------------------------------------------------------------------------
-- 2) Array Properties
--------------------------------------------------------------------------------
TestArrayProperties = {}

function TestArrayProperties:test_array_properties()
    local arrObj = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arr = arrObj.handle

    local d0, d1, d2, d3, errDims = af.af_get_dims(arr)
    luaunit.assertEquals(errDims, 0)
    luaunit.assertEquals(d0, 2)
    luaunit.assertEquals(d1, 2)
    luaunit.assertEquals(d2, 1)
    luaunit.assertEquals(d3, 1)

    local nd, errND = af.af_get_numdims(arr)
    luaunit.assertEquals(errND, 0)
    luaunit.assertTrue(nd >= 2 and nd <= 4)

    local isEmp, errEmpty = af.af_is_empty(arr)
    luaunit.assertEquals(errEmpty, 0)
    luaunit.assertFalse(isEmp)

    local isScalar, errScalar = af.af_is_scalar(arr)
    luaunit.assertEquals(errScalar, 0)
    luaunit.assertFalse(isScalar)

    local isRow, errRow = af.af_is_row(arr)
    luaunit.assertEquals(errRow, 0)
    luaunit.assertFalse(isRow)

    local isCol, errCol = af.af_is_column(arr)
    luaunit.assertEquals(errCol, 0)
    luaunit.assertFalse(isCol)

    local isVec, errVec = af.af_is_vector(arr)
    luaunit.assertEquals(errVec, 0)
    luaunit.assertFalse(isVec)

    local isDbl, errDbl = af.af_is_double(arr)
    luaunit.assertEquals(errDbl, 0)
    luaunit.assertFalse(isDbl)

    local isSgl, errSgl = af.af_is_single(arr)
    luaunit.assertEquals(errSgl, 0)
    luaunit.assertTrue(isSgl)
end

--------------------------------------------------------------------------------
-- 3) Data Transfer
--------------------------------------------------------------------------------
TestDataTransfer = {}

function TestDataTransfer:test_write_array()
    local arrObj = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arr = arrObj.handle

    local newData = ffi.new("float[4]", {5,6,7,8})
    local errWrite = af.af_write_array(arr, newData, ffi.sizeof(newData), ffi.C.afHost)
    luaunit.assertEquals(errWrite, 0)

    local outBuf, n, errGet = af.af_get_data_ptr(arr)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(outBuf[0]), 5)
    luaunit.assertEquals(tonumber(outBuf[1]), 6)
    luaunit.assertEquals(tonumber(outBuf[2]), 7)
    luaunit.assertEquals(tonumber(outBuf[3]), 8)
end

--------------------------------------------------------------------------------
-- 4) Evaluation
--------------------------------------------------------------------------------
TestEvaluation = {}

function TestEvaluation:test_eval_and_manual_flag()
    local arrObj = createTestArrayFloat({1,2}, {2,1,1,1})
    local arr = arrObj.handle

    local errEval = af.af_eval(arr)
    luaunit.assertEquals(errEval, 0)

    local origFlag, errFlag1 = af.af_get_manual_eval_flag()
    luaunit.assertEquals(errFlag1, 0)

    local errSet = af.af_set_manual_eval_flag(not origFlag)
    luaunit.assertEquals(errSet, 0)

    local newFlag, errFlag2 = af.af_get_manual_eval_flag()
    luaunit.assertEquals(errFlag2, 0)
    luaunit.assertEquals(newFlag, not origFlag)

    local errRestore = af.af_set_manual_eval_flag(origFlag)
    luaunit.assertEquals(errRestore, 0)
end

function TestEvaluation:test_eval_multiple()
    local arrA = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arrB = createTestArrayFloat({5,7,6,8}, {2,2,1,1})
    local inputs = {arrA.handle, arrB.handle}
    local errMul = af.af_eval_multiple(inputs)
    luaunit.assertEquals(errMul, 0)
end

--------------------------------------------------------------------------------
-- 5) Random Engine & Random Arrays
--------------------------------------------------------------------------------
TestRandom = {}

function TestRandom:test_randu_randn()
    local dims = ffi.new("dim_t[4]", {5,5,1,1})
    -- These calls return raw af_array, not a managed object, so we can wrap or just test directly:
    local arrRandU, errU = af.af_randu(2, dims, ffi.C.f32)
    luaunit.assertEquals(errU, 0)
    local arrRandN, errN = af.af_randn(2, dims, ffi.C.f32)
    luaunit.assertEquals(errN, 0)

    local sumU_real, sumU_imag, errSumU = af.af_sum_all(arrRandU)
    luaunit.assertEquals(errSumU, 0)
    luaunit.assertTrue(sumU_real >= 0)
    luaunit.assertTrue(sumU_real <= 25)
    luaunit.assertEquals(sumU_imag, 0)

    local sumN_real, sumN_imag, errSumN = af.af_sum_all(arrRandN)
    luaunit.assertEquals(errSumN, 0)
    luaunit.assertTrue(sumN_real ~= 0)
    luaunit.assertEquals(sumN_imag, 0)

    -- Manually release these since we didn't create them via create_managed_array.
    af.af_release_array(arrRandU)
    af.af_release_array(arrRandN)
end

--------------------------------------------------------------------------------
-- 7) Events
--------------------------------------------------------------------------------
TestEvents = {}

function TestEvents:test_create_and_block_event()
    local evt, errEvt = af.af_create_event()
    luaunit.assertEquals(errEvt, 0)
    luaunit.assertNotNil(evt)

    local errMark = af.af_mark_event(evt)
    luaunit.assertEquals(errMark, 0)

    local errBlock = af.af_block_event(evt)
    luaunit.assertEquals(errBlock, 0)

    local errDel = af.af_delete_event(evt)
    luaunit.assertEquals(errDel, 0)
end

--------------------------------------------------------------------------------
-- 8) Feature Structures
--------------------------------------------------------------------------------
TestFeatures = {}

function TestFeatures:test_create_features()
    local feat, errFeat = af.af_create_features(10)
    luaunit.assertEquals(errFeat, 0)
    luaunit.assertNotNil(feat)

    local n, errN = af.af_get_features_num(feat)
    luaunit.assertEquals(errN, 0)
    luaunit.assertEquals(n, 10)

    local errRel = af.af_release_features(feat)
    luaunit.assertEquals(errRel, 0)
end

--------------------------------------------------------------------------------
-- 9) Algebra: Matmul, Transpose
--------------------------------------------------------------------------------
TestAlgebra = {}

function TestAlgebra:test_matmul()
    local aObj = createTestArrayFloat({1,2,3,4}, {2,2,1,1})
    local bObj = createTestArrayFloat({5,6,7,8}, {2,2,1,1})
    local c, errMatmul = af.af_matmul(aObj.handle, bObj.handle, 0, 0)
    luaunit.assertEquals(errMatmul, 0)
    luaunit.assertNotNil(c)

    local outBuf, n, errGet = af.af_get_data_ptr(c)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)

    -- Final result => [23,34,31,46]
    luaunit.assertAlmostEquals(tonumber(outBuf[0]), 23, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[1]), 34, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[2]), 31, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[3]), 46, 1e-5)

    -- Release the result handle explicitly:
    af.af_release_array(c)
end

function TestAlgebra:test_transpose()
    local arrObj = createTestArrayFloat({1,2,3,4,5,6}, {2,3,1,1})
    local arr = arrObj.handle

    local tarr, errT = af.af_transpose(arr, false)
    luaunit.assertEquals(errT, 0)
    luaunit.assertNotNil(tarr)

    local outBuf, n, errGet = af.af_get_data_ptr(tarr)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 6)
    -- Checking transpose:
    luaunit.assertEquals(tonumber(outBuf[0]), 1)
    luaunit.assertEquals(tonumber(outBuf[1]), 3)
    luaunit.assertEquals(tonumber(outBuf[2]), 5)
    luaunit.assertEquals(tonumber(outBuf[3]), 2)
    luaunit.assertEquals(tonumber(outBuf[4]), 4)
    luaunit.assertEquals(tonumber(outBuf[5]), 6)

    af.af_release_array(tarr)
end

--------------------------------------------------------------------------------
-- 10) Elementwise Ops
--------------------------------------------------------------------------------
TestElementwise = {}

function TestElementwise:test_af_add()
    local arrA = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arrB = createTestArrayFloat({10,20,30,40}, {2,2,1,1})

    local arrC, errC = af.af_add(arrA.handle, arrB.handle, false)
    luaunit.assertEquals(errC, 0)
    luaunit.assertNotNil(arrC)

    local outBuf, n, errGet = af.af_get_data_ptr(arrC)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(outBuf[0]), 11)
    luaunit.assertEquals(tonumber(outBuf[1]), 23)
    luaunit.assertEquals(tonumber(outBuf[2]), 32)
    luaunit.assertEquals(tonumber(outBuf[3]), 44)

    af.af_release_array(arrC)
end

function TestElementwise:test_af_sub()
    local arrA = createTestArrayFloat({5,15,10,20}, {2,2,1,1})
    local arrB = createTestArrayFloat({1,2,3,4}, {2,2,1,1})

    local arrC, errC = af.af_sub(arrA.handle, arrB.handle, false)
    luaunit.assertEquals(errC, 0)
    luaunit.assertNotNil(arrC)

    local outBuf, n, errGet = af.af_get_data_ptr(arrC)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(outBuf[0]), 4)
    luaunit.assertEquals(tonumber(outBuf[1]), 13)
    luaunit.assertEquals(tonumber(outBuf[2]), 7)
    luaunit.assertEquals(tonumber(outBuf[3]), 16)

    af.af_release_array(arrC)
end

function TestElementwise:test_af_mul()
    local arrA = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arrB = createTestArrayFloat({10,20,30,40}, {2,2,1,1})

    local arrC, errC = af.af_mul(arrA.handle, arrB.handle, false)
    luaunit.assertEquals(errC, 0)
    luaunit.assertNotNil(arrC)

    local outBuf, n, errGet = af.af_get_data_ptr(arrC)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(outBuf[0]), 10)
    luaunit.assertEquals(tonumber(outBuf[1]), 60)
    luaunit.assertEquals(tonumber(outBuf[2]), 60)
    luaunit.assertEquals(tonumber(outBuf[3]), 160)

    af.af_release_array(arrC)
end

function TestElementwise:test_af_div()
    local arrA = createTestArrayFloat({2,6,4,8}, {2,2,1,1})
    local arrB = createTestArrayFloat({1,2,3,4}, {2,2,1,1})

    local arrC, errC = af.af_div(arrA.handle, arrB.handle, false)
    luaunit.assertEquals(errC, 0)
    luaunit.assertNotNil(arrC)

    local outBuf, n, errGet = af.af_get_data_ptr(arrC)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertAlmostEquals(tonumber(outBuf[0]), 2, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[1]), 3, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[2]), 4/3, 1e-5)
    luaunit.assertAlmostEquals(tonumber(outBuf[3]), 2, 1e-5)

    af.af_release_array(arrC)
end

--------------------------------------------------------------------------------
-- 11) Scalar Ops
--------------------------------------------------------------------------------
TestScalarOps = {}

function TestScalarOps:test_af_add_scalar()
    local arrObj = createTestArrayFloat({1,3,2,4}, {2,2,1,1})
    local arr = arrObj.handle

    local out, err = af.af_add_scalar(arr, 10)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(buf[0]), 11)
    luaunit.assertEquals(tonumber(buf[1]), 13)
    luaunit.assertEquals(tonumber(buf[2]), 12)
    luaunit.assertEquals(tonumber(buf[3]), 14)

    af.af_release_array(out)
end

--------------------------------------------------------------------------------
-- 14) Accumulation, Abs, Minof, Maxof, Sqrt
--------------------------------------------------------------------------------
TestMiscOps = {}

function TestMiscOps:test_af_accum()
    local arrObj = createTestArrayFloat({1,2,3,4,5,6}, {2,3,1,1})
    local arr = arrObj.handle

    local out, err = af.af_accum(arr, 0)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 6)

    -- col0=[1,2] => prefix sums => [1,3]
    -- col1=[3,4] => prefix sums => [3,7]
    -- col2=[5,6] => prefix sums => [5,11]
    luaunit.assertEquals(tonumber(buf[0]), 1)
    luaunit.assertEquals(tonumber(buf[1]), 3)
    luaunit.assertEquals(tonumber(buf[2]), 3)
    luaunit.assertEquals(tonumber(buf[3]), 7)
    luaunit.assertEquals(tonumber(buf[4]), 5)
    luaunit.assertEquals(tonumber(buf[5]), 11)

    af.af_release_array(out)
end

function TestMiscOps:test_af_abs()
    local arrObj = createTestArrayFloat({-1,-3,2,4}, {2,2,1,1})
    local arr = arrObj.handle

    local out, err = af.af_abs(arr)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(buf[0]), 1)
    luaunit.assertEquals(tonumber(buf[1]), 3)
    luaunit.assertEquals(tonumber(buf[2]), 2)
    luaunit.assertEquals(tonumber(buf[3]), 4)

    af.af_release_array(out)
end

function TestMiscOps:test_af_minof()
    local arrA = createTestArrayFloat({1,50,20,7}, {2,2,1,1})
    local arrB = createTestArrayFloat({10,2,15,8}, {2,2,1,1})

    local out, err = af.af_minof(arrA.handle, arrB.handle)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(buf[0]), 1)
    luaunit.assertEquals(tonumber(buf[1]), 2)
    luaunit.assertEquals(tonumber(buf[2]), 15)
    luaunit.assertEquals(tonumber(buf[3]), 7)

    af.af_release_array(out)
end

function TestMiscOps:test_af_maxof()
    local arrA = createTestArrayFloat({1,50,1,7}, {2,2,1,1})
    local arrB = createTestArrayFloat({10,2,15,8}, {2,2,1,1})

    local out, err = af.af_maxof(arrA.handle, arrB.handle)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertEquals(tonumber(buf[0]), 10)
    luaunit.assertEquals(tonumber(buf[1]), 50)
    luaunit.assertEquals(tonumber(buf[2]), 15)
    luaunit.assertEquals(tonumber(buf[3]), 8)

    af.af_release_array(out)
end

function TestMiscOps:test_af_sqrt()
    local arrObj = createTestArrayFloat({1,9,4,16}, {2,2,1,1})
    local arr = arrObj.handle

    local out, err = af.af_sqrt(arr)
    luaunit.assertEquals(err, 0)
    luaunit.assertNotNil(out)

    local buf, n, errGet = af.af_get_data_ptr(out)
    luaunit.assertEquals(errGet, 0)
    luaunit.assertEquals(n, 4)
    luaunit.assertAlmostEquals(tonumber(buf[0]), 1.0, 1e-5)
    luaunit.assertAlmostEquals(tonumber(buf[1]), 3.0, 1e-5)
    luaunit.assertAlmostEquals(tonumber(buf[2]), 2.0, 1e-5)
    luaunit.assertAlmostEquals(tonumber(buf[3]), 4.0, 1e-5)

    af.af_release_array(out)
end

--------------------------------------------------------------------------------
-- Test Runner
--------------------------------------------------------------------------------
local runner = luaunit.LuaUnit.new()
os.exit(runner:runSuite())
