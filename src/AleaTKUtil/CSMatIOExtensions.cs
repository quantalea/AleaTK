using System;
using csmatio.io;
using csmatio.types;

namespace AleaTKUtil
{
    public static class CSMatIOExtensions
    {
        public static float GetSingle(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsSingle) throw new InvalidCastException("data is not of type float");
            var n = marray.Size;
            var darray = (MLSingle)marray;
            return darray.GetReal(0);
        }

        public static double GetDouble(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsDouble) throw new InvalidCastException("data is not of type double");
            var n = marray.Size;
            var darray = (MLDouble)marray;
            return darray.GetReal(0);
        }

        public static Int64 GetInt64(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt64) throw new InvalidCastException("data is not of type Int64");
            var n = marray.Size;
            var darray = (MLInt64)marray;
            return darray.GetReal(0);
        }

        public static UInt64 GetUInt64(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt64) throw new InvalidCastException("data is not of type UInt64");
            var n = marray.Size;
            var darray = (MLUInt64)marray;
            return darray.GetReal(0);
        }

        public static int GetInt(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt32) throw new InvalidCastException("data is not of type Int32");
            var n = marray.Size;
            var darray = (MLInt32)marray;
            return darray.GetReal(0);
        }

        public static uint GetUInt(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt32) throw new InvalidCastException("data is not of type UInt32");
            var n = marray.Size;
            var darray = (MLUInt32)marray;
            return darray.GetReal(0);
        }

        public static float[] GetSingleArray(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsSingle) throw new InvalidCastException("data is not of type float");
            var n = marray.Size;
            var darray = (MLSingle) marray;
            var data = new float[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static double[] GetDoubleArray(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsDouble) throw new InvalidCastException("data is not of type double");
            var n = marray.Size;
            var darray = (MLDouble) marray;
            var data = new double[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static Int64[] GetInt64Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt64) throw new InvalidCastException("data is not of type Int64");
            var n = marray.Size;
            var darray = (MLInt64)marray;
            var data = new Int64[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static UInt64[] GetUInt64Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt64) throw new InvalidCastException("data is not of type UInt64");
            var n = marray.Size;
            var darray = (MLUInt64)marray;
            var data = new UInt64[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static int[] GetInt32Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt32) throw new InvalidCastException("data is not of type Int32");
            var n = marray.Size;
            var darray = (MLInt32)marray;
            var data = new int[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static uint[] GetUInt32Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt32) throw new InvalidCastException("data is not of type UInt32");
            var n = marray.Size;
            var darray = (MLUInt32)marray;
            var data = new uint[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }
    }
}