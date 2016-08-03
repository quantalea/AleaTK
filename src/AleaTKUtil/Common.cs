using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace AleaTKUtil
{
    public static class Common
    {
        public static double[,] RandMat(Random rng, int rows, int cols)
        {
            var mat = new double[rows, cols];
            for (var row = 0; row < rows; ++row)
            {
                for (var col = 0; col < cols; ++col)
                {
                    var u1 = rng.NextDouble();
                    var u2 = rng.NextDouble();
                    var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    mat[row, col] = normal;
                }
            }
            return mat;
        }

        public static double[,] Dot(double[,] matA, double[,] matB)
        {
            var aRows = matA.GetLength(0);
            var aCols = matA.GetLength(1);
            var bRows = matB.GetLength(0);
            var bCols = matB.GetLength(1);

            if (aCols != bRows)
            {
                throw new Exception("Wrong size");
            }

            var matC = new double[aRows, bCols];
            for (var i = 0; i < aRows; ++i)
            {
                for (var j = 0; j < bCols; ++j)
                {
                    for (var k = 0; k < aCols; ++k)
                    {
                        matC[i, j] += matA[i, k] * matB[k, j];
                    }
                }
            }
            return matC;
        }

        public static double[,] Mul(this double[,] matA, double scalar)
        {
            var rows = matA.GetLength(0);
            var cols = matA.GetLength(1);

            var matR = new double[rows, cols];
            for (var row = 0; row < rows; ++row)
            {
                for (var col = 0; col < cols; ++col)
                {
                    matR[row, col] = matA[row, col] * scalar;
                }
            }
            return matR;
        }

        public static double[,] Add(this double[,] matA, double[,] matB)
        {
            var rows = matA.GetLength(0);
            var cols = matA.GetLength(1);

            if (matB.GetLength(0) != rows || matB.GetLength(1) != cols)
            {
                throw new Exception("Bad size");
            }

            var matR = new double[rows, cols];
            for (var row = 0; row < rows; ++row)
            {
                for (var col = 0; col < cols; ++col)
                {
                    matR[row, col] = matA[row, col] + matB[row, col];
                }
            }
            return matR;
        }

        public static T[,] Transpose<T>(T[,] input)
        {
            var rows = input.GetLength(0);
            var cols = input.GetLength(1);
            var output = new T[cols, rows];
            for (var i = 0; i < rows; ++i)
            {
                for (var j = 0; j < cols; ++j)
                {
                    output[j, i] = input[i, j];
                }
            }
            return output;
        }

        public static Random Random = new Random();

        public static T[] GenerateRandomData<T>(int numElements)
        {
            var rawData = new byte[numElements * Marshal.SizeOf(typeof(T))];
            Random.NextBytes(rawData);
            var data = new T[numElements];
            Buffer.BlockCopy(rawData, 0, data, 0, rawData.Length);
            return data;
        }

        public static bool[] GenerateRandomBooleanData(int numElements)
        {
            return Enumerable.Range(0, numElements).Select(_ => Random.Next(0, 2) == 1).ToArray();
        }

        public static double[] GenerateRandomDoubleData(int numElements, double minValue, double maxValue)
        {
            return Enumerable.Range(0, numElements).Select(_ => Random.NextDouble() * (maxValue - minValue) + minValue).ToArray();
        }

        public static float[] GenerateRandomSingleData(int numElements, double minValue, double maxValue)
        {
            return Enumerable.Range(0, numElements).Select(_ => (float)(Random.NextDouble() * (maxValue - minValue) + minValue)).ToArray();
        }

        public static void Dump<T>(T[] array)
        {
            for (var i = 0; i < array.Length; ++i)
            {
                Console.WriteLine($"#.{i}: {array[i]}");
            }
        }

        public static void Dump<T1, T2>(T1[] array1, T2[] array2)
        {
            Assert.AreEqual(array1.Length, array2.Length);
            for (var i = 0; i < array1.Length; ++i)
            {
                Console.WriteLine($"#.{i}: {array1[i]} - {array2[i]}");
            }
        }

        public static void Dump<T1, T2>(T1[] array1, T2[] array2, int atMost)
        {
            Assert.AreEqual(array1.Length, array2.Length);
            for (var i = 0; i < array1.Length && i < atMost; ++i)
            {
                Console.WriteLine($"#.{i}: {array1[i]} - {array2[i]}");
            }
        }

        public static void Iter<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }

        public static T[,] CreateArray2D<T>(int rows, int cols, Func<int, int, T> init)
        {
            var array = new T[rows, cols];
            for (var row = 0; row < array.GetLength(0); ++row)
            {
                for (var col = 0; col < array.GetLength(1); ++col)
                {
                    array[row, col] = init(row, col);
                }
            }
            return array;
        }

        public static void AreClose(double[] expected, double[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static void AreClose(float[] expected, float[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static void AreEqual<T>(T[,] expected, T[,] actual)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var row = 0; row < expected.GetLength(0); ++row)
            {
                for (var col = 0; col < expected.GetLength(1); ++col)
                {
                    Assert.AreEqual(expected[row, col], actual[row, col]);
                }
            }
        }

        public static void AreClose(double[,] expected, double[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var row = 0; row < expected.GetLength(0); ++row)
            {
                for (var col = 0; col < expected.GetLength(1); ++col)
                {
                    Assert.That(actual[row, col], Is.EqualTo(expected[row, col]).Within(error));
                }
            }
        }

        public static void AreClose(double[,,] expected, double[,,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            Assert.AreEqual(expected.GetLength(2), actual.GetLength(2));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    for (var k = 0; k < expected.GetLength(2); ++k)
                    {
                        Assert.That(actual[i, j, k], Is.EqualTo(expected[i, j, k]).Within(error));
                    }
                }
            }
        }

        public static void AreClose(float[,] expected, float[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var row = 0; row < expected.GetLength(0); ++row)
            {
                for (var col = 0; col < expected.GetLength(1); ++col)
                {
                    Assert.That(actual[row, col], Is.EqualTo(expected[row, col]).Within(error));
                }
            }
        }

        public static void AreClose(float[,,] expected, float[,,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            Assert.AreEqual(expected.GetLength(2), actual.GetLength(2));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    for (var k = 0; k < expected.GetLength(2); ++k)
                    {
                        Assert.That(actual[i, j, k], Is.EqualTo(expected[i, j, k]).Within(error));
                    }
                }
            }
        }
    }
}
