using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using Alea;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using AleaTKUtil;
using csmatio.io;
using csmatio.types;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using Context = AleaTK.Context;

namespace AleaTKTest
{
    public static class BinPackingLab
    {
        [Serializable]
        public class DataPersistence
        {
            public Dictionary<string, Array> Arrays { get; } = new Dictionary<string, Array>();

            public void Save(string path)
            {
                using (var file = File.OpenWrite(path))
                {
                    var serilizer = new BinaryFormatter();
                    serilizer.Serialize(file, this);
                }
            }

            public static DataPersistence Load(string path)
            {
                using (var file = File.OpenRead(path))
                {
                    var serilizer = new BinaryFormatter();
                    return (DataPersistence)serilizer.Deserialize(file);
                }
            }
        }

        public static class Helper
        {
            public static object Load(string path)
            {
                using (var file = File.OpenRead(path))
                {
                    var serilizer = new BinaryFormatter();
                    return serilizer.Deserialize(file);
                }
            }
        }

        [Test]
        public static void TestSave()
        {
            var data = new DataPersistence();
            data.Arrays.Add("array1", new [] {1,2,3,4});
            data.Arrays.Add("array2", new [] {1.1,2.2,3.3,4.4});
            data.Arrays.Add("array3", new[,] { {1,2,3,4}, {11,22,33,44} });
            data.Save("test.bin");
        }

        [Test]
        public static void TestLoad()
        {
            var data = DataPersistence.Load("test.bin");
            Console.WriteLine(data);
        }

        private static void CleanMem()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        public class PackingData
        {
            public float[,] Data { get; }
            public float[,] Label { get; }

            public PackingData(int m, int n, int k, int numSamples_)
            {
                //var dataFile = new MatFileReader("PackingData.mat");
                //var data = dataFile.GetSingleArray2D("data");
                //var label = dataFile.GetInt32Array2D("label");

                var numSamples = numSamples_;
                var data = (float[,]) Helper.Load($@"C:\Projects\clarkkent_2\data_{numSamples}.bin");
                var label = (int[]) Helper.Load($@"C:\Projects\clarkkent_2\label_{numSamples}.bin");

                Assert.AreEqual(numSamples * n, data.GetLength(0));
                Assert.AreEqual(3 * (m + n), data.GetLength(1));
                Assert.AreEqual(numSamples * n, label.Length);

                var labelOneHot = new float[numSamples * n, k];
                for (var i = 0; i < numSamples * n; ++i)
                {
                    var kid = label[i];
                    Assert.IsTrue(kid >= 0 && kid < k);
                    labelOneHot[i, kid] = 1.0f;
                }

                Data = data;
                Label = labelOneHot;
            }
        }

        [Test]
        public static void TestMatFileWriteRead()
        {
            var matdata = new MatFileData();
            matdata.Add("1d", new[] { 2.3f, 4.5f, 9.9f });
            matdata.Add("2d", new[,] { { 1.0f, 2.0f, 3.0f }, { 11.0f, 12.0f, 13.0f } });
            matdata.Save("mymatdata.mat");

            var mfr = new MatFileReader("mymatdata.mat");
            var a1d1 = mfr.GetSingleArray("1d");
            var a1d2 = mfr.GetSingleArray2D("1d");
            var a2d1 = mfr.GetSingleArray("2d");
            var a2d2 = mfr.GetSingleArray2D("2d");
            Console.WriteLine(a1d1.Length);
        }

        public class MultinomialRegressionModel
        {
            public MultinomialRegressionModel(int m, int n, int k)
            {
                M = m;
                N = n;
                K = k;

                // input, boxes, it is of M*3 numbers, first dimension is batch
                X = Variable<float>(PartialShape.Create(-1, (M+N)*3));
                Z = Variable<float>(PartialShape.Create(-1, K));

                //{
                //    var w = Parameter(Fill(Shape.Create((M + N) * 3, K), 0.0f));
                //    var b = Parameter(Fill(Shape.Create(K), 1.0f));
                //    Y = Dot(X, w) + b;
                //}

                {
                    ILayer<float> net = new FullyConnected<float>(X, 1024);
                    net = new ActivationReLU<float>(net.Output);
                    net = new FullyConnected<float>(net.Output, 512);
                    net = new ActivationReLU<float>(net.Output);
                    net = new FullyConnected<float>(net.Output, K);
                    Y = net.Output;
                }

                Loss = new SoftmaxCrossEntropy<float>(Y, Z);
            }

            public int M { get; }
            public int N { get; }
            public int K { get; }

            public Variable<float> X { get; }
            //public Variable<float> W { get; }
            //public Variable<float> B { get; }
            public Variable<float> Y { get; }
            public Variable<float> Z { get; }
            public SoftmaxCrossEntropy<float> Loss { get; }

            public void Train(Context ctx, long batchSize, long epochs, int numSamples)
            {
                //var opt = new GradientDescentOptimizer(ctx, Loss.Loss, 0.00005);
                var opt = new RMSpropOptimizer(ctx, Loss.Loss, 0.00005, 0.9, float.Epsilon,
                    new GlobalNormGradientClipper(3.0));

                opt.Initalize();

                //if (File.Exists("MultinomialRegressionModel.bin"))
                //{
                //    var dataPersistence = DataPersistence.Load("MultinomialRegressionModel.bin");
                //    opt.SetTensor(W, ((float[,])dataPersistence.Arrays["W"]).AsTensor());
                //    opt.SetTensor(B, ((float[])dataPersistence.Arrays["B"]).AsTensor());
                //}

                var data = new PackingData(M, N, K, numSamples);
                var batcher = new Batcher(ctx, data.Data, data.Label);

                for (var e = 1; e <= epochs; ++e)
                {
                    var i = 0;
                    while (batcher.Next(batchSize, opt, X, Z))
                    {
                        i++;
                        opt.Forward();
                        opt.Backward();
                        opt.Optimize();
                        //if ((i%10 == 0) || (i == 1))
                        if (i == 1)
                        {
                            var currentLoss = opt.GetTensor(Loss.Loss).ToScalar();
                            Console.WriteLine($"#.{e}.{i} : LOSS({currentLoss:F4})");
                        }
                    }

                    //var dataPersistence = new DataPersistence();
                    //dataPersistence.Arrays.Add("W", opt.GetTensor(W).ToArray2D());
                    //dataPersistence.Arrays.Add("B", opt.GetTensor(B).ToArray());
                    //dataPersistence.Save("MultinomialRegressionModel.bin");
                }
            }
        }

        [Test]
        public static void TestMultinomialRegressionModel()
        {
            //{
            //    const long batchSize = 100;
            //    const long epochs = 30;
            //    const int m = 15;
            //    const int n = 9;
            //    const int k = 21;
            //    const int numSamples = 1000 * 1;

            //    var model = new MultinomialRegressionModel(m, n, k);
            //    var ctx = Context.GpuContext(0);

            //    model.Train(ctx, batchSize, epochs, numSamples);
            //}

            {
                const long batchSize = 1000;
                const long epochs = 30;
                const int m = 15;
                const int n = 10;
                const int k = 21;
                const int numSamples = 1000 * 10;

                var model = new MultinomialRegressionModel(m, n, k);
                var ctx = Context.GpuContext(0);

                model.Train(ctx, batchSize, epochs, numSamples);
            }
        }
    }
}
