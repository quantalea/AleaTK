using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using Alea.cuDNN;
using Alea.Parallel.Device;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using Context = AleaTK.Context;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace Tutorial.Samples
{
    public static class TrainMNIST
    {
        public struct Model
        {
            public SoftmaxCrossEntropy<float> Loss { get; set; }
            public Variable<float> Images { get; set; }
            public Variable<float> Labels { get; set; }
        }

        public class MNIST
        {
            public const string Url = @"http://yann.lecun.com/exdb/mnist/";
            public const string FileTrainImages = @"Data\MNIST\train-images-idx3-ubyte";
            public const string FileTrainLabels = @"Data\MNIST\train-labels-idx1-ubyte";
            public const string FileTestImages = @"Data\MNIST\t10k-images-idx3-ubyte";
            public const string FileTestLabels = @"Data\MNIST\t10k-labels-idx1-ubyte";
            public const long NumTrain = 55000L;
            public const long NumTest = 10000L;
            public const long NumValidation = 60000L - NumTrain;

            private static void SkipImages(BinaryReader brImages)
            {
                brImages.ReadInt32(); // skip magic
                brImages.ReadInt32(); // skip num images
                brImages.ReadInt32(); // skip rows
                brImages.ReadInt32(); // skip cols
            }

            private static void SkipLabels(BinaryReader brLabels)
            {
                brLabels.ReadInt32(); // skip magic
                brLabels.ReadInt32(); // skip num labels
            }

            private static void Decompress(string fileName)
            {
                var fileToDecompress = new FileInfo(fileName + ".gz");
                using (var originalFileStream = fileToDecompress.OpenRead())
                using (var decompressedFileStream = File.Create(fileName))
                using (var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
                {
                    decompressionStream.CopyTo(decompressedFileStream);
                }
            }

            public static void Download()
            {
                var files = new [] { "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte" };

                Directory.CreateDirectory(@"Data\MNIST\");
                files.ToList().ForEach(file =>
                {
                    if (!File.Exists(@"Data\MNIST\" + file))
                    {
                        using (var client = new WebClient())
                        {
                            var url = Url + file + ".gz";
                            Console.WriteLine($"Downloading {url} ...");
                            client.DownloadFile(url, @"Data\MNIST\" + file + ".gz");
                            Decompress(@"Data\MNIST\" + file);
                        }
                    }
                });
            }

            private static void ReadData(BinaryReader brImages, BinaryReader brLabels, float[,] images, float[,] labels)
            {
                var numSamples = images.GetLength(0);
                if (numSamples != labels.GetLength(0)) throw new InvalidOperationException();

                for (var i = 0; i < numSamples; ++i)
                {
                    for (var x = 0; x < 28; ++x)
                    {
                        for (var y = 0; y < 28; ++y)
                        {
                            images[i, x * 28 + y] = brImages.ReadByte() / 255.0f;
                        }
                    }
                    labels[i, brLabels.ReadByte()] = 1.0f;
                }
            }

            public float[,] TrainImages { get; }

            public float[,] TrainLabels { get; }

            public float[,] ValidationImages { get; }

            public float[,] ValidationLabels { get; }

            public float[,] TestImages { get; }

            public float[,] TestLabels { get; }

            public MNIST()
            {
                Download();

                using (var ifsTestLabels = new FileStream(FileTestLabels, FileMode.Open))
                using (var ifsTestImages = new FileStream(FileTestImages, FileMode.Open))
                using (var ifsTrainLabels = new FileStream(FileTrainLabels, FileMode.Open))
                using (var ifsTrainImages = new FileStream(FileTrainImages, FileMode.Open))
                using (var brTestLabels = new BinaryReader(ifsTestLabels))
                using (var brTestImages = new BinaryReader(ifsTestImages))
                using (var brTrainLabels = new BinaryReader(ifsTrainLabels))
                using (var brTrainImages = new BinaryReader(ifsTrainImages))
                {
                    SkipImages(brTestImages);
                    SkipLabels(brTestLabels);
                    SkipImages(brTrainImages);
                    SkipLabels(brTrainLabels);

                    TestImages = new float[NumTest, 28 * 28];
                    TestLabels = new float[NumTest, 10];
                    ReadData(brTestImages, brTestLabels, TestImages, TestLabels);

                    TrainImages = new float[NumTrain, 28 * 28];
                    TrainLabels = new float[NumTrain, 10];
                    ReadData(brTrainImages, brTrainLabels, TrainImages, TrainLabels);

                    ValidationImages = new float[NumValidation, 28 * 28];
                    ValidationLabels = new float[NumValidation, 10];
                    ReadData(brTrainImages, brTrainLabels, ValidationImages, ValidationLabels);
                }
            }
        }

        public class Batcher
        {
            private long _walker = 0L;

            public Batcher(Context context, float[,] images, float[,] labels)
            {
                Context = context;
                Random = new Random(0);

                Indices = Enumerable.Range(0, images.GetLength(0)).ToArray();
                Images = images;
                Labels = labels;

                IndicesTensor = context.Allocate(Indices);
                ImagesTensor1 = context.Allocate(images);
                LabelsTensor1 = context.Allocate(labels);

                ImagesTensor2 = context.Device.Allocate<float>(Shape.Create(images.GetLength(0), images.GetLength(1)));
                LabelsTensor2 = context.Device.Allocate<float>(Shape.Create(labels.GetLength(0), labels.GetLength(1)));

                ImagesTensor = ImagesTensor1;
                LabelsTensor = LabelsTensor1;
            }

            public Context Context { get; }

            public Random Random { get; }

            public int[] Indices { get; }

            public float[,] Images { get; }

            public float[,] Labels { get; }

            public Tensor<int> IndicesTensor { get; }

            public Tensor<float> ImagesTensor { get; private set; }

            public Tensor<float> ImagesTensor1 { get; }

            public Tensor<float> ImagesTensor2 { get; }

            public Tensor<float> LabelsTensor { get; private set; }

            public Tensor<float> LabelsTensor1 { get; }

            public Tensor<float> LabelsTensor2 { get; }

            private void ShuffleIndices()
            {
                var rng = Random;
                var array = Indices;
                var n = array.Length;
                while (n > 1)
                {
                    var k = rng.Next(n--);
                    var temp = array[n];
                    array[n] = array[k];
                    array[k] = temp;
                }
            }

            public void Reset()
            {
                if (_walker != 0L)
                {
                    _walker = 0L;

                    if (Context.Type == ContextType.Gpu)
                    {
                        ShuffleIndices();
                        Context.Copy(IndicesTensor, Indices.AsTensor());
                        var stream = Context.ToGpuContext().Stream;
                        var srcImages = ImagesTensor == ImagesTensor1 ? ImagesTensor1.Buffer.Ptr : ImagesTensor2.Buffer.Ptr;
                        var dstImages = ImagesTensor == ImagesTensor1 ? ImagesTensor2.Buffer.Ptr : ImagesTensor1.Buffer.Ptr;
                        var srcLabels = LabelsTensor == LabelsTensor1 ? LabelsTensor1.Buffer.Ptr : LabelsTensor2.Buffer.Ptr;
                        var dstLabels = LabelsTensor == LabelsTensor1 ? LabelsTensor2.Buffer.Ptr : LabelsTensor1.Buffer.Ptr;
                        var idx = IndicesTensor.Buffer.Ptr;
                        DeviceFor.For(stream, 0, Indices.Length, i =>
                        {
                            var j = idx[i];
                            var srcImagesOffseted = srcImages + i * 28 * 28;
                            var dstImagesOffseted = dstImages + j * 28 * 28;
                            for (var k = 0; k < 28 * 28; ++k)
                            {
                                dstImagesOffseted[k] = srcImagesOffseted[k];
                            }
                            var srcLabelsOffseted = srcLabels + i * 10;
                            var dstLabelsOffseted = dstLabels + j * 10;
                            for (var k = 0; k < 10; ++k)
                            {
                                dstLabelsOffseted[k] = srcLabelsOffseted[k];
                            }
                        });
                        ImagesTensor = ImagesTensor == ImagesTensor1 ? ImagesTensor2 : ImagesTensor1;
                        LabelsTensor = LabelsTensor == LabelsTensor1 ? LabelsTensor2 : LabelsTensor1;
                    }
                }
            }

            public void Next(long batchSize, Executor executor, Variable<float> imagesVar, Variable<float> labelsVar)
            {
                var imagesBuffer = new Buffer<float>(ImagesTensor.Device, ImagesTensor.Memory,
                    new Layout(Shape.Create(batchSize, 28 * 28)), ImagesTensor.Buffer.Ptr.LongPtr(_walker * 28 * 28));
                var labelsBuffer = new Buffer<float>(LabelsTensor.Device, LabelsTensor.Memory,
                    new Layout(Shape.Create(batchSize, 10)), LabelsTensor.Buffer.Ptr.LongPtr(_walker * 10));
                _walker += batchSize;

                executor.SetTensor(imagesVar, new Tensor<float>(imagesBuffer));
                executor.SetTensor(labelsVar, new Tensor<float>(labelsBuffer));
            }
        }

        public static void PrintStatus(
            int e, int i, Executor exe,
            Variable<float> lossVar, Variable<float> predVar,
            Variable<float> imagesVar, Variable<float> labelsVar,
            Tensor<float> imagesTensor, Tensor<float> labelsTensor, float[,] labels)
        {
            var currentLoss = exe.GetTensor(lossVar).ToScalar();
            exe.SetTensor(imagesVar, imagesTensor);
            exe.SetTensor(labelsVar, labelsTensor);
            exe.Forward();
            var total = labels.GetLength(0);
            var pred = exe.GetTensor(predVar).ToArray2D();
            var correct = TestAccuracy(pred, labels);
            Console.WriteLine($"#[{e:D2}.{i:D4}] {correct}/{total} {correct / (double)total * 100.0:F2}% LOSS({currentLoss:F4})");
        }

        public static void PrintResult(
            Executor exe,
            Variable<float> predVar,
            Variable<float> imagesVar, Variable<float> labelsVar,
            Tensor<float> imagesTensor, Tensor<float> labelsTensor, float[,] labels)
        {
            exe.SetTensor(imagesVar, imagesTensor);
            exe.SetTensor(labelsVar, labelsTensor);
            exe.Forward();
            var total = labels.GetLength(0);
            var pred = exe.GetTensor(predVar).ToArray2D();
            var correct = TestAccuracy(pred, labels);
            Console.WriteLine($"====> {correct}/{total} {correct / (double)total * 100.0:F2}% <====");
        }

        public static long TestAccuracy(float[,] pred, float[,] label)
        {
            var num = pred.GetLength(0);
            var correct = 0L;

            for (var i = 0L; i < num; ++i)
            {
                var predv = pred[i, 0];
                var predi = 0;
                var labelv = label[i, 0];
                var labeli = 0;

                for (var j = 1; j < 10; ++j)
                {
                    if (pred[i, j] > predv)
                    {
                        predv = pred[i, j];
                        predi = j;
                    }
                    if (label[i, j] > labelv)
                    {
                        labelv = label[i, j];
                        labeli = j;
                    }
                }

                if (predi == labeli)
                {
                    correct++;
                }
            }

            return correct;
        }

        private static void Main()
        {
            MultiLayerPerceptron();
        }

        public static Model MultinomialRegressionModel()
        {
            var images = Variable<float>();
            var w = Parameter(Fill(Shape.Create(28 * 28, 10), 0.0f));
            var b = Parameter(Fill(Shape.Create(10), 1.0f));
            var y = Dot(images, w) + b;
            var labels = Variable<float>();
            return new Model { Loss = new SoftmaxCrossEntropy<float>(y, labels), Images = images, Labels = labels };
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void MultinomialRegression()
        {
            const double eta = 0.0005;
            const long batchSize = 1000L;
            const long epochs = 20;

            var model = MultinomialRegressionModel();
            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, eta);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var validationImages = ctx.Allocate(mnist.ValidationImages);
            var validationLabels = ctx.Allocate(mnist.ValidationLabels);

            for (var e = 1; e <= epochs; ++e)
            {
                batcher.Reset();

                for (var i = 1; i <= MNIST.NumTrain / batchSize; ++i)
                {
                    batcher.Next(batchSize, opt, model.Images, model.Labels);
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if (i % 10 == 0 || (i == 1 && e == 1))
                    {
                        PrintStatus(e, i, opt, model.Loss.Loss, model.Loss.Pred, model.Images, model.Labels, validationImages, validationLabels, mnist.ValidationLabels);
                    }
                }
            }

            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        public static Model MultiLayerPerceptronModel()
        {
            var images = Variable<float>(PartialShape.Create(-1, 28 * 28));
            var fc1 = new FullyConnected<float>(images, 128);
            var act1 = new ActivationReLU<float>(fc1.Output);
            var fc2 = new FullyConnected<float>(act1.Output, 64);
            var act2 = new ActivationReLU<float>(fc2.Output);
            var fc3 = new FullyConnected<float>(act2.Output, 10);
            var labels = Variable<float>(PartialShape.Create(-1, 10));

            return new Model() { Loss = new SoftmaxCrossEntropy<float>(fc3.Output, labels), Images = images, Labels = labels };
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void MultiLayerPerceptron()
        {
            const double eta = 0.005;
            const double rho = 0.9;
            const double clipNorm = 3.0;
            const long batchSize = 1000L;
            const long epochs = 20;

            var model = MultiLayerPerceptronModel();
            var ctx = Context.GpuContext(0);
            var opt = new RMSpropOptimizer(ctx, model.Loss.Loss, eta, rho, float.Epsilon, new GlobalNormGradientClipper(clipNorm));
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var validationImages = ctx.Allocate(mnist.ValidationImages);
            var validationLabels = ctx.Allocate(mnist.ValidationLabels);

            for (var e = 1; e <= epochs; ++e)
            {
                batcher.Reset();

                for (var i = 1; i <= MNIST.NumTrain / batchSize; ++i)
                {
                    batcher.Next(batchSize, opt, model.Images, model.Labels);
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if (i % 10 == 0 || (i == 1 && e == 1))
                    {
                        PrintStatus(e, i, opt, model.Loss.Loss, model.Loss.Pred, model.Images, model.Labels, validationImages, validationLabels, mnist.ValidationLabels);
                    }
                }
            }

            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        public static Model ConvolutionalNeuralNetworkModel()
        {
            var images = Variable<float>();
            var labels = Variable<float>();

            var conv1 = new Convolution2D<float>(images.Reshape(-1, 1, 28, 28), 5, 5, 20);
            var act1 = new ActivationTanh<float>(conv1.Output);
            var pool1 = new Pooling2D<float>(act1.Output, PoolingMode.MAX, 2, 2, 2, 2);

            var conv2 = new Convolution2D<float>(pool1.Output, 5, 5, 50);
            var act2 = new ActivationTanh<float>(conv2.Output);
            var pool2 = new Pooling2D<float>(act2.Output, PoolingMode.MAX, 2, 2, 2, 2);

            var fc1 = new FullyConnected<float>(pool2.Output.Reshape(-1, pool2.Output.Shape.Skip(1).Aggregate(ScalarOps.Mul)), 500);
            var act3 = new ActivationTanh<float>(fc1.Output);
            var fc2 = new FullyConnected<float>(act3.Output, 10);

            return new Model() { Loss = new SoftmaxCrossEntropy<float>(fc2.Output, labels), Images = images, Labels = labels };
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void ConvolutionalNeuralNetwork()
        {
            const double eta = 0.001;
            const double rho = 0.9;
            const long batchSize = 1000L;
            const long epochs = 20;

            var model = ConvolutionalNeuralNetworkModel();
            var ctx = Context.GpuContext(0);
            var opt = new RMSpropOptimizer(ctx, model.Loss.Loss, eta, rho, float.Epsilon);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var validationImages = ctx.Allocate(mnist.ValidationImages);
            var validationLabels = ctx.Allocate(mnist.ValidationLabels);

            for (var e = 1; e <= epochs; ++e)
            {
                batcher.Reset();

                for (var i = 1; i <= MNIST.NumTrain / batchSize; ++i)
                {
                    batcher.Next(batchSize, opt, model.Images, model.Labels);
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if (i % 10 == 0 || (i == 1 && e == 1))
                    {
                        PrintStatus(e, i, opt, model.Loss.Loss, model.Loss.Pred, model.Images, model.Labels, validationImages, validationLabels, mnist.ValidationLabels);
                    }
                }
            }

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();

            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void CompareMultinomialRegression()
        {
            var model = MultinomialRegressionModel();
            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.00005);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 1; ++i)
            {
                batcher.Next(5000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 5; ++i)
            {
                batcher.Next(10000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void CompareMultiLayerPerceptron()
        {
            var model = MultiLayerPerceptronModel();
            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.00008);

            // now we need to initalize the parameters for the optimizer
            opt.Initalize();

            // load mnist data
            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 1; ++i)
            {
                batcher.Next(5000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 5; ++i)
            {
                batcher.Next(10000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        [Test, Ignore("Better explicitly run it.")]
        public static void CompareConvolutionalNeuralNetwork()
        {
            var model = ConvolutionalNeuralNetworkModel();
            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.000008);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 2; ++i)
            {
                batcher.Next(2500, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 20; ++i)
            {
                batcher.Next(2500, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            var testImages = ctx.Allocate(mnist.TestImages);
            var testLabels = ctx.Allocate(mnist.TestLabels);
            PrintResult(opt, model.Loss.Pred, model.Images, model.Labels, testImages, testLabels, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            // cleanup memory
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
