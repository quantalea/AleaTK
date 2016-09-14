using System;
using System.Diagnostics;
using System.Linq;
using Alea.cuDNN;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace Tutorial.Samples {
    public struct Model {
        public SoftmaxCrossEntropy<float> Loss { get; set; }
        public Variable<float> Images { get; set; }
        public Variable<float> Labels { get; set; }
    }

    [TestFixture] public class TrainMNIST {
        private static void PrintStatus(int e, int i, Executor exe, Model model, float[,] images, float[,] labels, string format) {
            var currentLoss = exe.GetTensor(model.Loss.Loss).ToScalar();
            const long batchSize = 1000L;
            var total = labels.GetLength(0);
            var correct = 0L;
            using (var batcher = new Batcher(Context.GpuContext(0), images, labels, false)) {
                var bidx = batcher.Index;
                while (batcher.Next(batchSize, exe, model.Images, model.Labels)) {
                    exe.Forward();
                    var pred = exe.GetTensor(model.Loss.Pred).ToArray2D();
                    correct += TestAccuracy(pred, labels, bidx);
                    bidx = batcher.Index;
                }
            }
            Console.WriteLine(format, e, i, correct, total, correct/(double) total*100.0, currentLoss);
        }

        public static void PrintStatus(int e, int i, Executor exe, Model model, float[,] images, float[,] labels) {
            PrintStatus(e, i, exe, model, images, labels, "#[{0:D2}.{1:D4}] {2}/{3} {4:F2}% LOSS({5:F4})");
        }
        public static void PrintResult(Executor exe, Model model, float[,] images, float[,] labels) {
            PrintStatus(0, 0, exe, model, images, labels, "====> {2}/{3} {4:F2}% <====");
        }

        public static long TestAccuracy(float[,] pred, float[,] label, long idx) {
            var num = pred.GetLength(0);
            var correct = 0L;

            for (var i = 0L; i < num; ++i) {
                var li = i + idx;
                var predv = pred[i, 0];
                var predi = 0;
                var labelv = label[li, 0];
                var labeli = 0;

                for (var j = 1; j < 10; ++j) {
                    if (pred[i, j] > predv) {
                        predv = pred[i, j];
                        predi = j;
                    }
                    if (label[li, j] > labelv) {
                        labelv = label[li, j];
                        labeli = j;
                    }
                }

                if (predi == labeli) correct++;
            }

            return correct;
        }

        public static Model MultinomialRegressionModel() {
            var images = Variable<float>();
            var w = Parameter(Fill(Shape.Create(28*28, 10), 0.0f));
            var b = Parameter(Fill(Shape.Create(10), 1.0f));
            var y = Dot(images, w) + b;
            var labels = Variable<float>();
            return new Model {Loss = new SoftmaxCrossEntropy<float>(y, labels), Images = images, Labels = labels};
        }

        public static Model MultiLayerPerceptronModel() {
            var images = Variable<float>(PartialShape.Create(-1, 28*28));
            ILayer<float> net = new FullyConnected<float>(images, 128);
            net = new ActivationReLU<float>(net.Output);
            net = new FullyConnected<float>(net.Output, 64);
            net = new ActivationReLU<float>(net.Output);
            net = new FullyConnected<float>(net.Output, 10);
            var labels = Variable<float>(PartialShape.Create(-1, 10));

            return new Model {
                Loss = new SoftmaxCrossEntropy<float>(net.Output, labels),
                Images = images,
                Labels = labels
            };
        }

        public static Model ConvolutionalNeuralNetworkModel() {
            var images = Variable<float>();
            var labels = Variable<float>();

            ILayer<float> net = new Reshape<float>(images, PartialShape.Create(-1, 1, 28, 28));
            net = new Convolution2D<float>(net.Output, 5, 5, 16);
            net = new ActivationReLU<float>(net.Output);
            net = new Pooling2D<float>(net.Output, PoolingMode.MAX, 2, 2, 2, 2);

            net = new Convolution2D<float>(net.Output, 5, 5, 32);
            net = new ActivationTanh<float>(net.Output);
            net = new Pooling2D<float>(net.Output, PoolingMode.MAX, 2, 2, 2, 2);

            net = new Reshape<float>(net.Output, PartialShape.Create(-1, net.Output.Shape.Skip(1).Aggregate(ScalarOps.Mul)));
            net = new FullyConnected<float>(net.Output, 50);
            net = new ActivationTanh<float>(net.Output);
            net = new FullyConnected<float>(net.Output, 10);

            return new Model {
                Loss = new SoftmaxCrossEntropy<float>(net.Output, labels),
                Images = images,
                Labels = labels
            };
        }

        [Test] public void CompareConvolutionalNeuralNetwork() {
            var model = ConvolutionalNeuralNetworkModel();
            var ctx = Context.GpuContext(0);

            var memMb = ctx.ToGpuContext().Gpu.Device.TotalMemory/1024.0/1024.0;
            if (memMb < 4096.0) Assert.Inconclusive("Need more Gpu memory.");

            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.000008);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 2; ++i) {
                batcher.Next(2500, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 20; ++i) {
                batcher.Next(2500, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            CleanMem_();
        }

        [Test] public void CompareMultiLayerPerceptron() {
            var model = MultiLayerPerceptronModel();
            var ctx = Context.GpuContext(0);

            var memMb = ctx.ToGpuContext().Gpu.Device.TotalMemory/1024.0/1024.0;
            if (memMb < 4096.0) Assert.Inconclusive("Need more Gpu memory.");

            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.00008);

            // now we need to initalize the parameters for the optimizer
            opt.Initalize();

            // load mnist data
            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 1; ++i) {
                batcher.Next(5000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 5; ++i) {
                batcher.Next(10000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            CleanMem_();
        }

        [Test] public void CompareMultinomialRegression() {
            var model = MultinomialRegressionModel();
            var ctx = Context.GpuContext(0);

            var memMb = ctx.ToGpuContext().Gpu.Device.TotalMemory/1024.0/1024.0;
            if (memMb < 4096.0) Assert.Inconclusive("Need more Gpu memory.");

            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.00005);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            var timer = Stopwatch.StartNew();
            for (var i = 0; i < 1; ++i) {
                batcher.Next(5000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            for (var i = 0; i < 5; ++i) {
                batcher.Next(10000, opt, model.Images, model.Labels);
                opt.Forward();
                opt.Backward();
                opt.Optimize();
            }
            ctx.ToGpuContext().Stream.Synchronize();
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            timer.Restart();
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);
            timer.Stop();
            Console.WriteLine(timer.Elapsed);

            CleanMem_();
        }

        [Test] public void ConvolutionalNeuralNetwork() {
            CleanMem_();
            const long batchSize = 500L;
            const long epochs = 2;

            var model = ConvolutionalNeuralNetworkModel();
            var ctx = Context.GpuContext(0);
            var opt = new RMSpropOptimizer(ctx, model.Loss.Loss, 0.005, 0.9, 1e-9);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            for (var e = 1; e <= epochs; ++e) {
                var i = 0;
                while (batcher.Next(batchSize, opt, model.Images, model.Labels)) {
                    i++;
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if ((i%20 == 0) || ((i == 1) && (e == 1)))
                        PrintStatus(e, i, opt, model, mnist.ValidationImages, mnist.ValidationLabels);
                }
            }
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);

            CleanMem_();
        }

        [Test] public void MultiLayerPerceptron() {
            CleanMem_();
            const double clipNorm = 3.0;
            const long batchSize = 1000L;
            const long epochs = 3;

            var model = MultiLayerPerceptronModel();
            var ctx = Context.GpuContext(0);
            var opt = new RMSpropOptimizer(ctx, model.Loss.Loss, 0.005, 0.9, float.Epsilon,
                new GlobalNormGradientClipper(clipNorm));
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            for (var e = 1; e <= epochs; ++e) {
                var i = 0;
                while (batcher.Next(batchSize, opt, model.Images, model.Labels)) {
                    i++;
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if ((i%10 == 0) || ((i == 1) && (e == 1)))
                        PrintStatus(e, i, opt, model, mnist.ValidationImages, mnist.ValidationLabels);
                }
            }
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);
        }

        [Test] public void MultinomialRegression() {
            CleanMem_();
            const long batchSize = 1000L;
            const long epochs = 3;

            var model = MultinomialRegressionModel();
            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, 0.0005);
            opt.Initalize();

            var mnist = new MNIST();
            var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);

            for (var e = 1; e <= epochs; ++e) {
                var i = 0;
                while (batcher.Next(batchSize, opt, model.Images, model.Labels)) {
                    i++;
                    opt.Forward();
                    opt.Backward();
                    opt.Optimize();

                    if ((i%10 == 0) || ((i == 1) && (e == 1)))
                        PrintStatus(e, i, opt, model, mnist.ValidationImages, mnist.ValidationLabels);
                }
            }
            PrintResult(opt, model, mnist.TestImages, mnist.TestLabels);

            CleanMem_();
        }

        private static void CleanMem_() {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
