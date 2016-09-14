using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;

namespace Tutorial.Samples {
    public class MNIST {
        #region consts

        public const string Url = @"http://yann.lecun.com/exdb/mnist/";
        public const string FileTrainImages = @"Data\MNIST\train-images-idx3-ubyte";
        public const string FileTrainLabels = @"Data\MNIST\train-labels-idx1-ubyte";
        public const string FileTestImages = @"Data\MNIST\t10k-images-idx3-ubyte";
        public const string FileTestLabels = @"Data\MNIST\t10k-labels-idx1-ubyte";
        public const long NumTrain = 55000L;
        public const long NumTest = 10000L;
        public const long NumValidation = 60000L - NumTrain;

        #endregion

        private static void SkipImages(BinaryReader brImages) {
            brImages.ReadInt32(); // skip magic
            brImages.ReadInt32(); // skip num images
            brImages.ReadInt32(); // skip rows
            brImages.ReadInt32(); // skip cols
        }

        private static void SkipLabels(BinaryReader brLabels) {
            brLabels.ReadInt32(); // skip magic
            brLabels.ReadInt32(); // skip num labels
        }

        private static void Decompress(string fileName) {
            var fileToDecompress = new FileInfo(fileName + ".gz");
            using (var originalFileStream = fileToDecompress.OpenRead())
            using (var decompressedFileStream = File.Create(fileName))
            using (var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress)) {
                decompressionStream.CopyTo(decompressedFileStream);
            }
        }

        public static void Download() {
            var files = new[]
                {"train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"};

            Directory.CreateDirectory(@"Data\MNIST\");
            files.ToList().ForEach(file => {
                if (!File.Exists(@"Data\MNIST\" + file)) {
                    using (var client = new WebClient()) {
                        var url = Url + file + ".gz";
                        Console.WriteLine($"Downloading {url} ...");
                        client.DownloadFile(url, @"Data\MNIST\" + file + ".gz");
                        Decompress(@"Data\MNIST\" + file);
                    }
                }
            });
        }

        private static void ReadData(BinaryReader brImages, BinaryReader brLabels, float[,] images, float[,] labels) {
            var numSamples = images.GetLength(0);
            if (numSamples != labels.GetLength(0)) throw new InvalidOperationException();

            for (var i = 0; i < numSamples; ++i) {
                for (var x = 0; x < 28; ++x) {
                    for (var y = 0; y < 28; ++y) {
                        images[i, x*28 + y] = brImages.ReadByte()/255.0f;
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

        public MNIST() {
            Download();

            using (var ifsTestLabels = new FileStream(FileTestLabels, FileMode.Open))
            using (var ifsTestImages = new FileStream(FileTestImages, FileMode.Open))
            using (var ifsTrainLabels = new FileStream(FileTrainLabels, FileMode.Open))
            using (var ifsTrainImages = new FileStream(FileTrainImages, FileMode.Open))
            using (var brTestLabels = new BinaryReader(ifsTestLabels))
            using (var brTestImages = new BinaryReader(ifsTestImages))
            using (var brTrainLabels = new BinaryReader(ifsTrainLabels))
            using (var brTrainImages = new BinaryReader(ifsTrainImages)) {
                SkipImages(brTestImages);
                SkipLabels(brTestLabels);
                SkipImages(brTrainImages);
                SkipLabels(brTrainLabels);

                TestImages = new float[NumTest, 28*28];
                TestLabels = new float[NumTest, 10];
                ReadData(brTestImages, brTestLabels, TestImages, TestLabels);

                TrainImages = new float[NumTrain, 28*28];
                TrainLabels = new float[NumTrain, 10];
                ReadData(brTrainImages, brTrainLabels, TrainImages, TrainLabels);

                ValidationImages = new float[NumValidation, 28*28];
                ValidationLabels = new float[NumValidation, 10];
                ReadData(brTrainImages, brTrainLabels, ValidationImages, ValidationLabels);
            }
        }
    }
}