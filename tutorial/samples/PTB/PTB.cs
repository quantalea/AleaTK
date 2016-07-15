using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using ICSharpCode.SharpZipLib;
using ICSharpCode.SharpZipLib.Tar;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace Tutorial.Samples
{
    internal static class TrainPTBUtil
    {
        public static void Iter<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }
    }

    public static class TrainPTB
    {
        public const string DataPath = @"Data\PTB\simple-examples\data";
        public const int TestMaxMaxEpoch = -1;
        public const int TestHiddenSize = -1;

        public class Config
        {
            public double InitScale;
            public double LearningRate;
            public double MaxGradNorm;
            public int NumLayers;
            public int NumSteps;
            public int HiddenSize;
            public int MaxEpoch; // learning rate start to reduce after this epoch
            public int MaxMaxEpoch; // epoches to run
            public double KeepProb;
            public double LrDecay;
            public int BatchSize;
            public int VocabSize;

            public static Config Medium(int batchSize = 20, int numSteps = 35, double keepProb = 0.5)
            {
                return new Config
                {
                    InitScale = 0.05,
                    LearningRate = 1.0,
                    MaxGradNorm = 5.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 650,
                    MaxEpoch = 6,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 39,
                    KeepProb = keepProb,
                    LrDecay = 0.8,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }
        }

        [Test, Ignore("This is just developing test.")]
        public static void TestEnsureDataFile()
        {
            Data.EnsureDataFile();
        }

        public class Data
        {
            private static void Decompress(string src, string dst)
            {
                using (var originalFileStream = File.OpenRead(src))
                using (var decompressedFileStream = File.Create(dst))
                using (var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
                {
                    decompressionStream.CopyTo(decompressedFileStream);
                }
            }

            public static void EnsureDataFile()
            {
                const string doneFileName = @"Data\PTB.done";
                const string url = @"http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz";

                if (!Directory.Exists("Data"))
                {
                    Directory.CreateDirectory("Data");
                }

                if (!File.Exists(doneFileName))
                {
                    using (var client = new WebClient())
                    {
                        Console.WriteLine($"Downloading {url} ...");
                        client.DownloadFile(url, @"Data\PTB.tgz");
                    }

                    Decompress(@"Data\PTB.tgz", @"Data\PTB.tar");

                    using (var tarFile = File.OpenRead(@"Data\PTB.tar"))
                    using (var tarArchive = TarArchive.CreateInputTarArchive(tarFile))
                    {
                        tarArchive.ExtractContents(@"Data\PTB");
                    }

                    using (var doneFile = File.CreateText(doneFileName))
                    {
                        doneFile.WriteLine($"{DateTime.Now}");
                    }
                }
            }

            public static List<string> ReadWords(string path)
            {
                var totalWords = new List<string>();
                using (var file = File.Open(path, FileMode.Open))
                using (var reader = new StreamReader(file, Encoding.UTF8))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var words = line?.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (!(words?.Length > 0)) continue;
                        totalWords.AddRange(words);
                        totalWords.Add("<eos>");
                    }
                }
                return totalWords;
            }

            public static void BuildVocab(string path, out Dictionary<string, int> word2id, out Dictionary<int, string> id2word)
            {
                var data = ReadWords(path).Distinct().ToList();
                data.Sort();
                word2id = new Dictionary<string, int>();
                id2word = new Dictionary<int, string>();
                var id = 0;
                //foreach (var word in data)
                //{
                //    Console.WriteLine(word);
                //}
                foreach (var word in data)
                {
                    word2id.Add(word, id);
                    id2word.Add(id, word);
                    id++;
                }
            }

            public readonly Dictionary<string, int> WordToIdDict;

            public readonly Dictionary<int, string> IdToWordDict;

            public readonly int[] TrainData;

            public readonly int[] ValidData;

            public readonly int[] TestData;

            public int WordToId(string word)
            {
                return WordToIdDict.ContainsKey(word) ? WordToIdDict[word] : WordToIdDict["<unk>"];
            }

            public string IdToWord(int id)
            {
                return IdToWordDict[id];
            }

            public Data(string dataPath)
            {
                EnsureDataFile();

                var TrainPath = Path.Combine(dataPath, "ptb.train.txt");
                var ValidPath = Path.Combine(dataPath, "ptb.valid.txt");
                var TestPath = Path.Combine(dataPath, "ptb.test.txt");

                BuildVocab(TrainPath, out WordToIdDict, out IdToWordDict);

                TrainData = ReadWords(TrainPath).Select(WordToId).ToArray();
                ValidData = ReadWords(ValidPath).Select(WordToId).ToArray();
                TestData = ReadWords(TestPath).Select(WordToId).ToArray();
            }

            public class Batch
            {
                public int[,] Inputs { get; set; }
                public int[,] Targets { get; set; }
            }

            public static IEnumerable<Batch> Iterator(int[] rawData, int batchSize, int numSteps)
            {
                var dataLen = rawData.Length;
                var batchLen = dataLen / batchSize;
                var data = new int[batchSize, batchLen];
                for (var i = 0; i < batchSize; ++i)
                {
                    for (var j = 0; j < batchLen; ++j)
                    {
                        data[i, j] = rawData[batchLen * i + j];
                    }
                }

                var epochSize = (batchLen - 1) / numSteps;

                Util.EnsureTrue(epochSize != 0);

                for (var i = 0; i < epochSize; ++i)
                {
                    var x = new int[batchSize, numSteps];
                    var y = new int[batchSize, numSteps];

                    for (var j = 0; j < batchSize; ++j)
                    {
                        for (var t = 0; t < numSteps; ++t)
                        {
                            x[j, t] = data[j, numSteps * i + t];
                            y[j, t] = data[j, numSteps * i + t + 1];
                        }
                    }

                    yield return new Batch { Inputs = x, Targets = y };
                }
            }
        }

        public class Model
        {
            public Model(Context ctx, Config config, bool isTraining = true)
            {
                Config = config;
                IsTraining = isTraining;

                Inputs = Variable<int>(PartialShape.Create(Config.BatchSize, Config.NumSteps));
                Targets = Variable<int>(PartialShape.Create(Config.BatchSize, Config.NumSteps));

                Embedding = new Embedding<float>(Inputs, Config.VocabSize, Config.HiddenSize, initScale: Config.InitScale);

                RNN = new RNN<float>(Embedding.Output, Config.NumLayers, Config.HiddenSize, isTraining: IsTraining,
                    dropout: Config.KeepProb, bias: 0.0);

                FC = new FullyConnected<float>(RNN.Y.Reshape(RNN.Y.Shape[0] * RNN.Y.Shape[1], RNN.Y.Shape[2]),
                    Config.VocabSize);

                Loss = new SoftmaxCrossEntropySparse<float>(FC.Output,
                    Targets.Reshape(Targets.Shape[0] * Targets.Shape[1]));

                Optimizer = new GradientDescentOptimizer(ctx, Loss.Loss, Config.LearningRate,
                    new GlobalNormGradientClipper(Config.MaxGradNorm));

                Optimizer.Initalize();
            }

            public void CopyWeightsFrom(Model o)
            {
                Optimizer.AssignTensor(Embedding.Weights, o.Optimizer.GetTensor(o.Embedding.Weights));
                Optimizer.AssignTensor(RNN.W, o.Optimizer.GetTensor(o.RNN.W));
                Optimizer.AssignTensor(FC.Weights, o.Optimizer.GetTensor(o.FC.Weights));
                Optimizer.AssignTensor(FC.Bias, o.Optimizer.GetTensor(o.FC.Bias));
            }

            public double RunEpoch(int[] data, double learningRate = 1.0, bool verbose = false)
            {
                var epochSize = (data.Length / Config.BatchSize - 1) / Config.NumSteps;
                var time = Stopwatch.StartNew();
                var costs = 0.0;
                var iters = 0;
                var step = 0;
                RNN.InitializeStates(Optimizer);
                foreach (var batch in Data.Iterator(data, Config.BatchSize, Config.NumSteps))
                {
                    Optimizer.AssignTensor(Inputs, batch.Inputs.AsTensor());
                    Optimizer.AssignTensor(Targets, batch.Targets.AsTensor());

                    Optimizer.Forward();

                    if (IsTraining)
                    {
                        Optimizer.Backward();
                        Optimizer.Optimize(learningRate);
                    }

                    var loss = Optimizer.GetTensor(Loss.Loss).ToScalar();
                    var cost = loss / Config.BatchSize;
                    costs += cost;
                    iters += Config.NumSteps;

                    if (verbose && (step % (epochSize / 10) == 10))
                    //if (true)
                    {
                        var perplexity = Math.Exp(costs / iters);
                        var wps = (iters * Config.BatchSize) / (time.Elapsed.TotalMilliseconds / 1000.0);

                        Console.WriteLine($"{step:D4}: {step * 1.0 / epochSize:F3} perplexity: {perplexity:F3} speed:{wps:F0} wps cost: {cost:F3}");
                    }

                    //if (step > 100) break;

                    step++;
                }
                return Math.Exp(costs / iters);
            }

            public Config Config { get; }

            public bool IsTraining { get; }

            public Variable<int> Inputs { get; }

            public Variable<int> Targets { get; }

            public Embedding<float> Embedding { get; }

            public RNN<float> RNN { get; }

            public FullyConnected<float> FC { get; }

            public SoftmaxCrossEntropySparse<float> Loss { get; }

            public GradientDescentOptimizer Optimizer { get; }
        }

        public class IndexAndProb : IComparable
        {
            public int Index;
            public double Prob;

            public int CompareTo(object obj)
            {
                var o = (IndexAndProb)obj;
                if (Prob == o.Prob) return 0;
                return Prob > o.Prob ? -1 : 1;
            }

            public override string ToString()
            {
                return $"({Index}:{Prob:F2})";
            }
        }

        [Test]
        public static void Run()
        {
            Run(false);
        }

        public static void Run(bool isConsole)
        {
            var ptb = new Data(DataPath);
            var ctx = Context.GpuContext(0);

            var cfg = Config.Medium();
            var cfgValid = Config.Medium(keepProb: 0.0);
            var cfgTest = Config.Medium(batchSize: 1, numSteps: 1, keepProb: 0.0);
            var cfgInteractive = Config.Medium(batchSize: 1, numSteps: 10, keepProb: 0.0);

            Assert.AreEqual(ptb.WordToIdDict.Count, cfg.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgValid.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgTest.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgInteractive.VocabSize);

            var model = new Model(ctx, cfg, isTraining: true);
            var modelValid = new Model(ctx, cfgValid, isTraining: false);
            var modelTest = new Model(ctx, cfgTest, isTraining: false);
            var modelInteractive = new Model(ctx, cfgInteractive, isTraining: false);

            for (var i = 0; i < cfg.MaxMaxEpoch; ++i)
            {
                var lrDecay = Math.Pow(cfg.LrDecay, Math.Max(i - cfg.MaxEpoch, 0.0));
                var learningRate = cfg.LearningRate * lrDecay;

                Console.WriteLine($"Epoch: {i + 1} Learning rate: {learningRate:F3}");
                var trainPerplexity = model.RunEpoch(ptb.TrainData, learningRate: learningRate, verbose: true);
                Console.WriteLine($"Epoch: {i + 1} Train Perplexity: {trainPerplexity:F3}");

                modelValid.CopyWeightsFrom(model);
                var validPerplexity = modelValid.RunEpoch(ptb.ValidData);
                Console.WriteLine($"Epoch: {i + 1} Valid Perplexity: {validPerplexity:F3}");
            }

            modelTest.CopyWeightsFrom(model);
            Console.WriteLine("Testing with test data, this is slow, since batch size is set to small...");
            var testPerplexity = modelTest.RunEpoch(ptb.TestData, verbose: true);
            Console.WriteLine($"Test Perplexity: {testPerplexity:F3}");

            if (isConsole)
            {
                var inputs = new int[1, cfgInteractive.NumSteps];
                modelInteractive.CopyWeightsFrom(model);
                // since the entropy and softmax are merged , so we have to allocate the target (label) tensor
                // this could be improved , by adding some null checking?
                modelInteractive.Optimizer.AssignTensor(modelInteractive.Targets, inputs.AsTensor());

                while (true)
                {
                    Console.WriteLine();
                    Console.WriteLine($"Enter some words (less than {cfgInteractive.NumSteps} words)");
                    var readLine = Console.ReadLine();
                    if (readLine == null) break;
                    var line = readLine.Trim(' ', '\t', '\r', '\n');
                    var words = line.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length <= 0 || words.Length > cfgInteractive.NumSteps) continue;

                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        if (i < words.Length)
                        {
                            inputs[0, i] = ptb.WordToId(words[i]);
                        }
                        else
                        {
                            inputs[0, i] = ptb.WordToId("<unk>");
                        }
                    }

                    Console.WriteLine("Your inputs are:");
                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        Console.Write($"{ptb.IdToWord(inputs[0, i])} ");
                    }
                    Console.WriteLine();

                    modelInteractive.RNN.InitializeStates(modelInteractive.Optimizer);
                    modelInteractive.Optimizer.AssignTensor(modelInteractive.Inputs, inputs.AsTensor());
                    modelInteractive.Optimizer.Forward();

                    var logPred = modelInteractive.Optimizer.GetTensor(modelInteractive.Loss.LogPred).ToArray2D();
                    var pred = new List<IndexAndProb>();
                    var totalProb = 0.0;
                    for (var i = 0; i < cfgInteractive.VocabSize; ++i)
                    {
                        var p = new IndexAndProb { Index = i, Prob = Math.Exp(logPred[words.Length - 1, i]) };
                        pred.Add(p);
                        totalProb += p.Prob;
                    }
                    Console.WriteLine($"Total probability: {totalProb:F4}");
                    pred.Sort();
                    Console.WriteLine("Candidates are:");
                    pred.Take(10).Iter((x, o) =>
                    {
                        Console.WriteLine($" {x.Prob:P2} --> {ptb.IdToWord(x.Index)}");
                    });
                }
            }
        }

        private static void Main()
        {
            Run(true);
        }
    }
}
