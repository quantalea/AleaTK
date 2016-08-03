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
using Alea;
using Alea.Parallel;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using AleaTKUtil;
using csmatio.io;
using NUnit.Framework;
using ICSharpCode.SharpZipLib.Tar;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKUtil.Common;
using Context = AleaTK.Context;

namespace Tutorial.Samples
{
    public static class TrainPtb
    {
        public enum ConfigType
        {
            Small = 0,
            Medium,
            Large
        }

        public const string DataPath = @"Data\PTB\simple-examples\data";
        public const bool Profiling = false;
        public const int TestMaxMaxEpoch = Profiling ? 1 : -1;
        public const int TestHiddenSize = -1;
        public const ConfigType CfgType = ConfigType.Medium;  // ConfigType.Small, ConfigType.Large

        [Test]
        public static void TestLstmAgainstReferenceResults()
        {
            var mfr = new MatFileReader(@"lstm_small.mat");

            var inputSize = mfr.GetInt("InputSize");
            var seqLength = mfr.GetInt("SeqLength");
            var hiddenSize = mfr.GetInt("HiddenSize");
            var batchSize = mfr.GetInt("BatchSize");

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var lstm = new Lstm<float>(x, hiddenSize);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, lstm.Y);

            exe.Initalize();

            var h0 = mfr.GetDoubleArray("h0").Select(n => (float)n).ToArray();
            var c0 = mfr.GetDoubleArray("c0").Select(n => (float)n).ToArray();
            exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(batchSize, hiddenSize)));
            exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(batchSize, hiddenSize)));

            var input = mfr.GetDoubleArray("X").Select(n => (float)n).ToArray();
            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            var w = mfr.GetDoubleArray("W").Select(n => (float)n).ToArray();
            w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();
            exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)));

            exe.Forward();

            var H = mfr.GetDoubleArray("H").Select(n => (float)n).ToArray();
            H.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var myH = exe.GetTensor(lstm.Y).ToArray();
            myH.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            AreClose(H, myH, 1e-6);

            var CN = mfr.GetDoubleArray("cn").Select(n => (float)n).ToArray();
            CN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var myCN = exe.GetTensor(lstm.CY).ToArray();
            myCN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            AreClose(CN, myCN, 1e-6);

            var HN = mfr.GetDoubleArray("hn").Select(n => (float)n).ToArray();
            HN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var myHN = exe.GetTensor(lstm.HY).ToArray();
            myHN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            AreClose(HN, myHN, 1e-6);

            var dH = mfr.GetDoubleArray("dH").Select(n => (float)n).ToArray();
            exe.AssignGradientDirectly(lstm.Y, dH.AsTensor(Shape.Create(seqLength, batchSize, hiddenSize)));

            exe.Backward();

            var dX = mfr.GetDoubleArray("dX").Select(n => (float)n).ToArray();
            dX.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();

            var dXmy = exe.GetGradient(lstm.X).ToArray();
            dXmy.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            AreClose(dX, dXmy, 1e-6);

            var dW = mfr.GetDoubleArray("dW").Select(n => (float)n).ToArray();
            dW.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();

            var dWmy = exe.GetGradient(lstm.W).ToArray();
            dWmy.AsTensor(Shape.Create(lstm.W.Shape.AsArray)).Print();
            AreClose(dW, dWmy, 1e-6);

            var dc0 = mfr.GetDoubleArray("dc0").Select(n => (float)n).ToArray();
            dc0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dc0my = exe.GetGradient(lstm.CX).ToArray();
            dc0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            AreClose(dc0, dc0my, 1e-6);

            var dh0 = mfr.GetDoubleArray("dh0").Select(n => (float)n).ToArray();
            dh0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dh0my = exe.GetGradient(lstm.HX).ToArray();
            dh0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            AreClose(dh0, dh0my, 1e-6);

            ctx.ToGpuContext().Stream.Synchronize();
        }

        [Test]
        public static void TestLstmAgainstCuDnnVersion()
        {
            var ctx = Context.GpuContext(0);
            var inputSize = 5;
            var seqLength = 3;
            var batchSize = 2;
            var hiddenSize = 4;
            var error = 1e-5;

            var data = Context.CpuContext.Eval((2.0f.AsScalar()*
                                                RandomUniform<float>(Shape.Create(seqLength, batchSize, inputSize)) -
                                                1.0f.AsScalar())).ToArray3D();
            //data.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            var h0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var c0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var dy = Context.CpuContext.Eval((2.0f.AsScalar()*
                                              RandomUniform<float>(Shape.Create(seqLength, batchSize, hiddenSize)) -
                                              1.0f.AsScalar())).ToArray3D();
            //dy.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var wi = 0.5f;
            var wf = 0.4f;
            var wo = 0.3f;
            var wa = 0.2f;
            var ui = 0.5f;
            var uf = 0.4f;
            var uo = 0.3f;
            var ua = 0.1f;
            var bi = 0.5f;
            var bf = 0.4f;
            var bo = 0.3f;
            var ba = 0.2f;

            float[,,] y1, y2, dx1, dx2;
            float[,] cy1, cy2, hy1, hy2;
            float[,] dcx1, dcx2, dhx1, dhx2;
            float[,] dw1, dw2;

            {
                // calc with cuDNN
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new Rnn<float>(new LstmRnnType(), x, 1, hiddenSize, dropout: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths
                // cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize * 4 + hiddenSize * 4 + 2 * 4, hiddenSize);
                var offset = 0;
                // Wi
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                offset += inputSize;
                // Wf
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                offset += inputSize;
                // Wa
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                offset += inputSize;
                // Wo
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                offset += inputSize;
                // Ui
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                offset += hiddenSize;
                // Uf
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                offset += hiddenSize;
                // Ua
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                offset += hiddenSize;
                // Uo
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                offset += hiddenSize;
                // Bi
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bi));
                offset++;
                // Bf
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bf));
                offset++;
                // Ba
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), ba));
                offset++;
                // Bo
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bo));

                exe.Forward();

                y1 = exe.GetTensor(lstm.Y).ToArray3D();
                cy1 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy1 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx1 = exe.GetGradient(lstm.X).ToArray3D();
                dcx1 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx1 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();

                // we make dw follow the shape as (1 + inputSize + hiddenSize, 4*hiddenSize), need to transpose because cuDNN uses Fortran storge order
                var dwCUDNN = exe.GetGradient(lstm.W).ToArray().AsTensor();
                dw1 = new float[1 + inputSize + hiddenSize, 4 * hiddenSize];
                var dw1Tensor = Reference<float>(dw1);
                var cpu = Context.CpuContext;
                offset = 0;

                // cuDNN order: IFAO, need to transpose because cuDNN uses Fortran storge order

                // Wi
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wf
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wa
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wo
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Ui
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uf
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Ua
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uo
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Bi
                cpu.Assign(dw1Tensor.Slice(0, Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bf
                cpu.Assign(dw1Tensor.Slice(0, Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Ba
                cpu.Assign(dw1Tensor.Slice(0, Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bo
                cpu.Assign(dw1Tensor.Slice(0, Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
            }

            {
                // calc with direct LSTM implementation
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new Lstm<float>(x, hiddenSize, forgetBiasInit: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor());
                exe.AssignTensor(lstm.HX, h0.AsTensor());

                // set weights
                var w = exe.GetTensor(lstm.W);
                // Wi
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                // Wf
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                // Wo
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                // Wa
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                // Ui
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                // Uf
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                // Uo
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                // Ua
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                // Bi
                ctx.Assign(w.Slice(0, Range(0, hiddenSize)), Fill(Shape.Create(1, hiddenSize), bi));
                // Bf
                ctx.Assign(w.Slice(0, Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), bf));
                // Bo
                ctx.Assign(w.Slice(0, Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), bo));
                // Ba
                ctx.Assign(w.Slice(0, Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), ba));

                exe.Forward();

                y2 = exe.GetTensor(lstm.Y).ToArray3D();
                cy2 = exe.GetTensor(lstm.CY).ToArray2D();
                hy2 = exe.GetTensor(lstm.HY).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx2 = exe.GetGradient(lstm.X).ToArray3D();
                dcx2 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx2 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();
                dw2 = exe.GetGradient(lstm.W).ToArray2D();
            }

            AreClose(y1, y2, error);
            AreClose(cy1, cy2, error);
            AreClose(hy1, hy2, error);
            AreClose(dx1, dx2, error);
            AreClose(dcx1, dcx2, error);
            AreClose(dhx1, dhx2, error);
            AreClose(dw1, dw2, error);
        }

        public class Config
        {
            public double InitScale;
            public double LearningRate;
            public double MaxGradNorm;
            public int NumLayers;
            public int NumSteps;
            public int HiddenSize;
            public int MaxEpoch;    // learning rate start to reduce after this epoch
            public int MaxMaxEpoch; // epoches to run
            public double KeepProb;
            public double LrDecay;
            public int BatchSize;
            public int VocabSize;

            public static Config Small(int batchSize = 20, int numSteps = 20, double keepProb = 1.0)
            {
                return new Config
                {
                    InitScale = 0.1,
                    LearningRate = 1.0,
                    MaxGradNorm = 5.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 200,
                    MaxEpoch = 4,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 13,
                    KeepProb = keepProb,
                    LrDecay = 0.5,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }

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

            public static Config Large(int batchSize = 20, int numSteps = 35, double keepProb = 0.35)
            {
                return new Config
                {
                    InitScale = 0.04,
                    LearningRate = 1.0,
                    MaxGradNorm = 10.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 1500,
                    MaxEpoch = 14,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 55,
                    KeepProb = keepProb,
                    LrDecay = 1.0/1.15,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }
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

                var trainPath = Path.Combine(dataPath, "ptb.train.txt");
                var validPath = Path.Combine(dataPath, "ptb.valid.txt");
                var testPath = Path.Combine(dataPath, "ptb.test.txt");

                BuildVocab(trainPath, out WordToIdDict, out IdToWordDict);

                TrainData = ReadWords(trainPath).Select(WordToId).ToArray();
                ValidData = ReadWords(validPath).Select(WordToId).ToArray();
                TestData = ReadWords(testPath).Select(WordToId).ToArray();
            }

            public List<string> GetWords(int from, int to)
            {
                var words = new List<string>();
                for (var i = from; i < to; ++i)
                    words.Add(IdToWordDict[TrainData[i]]);
                return words;
            }

            public class Batch
            {
                public int[,] Inputs { get; set; }
                public int[,] Targets { get; set; }
            }

            public static IEnumerable<Batch> Iterator(int[] rawData, int numSteps, int batchSize)
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
                    var x = new int[numSteps, batchSize];
                    var y = new int[numSteps, batchSize];

                    for (var t = 0; t < numSteps; ++t)
                    {
                        for (var j = 0; j < batchSize; ++j)
                        {
                            x[t, j] = data[j, numSteps*i + t];
                            y[t, j] = data[j, numSteps*i + t + 1];
                        }
                    }

                    yield return new Batch { Inputs = x, Targets = y };
                }
            }
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

        public class Model
        {
            public Model(Context ctx, Config cfg, bool isTraining = true, bool usingCuDnn = true)
            {
                Config = cfg;
                IsTraining = isTraining;
                UsingCuDnn = usingCuDnn;

                Inputs = Variable<int>(PartialShape.Create(cfg.NumSteps, cfg.BatchSize));
                Targets = Variable<int>(PartialShape.Create(cfg.NumSteps, cfg.BatchSize));

                // embedding
                Embedding = new Embedding<float>(Inputs, cfg.VocabSize, cfg.HiddenSize, initScale: cfg.InitScale);

                // add dropout 
                EmbeddedOutput = Embedding.Output;
                if (isTraining && cfg.KeepProb < 1.0)
                {
                    var dropout = new Dropout<float>(EmbeddedOutput, dropoutProb: 1.0 - cfg.KeepProb);
                    EmbeddedOutput = dropout.Output;
                }

                // rnn layer, dropout for intermediate lstm layers and for output
                if (usingCuDnn)
                {
                    RnnAccelerated = new Rnn<float>(new LstmRnnType(forgetBiasInit: 0.0), EmbeddedOutput, cfg.NumLayers, cfg.HiddenSize, isTraining: isTraining, dropout: isTraining && cfg.KeepProb < 1.0 ? 1.0 - Config.KeepProb : 0.0);
                    RnnOutput = RnnAccelerated.Y;
                    if (isTraining && cfg.KeepProb < 1.0)
                    {
                        var dropout = new Dropout<float>(RnnOutput, dropoutProb: 1.0 - cfg.KeepProb);
                        RnnOutput = dropout.Output;
                    }
                }
                else
                {
                    RnnDirect = new Lstm<float>[cfg.NumLayers];
                    for (var i = 0; i < cfg.NumLayers; ++i)
                    {
                        var lstm = new Lstm<float>(i == 0 ? EmbeddedOutput : RnnOutput, cfg.HiddenSize, forgetBiasInit: 0.0);
                        RnnDirect[i] = lstm;
                        RnnOutput = lstm.Y;
                        if (isTraining && cfg.KeepProb < 1.0)
                        {
                            var dropout = new Dropout<float>(RnnOutput, dropoutProb: 1.0 - cfg.KeepProb);
                            RnnOutput = dropout.Output;
                        }
                    }
                }

                FC = new FullyConnected<float>(RnnOutput.Reshape(RnnOutput.Shape[0]*RnnOutput.Shape[1], RnnOutput.Shape[2]), cfg.VocabSize);

                Loss = new SoftmaxCrossEntropySparse<float>(FC.Output, Targets.Reshape(Targets.Shape[0] * Targets.Shape[1]));

                Optimizer = new GradientDescentOptimizer(ctx, Loss.Loss, cfg.LearningRate, new GlobalNormGradientClipper(cfg.MaxGradNorm));

                // warmup to force JIT compilation to get timings without JIT overhead
                Optimizer.Initalize();
                ResetStates();
                Optimizer.AssignTensor(Inputs, Fill(Shape.Create(Inputs.Shape.AsArray), 0));
                Optimizer.AssignTensor(Targets, Fill(Shape.Create(Targets.Shape.AsArray), 0));
                Optimizer.Forward();
                if (isTraining)
                {
                    Optimizer.Backward();
                }

                // now reset states
                Optimizer.Initalize();
                ResetStates();
            }

            public void CopyWeightsFrom(Model o)
            {
                Optimizer.AssignTensor(Embedding.Weights, o.Optimizer.GetTensor(o.Embedding.Weights));
                Optimizer.AssignTensor(FC.Weights, o.Optimizer.GetTensor(o.FC.Weights));
                Optimizer.AssignTensor(FC.Bias, o.Optimizer.GetTensor(o.FC.Bias));
                if (UsingCuDnn)
                {
                    Util.EnsureTrue(o.UsingCuDnn);
                    Optimizer.AssignTensor(RnnAccelerated.W, o.Optimizer.GetTensor(o.RnnAccelerated.W));
                }
                else
                {
                    Util.EnsureTrue(!o.UsingCuDnn);
                    for (var i = 0; i < Config.NumLayers; ++i)
                    {
                        Optimizer.AssignTensor(RnnDirect[i].W, o.Optimizer.GetTensor(o.RnnDirect[i].W));
                    }
                }
            }

            public void ResetStates()
            {
                if (UsingCuDnn)
                {
                    Optimizer.AssignTensor(RnnAccelerated.CX, Fill(Shape.Create(RnnAccelerated.CX.Shape.AsArray), 0.0f));
                    Optimizer.AssignTensor(RnnAccelerated.HX, Fill(Shape.Create(RnnAccelerated.HX.Shape.AsArray), 0.0f));
                }
                else
                {
                    for (var i = 0; i < Config.NumLayers; ++i)
                    {
                        var lstm = RnnDirect[i];
                        var shape = Shape.Create(Config.BatchSize, lstm.HiddenSize);
                        Optimizer.AssignTensor(lstm.CX, Fill(shape, 0.0f));
                        Optimizer.AssignTensor(lstm.HX, Fill(shape, 0.0f));
                    }
                }
            }

            public void CopyStates()
            {
                if (UsingCuDnn)
                {
                    Optimizer.AssignTensor(RnnAccelerated.CX, Optimizer.GetTensor(RnnAccelerated.CY));
                    Optimizer.AssignTensor(RnnAccelerated.HX, Optimizer.GetTensor(RnnAccelerated.HY));
                }
                else
                {
                    for (var i = 0; i < Config.NumLayers; ++i)
                    {
                        var lstm = RnnDirect[i];
                        Optimizer.AssignTensor(lstm.CX, Optimizer.GetTensor(lstm.CY));
                        Optimizer.AssignTensor(lstm.HX, Optimizer.GetTensor(lstm.HY));
                    }
                }
            }

            public double RunEpoch(int[] data, double learningRate = 1.0, bool verbose = false)
            {
                var cfg = Config;
                var isTraining = IsTraining;
                var epochSize = (data.Length / cfg.BatchSize - 1) / cfg.NumSteps;
                var time = Stopwatch.StartNew();
                var costs = 0.0;
                var iters = 0;
                var step = 0;
                var firstBatch = true;

                foreach (var batch in Data.Iterator(data, cfg.NumSteps, cfg.BatchSize))
                {
                    Optimizer.AssignTensor(Inputs, batch.Inputs.AsTensor());
                    Optimizer.AssignTensor(Targets, batch.Targets.AsTensor());

                    if (firstBatch)
                    {
                        ResetStates();
                        firstBatch = false;
                    }
                    else
                    {
                        CopyStates();
                    }

                    Optimizer.Forward();

                    if (isTraining)
                    {
                        Optimizer.Backward();
                        Optimizer.Optimize(learningRate);
                    }

                    var loss = Optimizer.GetTensor(Loss.Loss).ToScalar();
                    var cost = loss / cfg.BatchSize;
                    costs += cost;
                    iters += cfg.NumSteps;

                    if (Profiling || (verbose && (step % (epochSize / 10) == 10)))
                    {
                        var perplexity = Math.Exp(costs / iters);
                        var wps = (iters * cfg.BatchSize) / (time.Elapsed.TotalMilliseconds / 1000.0);

                        Console.WriteLine($"{step:D4}: {step * 1.0 / epochSize:F3} perplexity: {perplexity:F3} speed:{wps:F0} wps cost: {cost:F3}");
                    }

                    if (Profiling && step > 5) break;

                    step++;
                }
                return Math.Exp(costs / iters);
            }

            public Config Config { get; }

            public bool IsTraining { get; }

            public bool UsingCuDnn { get; }

            public Variable<int> Inputs { get; }

            public Variable<int> Targets { get; }

            public Embedding<float> Embedding { get; }

            public Variable<float> EmbeddedOutput { get; }

            public Lstm<float>[] RnnDirect { get; } 

            public Rnn<float> RnnAccelerated { get; } 

            public Variable<float> RnnOutput { get; }

            public FullyConnected<float> FC { get; }

            public SoftmaxCrossEntropySparse<float> Loss { get; }

            public GradientDescentOptimizer Optimizer { get; }
        }

        [Test, Ignore("Long running test, run it explicitly")]
        public static void Run1()
        {
            Run(false, CfgType, false);
        }

        [Test, Ignore("Long running test, run it explicitly")]
        public static void Run2()
        {
            Run(false, CfgType, true);
        }

        [Test]
        public static void PrintWords()
        {
            var ptb = new Data(DataPath);
            var words = ptb.GetWords(10000, 20000);
            foreach (var w in words)
            {
                Console.Write($"{w} ");
                if (w == "<eos>")
                    Console.WriteLine();
            }
        }

        public static void Run(bool isConsole, ConfigType cfgType, bool usingCuDnn)
        {
            Console.WriteLine($"UsingCUDNN({usingCuDnn}), Config: {cfgType}");

            var ptb = new Data(DataPath);
            var ctx = Context.GpuContext(0);

            Config cfg, cfgValid, cfgTest, cfgInteractive;

            switch (cfgType)
            {
                case ConfigType.Small:
                    cfg = Config.Small(batchSize: 20);
                    cfgValid = Config.Small(batchSize: 20);
                    cfgTest = Config.Small(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Small(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Medium:
                    cfg = Config.Medium(batchSize: 20);
                    cfgValid = Config.Medium(batchSize: 20);
                    cfgTest = Config.Medium(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Medium(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Large:
                    cfg = Config.Large(batchSize: 20);
                    cfgValid = Config.Large(batchSize: 20);
                    cfgTest = Config.Large(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Large(batchSize: 1, numSteps: 10);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(cfgType), cfgType, null);
            }

            Assert.AreEqual(ptb.WordToIdDict.Count, cfg.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgValid.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgTest.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgInteractive.VocabSize);

            var model = new Model(ctx, cfg, isTraining: true, usingCuDnn: usingCuDnn);
            var modelValid = new Model(ctx, cfgValid, isTraining: false, usingCuDnn: usingCuDnn);
            var modelTest = new Model(ctx, cfgTest, isTraining: false, usingCuDnn: usingCuDnn);
            var modelInteractive = new Model(ctx, cfgInteractive, isTraining: false, usingCuDnn: usingCuDnn);

            for (var i = 0; i < cfg.MaxMaxEpoch; ++i)
            {
                var lrDecay = Math.Pow(cfg.LrDecay, Math.Max(i - cfg.MaxEpoch, 0.0));
                var learningRate = cfg.LearningRate*lrDecay;

                Console.WriteLine($"Epoch: {i + 1} Learning rate: {learningRate:F3}");
                var trainPerplexity = model.RunEpoch(ptb.TrainData, learningRate: learningRate, verbose: true);
                Console.WriteLine($"Epoch: {i + 1} Train Perplexity: {trainPerplexity:F3}");

                if (!Profiling)
                {
                    modelValid.CopyWeightsFrom(model);
                    var validPerplexity = modelValid.RunEpoch(ptb.ValidData);
                    Console.WriteLine($"Epoch: {i + 1} Valid Perplexity: {validPerplexity:F3}");
                }
            }

            if (!Profiling)
            {
                modelTest.CopyWeightsFrom(model);
                Console.WriteLine("Testing with test data, this is slow, since batch size is set to small...");
                var testPerplexity = modelTest.RunEpoch(ptb.TestData, verbose: true);
                Console.WriteLine($"Test Perplexity: {testPerplexity:F3}");
            }

            if (!Profiling && isConsole)
            {
                var inputs = new int[cfgInteractive.NumSteps, 1];
                modelInteractive.CopyWeightsFrom(model);

                // since the entropy and softmax are merged, so we have to allocate the target (label) tensor
                modelInteractive.Optimizer.AssignTensor(modelInteractive.Targets, inputs.AsTensor());

                while (true)
                {
                    Console.WriteLine();
                    Console.WriteLine($"Enter some words (less than {cfgInteractive.NumSteps} words)");
                    var readLine = Console.ReadLine();
                    if (readLine == null) break;
                    var line = readLine.Trim(' ', '\t', '\r', '\n');
                    var words = line.Split(new[] {' ', '\t', '\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length <= 0 || words.Length > cfgInteractive.NumSteps) continue;

                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        if (i < words.Length)
                        {
                            inputs[i, 0] = ptb.WordToId(words[i]);
                        }
                        else
                        {
                            inputs[i, 0] = ptb.WordToId("<unk>");
                        }
                    }

                    Console.WriteLine("Your inputs are:");
                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        Console.Write($"{ptb.IdToWord(inputs[i, 0])} ");
                    }
                    Console.WriteLine();

                    modelInteractive.ResetStates();
                    modelInteractive.Optimizer.AssignTensor(modelInteractive.Inputs, inputs.AsTensor());
                    modelInteractive.Optimizer.Forward();

                    var logPred = modelInteractive.Optimizer.GetTensor(modelInteractive.Loss.LogPred).ToArray2D();
                    var pred = new List<IndexAndProb>();
                    var totalProb = 0.0;
                    for (var i = 0; i < cfgInteractive.VocabSize; ++i)
                    {
                        var p = new IndexAndProb {Index = i, Prob = Math.Exp(logPred[words.Length - 1, i])};
                        pred.Add(p);
                        totalProb += p.Prob;
                    }
                    Console.WriteLine($"Total probability: {totalProb:F4}");
                    pred.Sort();
                    Console.WriteLine("Candidates are:");
                    pred.Take(10).Iter((x, o) => { Console.WriteLine($" {x.Prob:P2} --> {ptb.IdToWord(x.Index)}"); });
                }
            }
        }

        private static void Main()
        {
            if (Profiling)
            {
                Run(false, CfgType, false);
                Run(false, CfgType, true);
            }
            else
            {
                Run(true, CfgType, true);
            }
        }
    }
}
