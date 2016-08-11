using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Alea;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using AleaTKTest;
using AleaTKUtil;
using ICSharpCode.SharpZipLib.Tar;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKUtil.Common;
using Context = AleaTK.Context;

namespace AleaTKTest
{
    public class Vocabulary
    {
        public static readonly string Pad = "_PAD_";
        public static readonly string Go = "_GO_";
        public static readonly string Eos = "_EOS_";
        public static readonly string Unk = "_UNK_";
        public static string[] SpecialWords = { Pad, Go, Eos, Unk };

        public static readonly int PadId = 0;
        public static readonly int GoId = 1;
        public static readonly int EosId = 2;
        public static readonly int UnkId = 3;

        public Dictionary<string, int> WordHistogram { get; }

        public string[] Words { get; }
         
        public Dictionary<string, int> TokenIds { get; }

        private static string[] SplitWord(string word)
        {
            return Regex.Split(word, "([.,!?\"':;)(])");
        }

        public static string[] Tokenizer(string sentence)
        {
            var parts = sentence.Trim().Split(null);
            return parts.Select(SplitWord).SelectMany(i => i).ToArray();
        }

        public static string NormalizeDigits(string token)
        {
            return Regex.Replace(token, @"\d+", "0");
        }

        public Vocabulary(IEnumerable<string> words)
        {
            var enumerable = words as string[] ?? words.ToArray();
            WordHistogram = enumerable.ToDictionary(x => x, x => 1);
            Words = enumerable.ToArray();
            TokenIds = Words.Select((w, i) => new KeyValuePair<string, int>(w, i)).ToDictionary(x => x.Key, x => x.Value);
        }

        public Vocabulary(Dictionary<string, int> wordHistogram, int maxVocabularySize)
        {
            var ordered = wordHistogram.OrderByDescending(kv => kv.Value);
            var special = SpecialWords.Select(w => new KeyValuePair<string, int>(w, -1));
            WordHistogram = special.Concat(ordered).Take(maxVocabularySize).ToDictionary(x => x.Key, x => x.Value);
            Words = WordHistogram.Keys.ToArray();
            TokenIds = Words.Select((w, i) => new KeyValuePair<string, int>(w, i)).ToDictionary(x => x.Key, x => x.Value);
        }

        public int TokenId(string word)
        {
            return TokenIds.ContainsKey(word) ? TokenIds[word] : UnkId;
        }

        public int[] SentenceToTokenIds(string sentence, bool normalizeDigits = true)
        {
            var tokens = Tokenizer(sentence);
            return tokens.Select(tok => normalizeDigits ? TokenId(NormalizeDigits(tok)) : TokenId(tok)).ToArray();
        }

        public void Save(string filename)
        {
            using (var file = new StreamWriter(filename))
            {
                foreach (var kv in WordHistogram)
                { 
                    file.WriteLine($"{kv.Key} {kv.Value}");
                }
            }
        }

        public static Vocabulary Load(string filename)
        {
            var wordHistogram = new Dictionary<string, int>();
            using (var file = new StreamReader(filename))
            {
                string line;
                while ((line = file.ReadLine()) != null)
                {
                    var parts = line.Trim().Split();
                    if (!string.IsNullOrEmpty(parts[0])) wordHistogram.Add(parts[0], int.Parse(parts[1]));
                }
            }
            return new Vocabulary(wordHistogram, wordHistogram.Count);
        }
    }

    public class BucketedData
    {
        public Tuple<int, int>[] BucketSequenceLengths { get; }

        public int NumBuckets { get; }

        public int MaxSourceSequenceLength { get; }

        public int MaxTargetSequenceLength { get; }

        public List<int[]>[] SourceLanguage { get; }

        public List<int[]>[] TargetLanguage { get; }

        public int Skipped { get; private set; } = 0;

        public int[] BucketSizes => SourceLanguage.Select(bucket => bucket.Count).ToArray();

        public int NumDataPoints => BucketSizes.Sum();

        public Random Random { get; }

        public BucketedData(IEnumerable<Tuple<int, int>> bucketSequenceLengths)
        {
            BucketSequenceLengths = bucketSequenceLengths.ToArray();
            NumBuckets = BucketSequenceLengths.Length;
            SourceLanguage = Enumerable.Range(0, BucketSequenceLengths.Length).Select(i => new List<int[]>()).ToArray();
            TargetLanguage = Enumerable.Range(0, BucketSequenceLengths.Length).Select(i => new List<int[]>()).ToArray();
            MaxSourceSequenceLength = BucketSequenceLengths.Select(i => i.Item1).Max();
            MaxTargetSequenceLength = BucketSequenceLengths.Select(i => i.Item2).Max() - 2;
            Random = new Random(0);
        }

        public static int[] PadSourceSequence(int[] indices, int paddedLength)
        {
            if (indices.Length > paddedLength)
                throw new ArgumentException("input array too long");

            var padded = new int[paddedLength];
            Array.Copy(indices, padded, indices.Length);
            for (var i = indices.Length; i < paddedLength; ++i)
            {
                padded[i] = Vocabulary.PadId;
            }
            return padded;
        }

        public static int[] PadTargetSequence(int[] indices, int paddedLength)
        {
            if (indices.Length > paddedLength - 2)
                throw new ArgumentException("input array too long, need space for <go> at beginning and <eos> at end");

            var padded = new int[paddedLength];
            padded[0] = Vocabulary.GoId;
            padded[indices.Length + 1] = Vocabulary.EosId;
            Array.Copy(indices, 0, padded, 1, indices.Length);
            for (var i = indices.Length + 2; i < paddedLength; ++i)
            {
                padded[i] = Vocabulary.PadId;
            }
            return padded;
        }

        public void Add(int[] source, int[] target)
        {
            if (source.Length > MaxSourceSequenceLength || target.Length > MaxTargetSequenceLength)
            {
                Skipped++;
                return;
            }
            for (var i = 0; i < BucketSequenceLengths.Length; ++i)
            {
                var sourceBucketLength = BucketSequenceLengths[i].Item1;
                var targetBucketLength = BucketSequenceLengths[i].Item2;
                if (source.Length <= sourceBucketLength && target.Length < targetBucketLength - 1)
                {
                    SourceLanguage[i].Add(PadSourceSequence(source, sourceBucketLength));
                    TargetLanguage[i].Add(PadTargetSequence(target, targetBucketLength));
                    break;
                }
            }
        }


    }

    public class BuckateDataBatcher
    {
        public Random Random { get; }

        public BucketedData Data { get; }

        public double[] CumulativeProbabilities { get; }

        public int BatchSize { get; }

        public BuckateDataBatcher(BucketedData data, int batchSize, Random random)
        {
            Random = random;
            Data = data;
            BatchSize = batchSize;
            var sizes = data.BucketSizes;
            var total = (double)data.NumDataPoints;
            CumulativeProbabilities = new double[data.NumBuckets];
            CumulativeProbabilities[0] = sizes[0]/total;
            for (var i = 1; i < data.NumBuckets; ++i)
                CumulativeProbabilities[i] = CumulativeProbabilities[i - 1] + sizes[i]/total;
        }

        public class Batch
        {
            public int[,] Source { get; set; }
            public int[,] Target { get; set; }
            public int[,] Mask { get; set; }
        }

        public Batch SampleNewBatch(int bucketId)
        {
            var bucketSize = Data.SourceLanguage[bucketId].Count;
            var sourceSequenceLength = Data.BucketSequenceLengths[bucketId].Item1;
            var targetSequenceLength = Data.BucketSequenceLengths[bucketId].Item2;
            var sourceLanguage = Data.SourceLanguage[bucketId];
            var targetLanguage = Data.TargetLanguage[bucketId];

            var source = new int[sourceSequenceLength, BatchSize];
            var target = new int[targetSequenceLength, BatchSize];
            var mask = new int[targetSequenceLength, BatchSize];
            for (var i = 0; i < BatchSize; ++i)
            {
                var choice = Random.Next(bucketSize);
                for (var t = 0; t < sourceSequenceLength; ++t)
                {
                    source[t, i] = sourceLanguage[choice][t];
                }
                for (var t = 0; t < targetSequenceLength; ++t)
                {
                    target[t, i] = targetLanguage[choice][t];
                    if (target[t, i] == Vocabulary.PadId)
                        mask[t, i] = 0;
                    else
                        mask[t, i] = 1;
                }
            }

            return new Batch() { Source = source, Target = target, Mask = mask};
        }

        public Batch SampleNewBatch()
        {
            var random = Random.NextDouble();
            int bucketId;
            for (bucketId = 0; bucketId < Data.NumBuckets - 1; bucketId++)
            {
                if (random <= CumulativeProbabilities[bucketId]) break;
            }
            return SampleNewBatch(bucketId);
        }

        public IEnumerable<Batch> Iterator(int numEpochs)
        {
            for (var i = 0; i < numEpochs; ++i)
                yield return SampleNewBatch();
        }
    }

    public class Data
    {
        public static string Name(string name)
        {
            return Path.Combine("Data", "Wmt15", name);
        }

        private static void Decompress(string src, string dst)
        {
            using (var originalFileStream = File.OpenRead(src))
            using (var decompressedFileStream = File.Create(dst))
            using (var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
            {
                decompressionStream.CopyTo(decompressedFileStream);
            }
        }

        private static void Extract(string src, string dst)
        {
            using (var tarFile = File.OpenRead(src))
            using (var tarArchive = TarArchive.CreateInputTarArchive(tarFile))
            {
                tarArchive.ExtractContents(dst);
            }
        }

        public static void EnsureDataFile()
        {
            const string doneFileName = @"Data\Wmt15.done";
            const string urlTrain = @"http://www.statmt.org/wmt10/training-giga-fren.tar";
            const string urlDev = @"http://www.statmt.org/wmt15/dev-v2.tgz";

            if (!Directory.Exists("Data"))
            {
                Directory.CreateDirectory("Data");
            }

            if (!File.Exists(doneFileName))
            {
                using (var client = new WebClient())
                {
                    if (!File.Exists(Name("training-giga-fren.tar")))
                    {
                        Console.WriteLine($"Downloading {urlTrain} ...");
                        client.DownloadFile(urlTrain, Name("training-giga-fren.tar"));
                    }
                    if (!File.Exists(Name("dev-v2.tgz")))
                    {
                        Console.WriteLine($"Downloading {urlDev} ...");
                        client.DownloadFile(urlDev, Name("dev-v2.tgz"));
                    }
                }

                Console.WriteLine($"Decompressing files ...");
                Decompress(Name("dev-v2.tgz"), Name("dev-v2.tar"));
                Extract(Name("dev-v2.tar"), Name("dev-v2"));
                Extract(Name("training-giga-fren.tar"), Name("training-giga-fren"));
                Decompress(Name(Path.Combine("training-giga-fren", "giga-fren.release2.en.gz")), Name(Path.Combine("training-giga-fren", "giga-fren.release2.en")));
                Decompress(Name(Path.Combine("training-giga-fren", "giga-fren.release2.fr.gz")), Name(Path.Combine("training-giga-fren", "giga-fren.release2.fr")));

                using (var doneFile = File.CreateText(doneFileName))
                {
                    doneFile.WriteLine($"{DateTime.Now}");
                }
            }
        }

        public static Tuple<Vocabulary, Dictionary<string, int>> CreateVocabulary(string filename, int maxVocabularySize, bool normalizeDigits = true, int print = 0)
        {
            var wordHistogram = new Dictionary<string, int>();
            using (var file = new StreamReader(filename, Encoding.UTF8, true))
            {
                string sentence;
                var counter = 0;
                while ((sentence = file.ReadLine()) != null)
                {
                    counter++;
                    if (counter%100000 == 0) Console.WriteLine($"CreateVocabulary from {filename} : line {counter}");
                    
                    var tokens = Vocabulary.Tokenizer(sentence);

                    foreach (var token in tokens)
                    {
                        var tok = normalizeDigits ? Vocabulary.NormalizeDigits(token) : token;

                        if (counter < print) Console.Write($"{tok}, ");

                        if (string.IsNullOrEmpty(tok)) continue;

                        if (wordHistogram.ContainsKey(tok))
                        {
                            wordHistogram[tok] += 1;
                        }
                        else
                        {
                            wordHistogram[tok] = 1;
                        }
                    }

                    if (counter < print) Console.WriteLine();
                }
            }

            var orderedWordHistogram = wordHistogram.OrderByDescending(kv => kv.Value).ToDictionary(x => x.Key, x => x.Value);

            Console.WriteLine($"{filename} : total number of words {orderedWordHistogram.Count}, building vocabulary with {maxVocabularySize} words");
            return new Tuple<Vocabulary, Dictionary<string, int>>(new Vocabulary(orderedWordHistogram, maxVocabularySize), orderedWordHistogram);
        }

        public static void TextToTokenIds(string filename, string tokenizedFilename, Vocabulary vocabulary, bool normalizeDigits = true)
        {
            using (var reader = new StreamReader(filename, Encoding.UTF8, true))
            using (var writer = new StreamWriter(tokenizedFilename))
            {
                string sentence;
                var counter = 0;
                while ((sentence = reader.ReadLine()) != null)
                {
                    counter++;
                    if (counter % 100000 == 0) Console.WriteLine($"TextToTokenIds of {filename} : line {counter}");

                    var tokenIds = vocabulary.SentenceToTokenIds(sentence, normalizeDigits);
                    var line = string.Join(" ", tokenIds.Select(i => i.ToString()));
                    writer.WriteLine(line);
                }
            }           
        }

        public static List<int[]> ReadTokenized(string filename)
        {
            var tokenIds = new List<int[]>();
            using (var reader = new StreamReader(filename, Encoding.UTF8, true))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var ids = line.Trim().Split(null).Select(int.Parse).ToArray();
                    tokenIds.Add(ids);
                }
            }
            return tokenIds;
        }

        public static string[] TokenIdsToText(int[] sentence, Vocabulary vocabulary)
        {
            return sentence.Select(tokenId => vocabulary.Words[tokenId]).ToArray();
        }

        public static List<string[]> TokenIdsToText(List<int[]> sentences, Vocabulary vocabulary)
        {
            return sentences.Select(sentence => TokenIdsToText(sentence, vocabulary).ToArray()).ToList();
        }

        public static List<int[]> FindSentenceContainingTokenId(List<int[]> sentences, int tokenId, int maxNum = 1)
        {
            var sentencesContainingToken = new List<int[]>();
 
            var i = 0;
            var counter = 0;
            foreach (var sentence in sentences)
            {
                counter++;
                if (counter % 1000 == 0) Console.WriteLine($"FindSentenceContainingTokenId : searched in {counter} lines ");

                if (Array.IndexOf(sentence, tokenId) >= 0)
                {
                    sentencesContainingToken.Add(sentence);
                    i++;
                }

                if (i >= maxNum) break;
            }
            return sentencesContainingToken;
        }

        public static void Display(string filename1, string filename2, int count)
        {
            using (var file1 = new StreamReader(filename1, Encoding.UTF8, true))
            using (var file2 = new StreamReader(filename2, Encoding.UTF8, true))
            {
                var i = 0;
                string line1, line2;
                while ((line1 = file1.ReadLine()) != null && (line2 = file2.ReadLine()) != null && i < count)
                {
                    Console.WriteLine($"{line1}");
                    Console.WriteLine($"{line2}");
                    Console.WriteLine();
                    i++;
                }
            }
        }

        public static void Preprocess(string trainingDataFilename, string testDataFilename, string trainingTokenizedFilename, string testTokenizedFilename, 
            string vocabularyFilename, int maxVocabularySize = 50000, bool normalizeDigits = true)
        {
            var vocabulary = CreateVocabulary(trainingDataFilename, maxVocabularySize, normalizeDigits);
            vocabulary.Item1.Save(vocabularyFilename);
            TextToTokenIds(trainingDataFilename, trainingTokenizedFilename, vocabulary.Item1);
            TextToTokenIds(testDataFilename, trainingTokenizedFilename, vocabulary.Item1, normalizeDigits);
        }

        /// <summary>
        /// Tokenized sequences are read from the source and target language file. The sequences are 
        /// distribute into different bucketSequenceLengths according to their sequence length. 
        /// By default the data is padded to the bucket length and the target sequence is prepended with the go symbol id.
        /// </summary>
        /// <param name="sourceLanguage"></param>
        /// <param name="targetLanguage"></param>
        /// <param name="bucketSequenceLengths"></param>
        /// <returns></returns>
        public static BucketedData BucketTokenizedData(string sourceLanguage, string targetLanguage, IEnumerable<Tuple<int, int>> bucketSequenceLengths)
        {
            var bucketedData = new BucketedData(bucketSequenceLengths);

            using (var file1 = new StreamReader(sourceLanguage, Encoding.UTF8, true))
            using (var file2 = new StreamReader(targetLanguage, Encoding.UTF8, true))
            {
                var counter = 0;
                string line1, line2;
                while ((line1 = file1.ReadLine()) != null && (line2 = file2.ReadLine()) != null)
                {
                    var source = line1.Trim().Split(null).Select(int.Parse).ToArray();
                    var target = line2.Trim().Split(null).Select(int.Parse).ToArray();

                    counter++;
                    if (counter%100000 == 0)
                        Console.WriteLine($"PrepareForTraining {sourceLanguage} {targetLanguage} : line {counter}");

                    bucketedData.Add(source, target);
                }
            }
            return bucketedData;
        }
    }

    public class Config
    {
        public double InitScale;
        public double LearningRate;
        public double MaxGradNorm;
        public int NumLayers;
        public int HiddenSize;
        public int ReduceLearningRateAfterEpoch;     
        public int NumEpochs;  
        public double DropoutProbability;
        public double LearningRateDecay;
        public int BatchSize;
        public int VocabularySize;
    }

    public class Model
    {
        public Model(Context ctx, int numInputSteps, Config cfg, bool isTraining = true)
        {
            var addDropout = isTraining && cfg.DropoutProbability > 0.0;

            EncoderInputs = Variable<int>(PartialShape.Create(numInputSteps, cfg.BatchSize));
            Embedding = new Embedding<float>(EncoderInputs, cfg.VocabularySize, cfg.HiddenSize, initScale: cfg.InitScale);

            EmbeddingOutput = addDropout ? new Dropout<float>(Embedding.Output, cfg.DropoutProbability).Output : Embedding.Output;

            var rnnType = new LstmRnnType();
            EncoderRnn = new Rnn<float>(rnnType, EmbeddingOutput, cfg.NumLayers, cfg.HiddenSize, isTraining: isTraining, dropout: addDropout ? cfg.DropoutProbability : 0.0);
            EncoderRnnOutput = addDropout ? new Dropout<float>(EncoderRnn.Y, cfg.DropoutProbability).Output : EncoderRnn.Y;

            // attention model


        }

        public Variable<int> EncoderInputs { get; }

        public Embedding<float> Embedding { get; }

        public Variable<float> EmbeddingOutput { get; }

        public Rnn<float> EncoderRnn { get; }

        public Variable<float> EncoderRnnOutput { get; }
    }

    public static class SequenceToSequenceTest
    {
        [Test]
        public static void TestCaseData()
        {
            Data.EnsureDataFile();
        }

        [Test]
        public static void Preprocess(int maxVocabularySize = 100000, bool normalizeDigits = true)
        {
            var englishTraining = Data.Name(Path.Combine("training-giga-fren", "giga-fren.release2.en"));
            var englishTest = Data.Name(Path.Combine("dev-v2", "dev", "newstest2013.en"));
            var englishTrainingTokenized = Data.Name("english_train.txt");
            var englishTestTokenized = Data.Name("english_dev.txt");
            var englishVocabulary = Data.Name("english_vocabulary.txt");
            Data.Preprocess(englishTraining, englishTest, englishTrainingTokenized, englishTestTokenized, englishVocabulary, maxVocabularySize, normalizeDigits);

            var frenchTraining = Data.Name(Path.Combine("training-giga-fren", "giga-fren.release2.fr"));
            var frenchTest = Data.Name(Path.Combine("dev-v2", "dev", "newstest2013.fr"));
            var frenchTrainingTokenized = Data.Name("french_train.txt");
            var frenchTestTokenized = Data.Name("french_dev.txt");
            var frenchVocabulary = Data.Name("french_vocabulary.txt");
            Data.Preprocess(frenchTraining, frenchTest, frenchTrainingTokenized, frenchTestTokenized, frenchVocabulary, maxVocabularySize, normalizeDigits);
        }

        // highlight word with tokenId in sencence with underscore for better readabiltiy
        private static void Emphasize(IEnumerable<int> sentence, int tokenId, Vocabulary vocabulary)
        {
            var text = sentence.Select(id => id == tokenId ? "__" + vocabulary.Words[id] + "__" : vocabulary.Words[id]).ToArray();
            Console.WriteLine($"{string.Join(" ", text)}");
        }

        private static void BackTranslate(string filename1, string filename2, Vocabulary vocabulary1, Vocabulary vocabulary2, int tokenId, int count, bool first)
        {
            var word = first ? vocabulary1.Words[tokenId] : vocabulary2.Words[tokenId];
            Console.WriteLine($"__{word}__");

            using (var file1 = new StreamReader(filename1, Encoding.UTF8, true))
            using (var file2 = new StreamReader(filename2, Encoding.UTF8, true))
            {
                var i = 0;
                var l = 0;
                string line1, line2;
                while ((line1 = file1.ReadLine()) != null && (line2 = file2.ReadLine()) != null && i < count)
                {
                    var tokens1 = line1.Trim().Split(null).Select(int.Parse).ToArray();
                    var tokens2 = line2.Trim().Split(null).Select(int.Parse).ToArray();
                    var found = first ? Array.IndexOf(tokens1, tokenId) >= 0 : Array.IndexOf(tokens2, tokenId) >= 0;
                    if (found)
                    {
                        i++;
                        Console.WriteLine($"[{l}]");
                        if (first)
                        {
                            Emphasize(tokens1, tokenId, vocabulary1);
                            Console.WriteLine($"{string.Join(" ", Data.TokenIdsToText(tokens2, vocabulary2))}");
                        }
                        else
                        {
                            Console.WriteLine($"{string.Join(" ", Data.TokenIdsToText(tokens1, vocabulary1))}");
                            Emphasize(tokens2, tokenId, vocabulary2);
                        }
                        Console.WriteLine();
                    }

                    l++;
                }
            }
        }

        [Test]
        public static void BackTranslate()
        {
            var vocabulary1 = Vocabulary.Load(Data.Name("english_vocabulary.txt"));
            var vocabulary2 = Vocabulary.Load(Data.Name("french_vocabulary.txt"));

            BackTranslate(Data.Name("english_train.txt"), Data.Name("french_train.txt"), vocabulary1, vocabulary2, 37, 10, true); // dot
            BackTranslate(Data.Name("english_train.txt"), Data.Name("french_train.txt"), vocabulary1, vocabulary2, 28, 10, true); // Canada
        }

        [Test]
        public static void TestCreateVocabulary()
        {
            Data.EnsureDataFile();
            var sentences = Data.Name(Path.Combine("training-giga-fren", "giga-fren.release2.en"));
            var vocabularyFile = Data.Name(Path.Combine("training-giga-fren", "vocabulary.en.txt"));
            var english = Data.CreateVocabulary(sentences, 100000);
            english.Item1.Save(vocabularyFile);
        }

        [Test]
        public static void TestDisplayTraining()
        {
            var en = Data.Name(Path.Combine("training-giga-fren", "giga-fren.release2.en"));
            var fr = Data.Name(Path.Combine("training-giga-fren", "giga-fren.release2.fr"));
            Data.Display(en, fr, 1000);
        }

        [Test]
        public static void TestBucketing()
        {
            Tuple<int, int>[] buckets = { new Tuple<int, int>(10, 15), new Tuple<int, int>(20, 25), new Tuple<int, int>(40, 50), new Tuple<int, int>(50, 60) };

            var bucketed = Data.BucketTokenizedData(Data.Name("english_dev.txt"), Data.Name("french_dev.txt"), buckets);

            var bucketSizes = bucketed.BucketSizes;
            var numDataPoints = bucketed.NumDataPoints;
            bucketSizes.Iter((b, i) => Console.Write($"{b}, "));
            Console.WriteLine($"data points : {numDataPoints}, skipped : {bucketed.Skipped} of total {numDataPoints + bucketed.Skipped}");
        }

        [Test]
        public static void TestBatching()
        {
            Gpu.Get(0);

            Tuple<int, int>[] buckets = { new Tuple<int, int>(10, 15), new Tuple<int, int>(20, 25), new Tuple<int, int>(40, 50), new Tuple<int, int>(50, 60) };

            var bucketed = Data.BucketTokenizedData(Data.Name("english_dev.txt"), Data.Name("french_dev.txt"), buckets);

            var batcher = new BuckateDataBatcher(bucketed, 5, new Random(0));
            var batch = batcher.SampleNewBatch();

            batch.Source.AsTensor().Print();
            batch.Target.AsTensor().Print();
            batch.Mask.AsTensor().Print();
        }
    }
}