using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using Alea;
using AleaTK;
using AleaTKUtil;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTKUtil.Common;

namespace Tutorial.Samples
{
    public static class Test
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

            var batcher = new BucketedDataBatcher(bucketed, 5, new Random(0));
            var batch = batcher.SampleNewBatch();

            batch.Source.AsTensor().Print();
            batch.Target.AsTensor().Print();
            batch.Mask.AsTensor().Print();
        }
    }
}