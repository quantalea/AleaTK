using System;
using System.Collections.Generic;

namespace Tutorial.Samples
{
    public class BucketedDataBatcher
    {
        public Random Random { get; }

        public BucketedData Data { get; }

        public double[] CumulativeProbabilities { get; }

        public int BatchSize { get; }

        public BucketedDataBatcher(BucketedData data, int batchSize, Random random)
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
}