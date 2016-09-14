using System;
using System.Collections.Generic;
using System.Linq;

namespace Tutorial.Samples
{
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
}