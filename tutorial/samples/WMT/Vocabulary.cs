using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Tutorial.Samples
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
}