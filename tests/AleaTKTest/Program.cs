using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AleaTKTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //Rnn.RnnAgainstRnnDynamic();
            //SequenceToSequenceTest.TestCreateVocabulary();
            //SequenceToSequenceTest.Preprocess();
            //SequenceToSequenceTest.BackTranslate();
            SeqToSeqTest.TestBucketing();
        }
    }
}
