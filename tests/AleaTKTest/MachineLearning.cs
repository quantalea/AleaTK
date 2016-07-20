using System;
using System.Linq;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using csmatio.io;
using csmatio.types;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKTest.Common;

namespace AleaTKTest
{
    public static class MachineLearning
    {
        [Test]
        public static void SimpleLogisticRegression()
        {
            //const int N = 8;
            //const int D = 5;
            //const int P = 3;
            //const double learn = 0.001;

            const int N = 100;
            const int D = 784;
            const int P = 10;
            const double learn = 0.00005;

            var input = Variable<double>();
            var label = Variable<double>();
            var weights = Parameter(0.01 * RandomUniform<double>(Shape.Create(D, P)));
            var pred = Dot(input, weights);
            var loss = L2Loss(pred, label);

            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, loss, learn);

            // set some data
            var rng = new Random(42);
            var _input = RandMat(rng, N, D);
            var _label = Dot(_input, RandMat(rng, D, P)).Add(RandMat(rng, N, P).Mul(0.1));
            opt.AssignTensor(input, _input.AsTensor());
            opt.AssignTensor(label, _label.AsTensor());

            opt.Initalize();
            for (var i = 0; i < 800; ++i)
            {
                opt.Forward();
                opt.Backward();
                opt.Optimize();
                if (i % 20 == 0)
                {
                    Console.WriteLine($"loss = {opt.GetTensor(loss).ToScalar()}");
                }
            }
        }

        [Test]
        public static void TestRNN()
        {
            const int miniBatch = 64;
            const int seqLength = 20;
            const int numLayers = 2;
            const int hiddenSize = 512;
            const int inputSize = hiddenSize;

            var x = Variable<float>(PartialShape.Create(miniBatch, seqLength, inputSize));
            //var x = Variable<float>(PartialShape.Create(miniBatch, inputSize, seqLength));
            var rnn = new RNN<float>(x, numLayers, hiddenSize);

            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, rnn.Y, 0.001);
            opt.Initalize();

            opt.AssignTensor(x, Fill(Shape.Create(miniBatch, seqLength, inputSize), 1.0f));
            //opt.AssignTensor(x, Fill(Shape.Create(miniBatch, inputSize, seqLength), 1.0f));

            opt.Forward();
            opt.Backward();
            ctx.ToGpuContext().Stream.Synchronize();
        }

        private static void RandomMat(float[,,] mat, Random rng)
        {
            for (var i = 0; i < mat.GetLength(0); ++i)
            {
                for (var j = 0; j < mat.GetLength(1); ++j)
                {
                    for (var k = 0; k < mat.GetLength(2); ++k)
                    {
                        mat[i, j, k] = (float) rng.NextDouble();
                    }
                }
            }
        }

        [Test]
        public static void TestLSTM()
        {

            var rng = new Random(0);

            var mfr = new MatFileReader("../tests/AleaTKTest/data/lstm_small.mat");

            var inputSize = ((MLInt32)mfr.GetMLArray("InputSize")).Get(0);
            var seqLength = ((MLInt32)mfr.GetMLArray("SeqLength")).Get(0);
            var hiddenSize = ((MLInt32)mfr.GetMLArray("HiddenSize")).Get(0);
            var batchSize = ((MLInt32)mfr.GetMLArray("BatchSize")).Get(0);

            //var inputSize = 10;
            //var seqLength = 5;
            ////var seqLength = 1;
            //var hiddenSize = 4;
            //var batchSize = 5;

            //var inputSize = 5;
            //var seqLength = 3;
            //var hiddenSize = 4;
            //var batchSize = 2;

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var lstm = new LSTM<float>(x, hiddenSize);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, lstm.Y);

            exe.Initalize();

            //var input = new float[seqLength, batchSize, inputSize];
            //RandomMat(input, rng);

            var input_ = /*[shape  (3, 2, 5) ]*/ new []{ -0.49803245069230, 1.92953205381699, 0.94942080692576, 0.08755124138519, -1.22543551883017, 0.84436297640155, -1.00021534738956, -1.54477109677761, 1.18802979235230, 0.31694261192485, 0.92085882378082, 0.31872765294302, 0.85683061190269, -0.65102559330015, -1.03424284178446, 0.68159451828163, -0.80340966417384, -0.68954977775020, -0.45553250351734, 0.01747915902506, -0.35399391125348, -1.37495129341802, -0.64361840283289, -2.22340315222443, 0.62523145102719, -1.60205765560675, -1.10438333942845, 0.05216507926097, -0.73956299639131, 1.54301459540674};
            var input = input_.Select(n => (float) n).ToArray();


            Context.CpuContext.Eval(input.AsTensor().Reshape(seqLength*batchSize, inputSize)).Print();

            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            var WLSTM_ = /*[shape  (10, 16) ]*/new []{ 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 3.00000000000000, 3.00000000000000, 3.00000000000000, 3.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.49802635771920, -0.06838608792193, 0.10435590055030, -0.28469857976724, -0.85099660527803, 0.21787286514679, 0.28814539961984, -0.24738834013548, 0.75658487466254, -0.48478855819959, 0.01525283910048, -0.06239461667528, 0.51092640478615, 0.48978625663343, 0.05164914189897, 0.12605417320072, -0.29592858254337, -0.66026548940798, -0.11597071644205, 0.05211632303466, 0.41009689357591, 0.40079328292814, -0.12910893913598, -0.10076758352511, -0.34951765502236, -0.47333931239299, -0.56875673020834, 0.65025846507726, -0.16988406058388, -0.14602476720373, -0.41759845334998, 0.25916345194397, -0.53796594918598, -0.07091342673799, -0.29848885373123, 0.12896749928642, -0.17026837918962, -0.39354406137414, -0.00939407611288, 0.14277729017681, 0.02217240746106, 0.10082396591326, -0.21144069789365, -0.12091372199571, -0.22415348259198, -0.11985105384685, -0.27104876068148, -0.57542753411056, 0.05914204741792, -0.13392697873609, -0.54339944898868, 0.15426075184192, -0.30243278812775, 0.01731513193205, 0.24303018739251, 0.04299430358580, 0.37980022818110, -0.41160860678455, 0.13411388039252, -0.22827003031344, -0.29026571639396, -0.19294988825481, -0.10385084404246, 0.01872178074325, -0.38838328026112, 0.30027549565140, 0.15522081324349, -0.51208122875907, 0.49608406459853, 0.63196305867686, 0.39292652371988, -0.05997494527078, -0.35691754050351, 0.35148390897705, -0.13439231565773, 0.40748169012748, 0.06942499269229, 0.32554634549457, 0.11878879905813, 0.23552438939732, 0.00350000690694, 0.59529016463528, 0.04230403090121, 0.13399645448157, 0.62771689901875, -0.44925302038082, -0.42349499949524, 0.32313223605267, -0.39104113503805, 0.64787372854976, -0.13787299358658, -0.24915160381359, 0.64098067549346, 0.49350493047814, 0.62251965347552, 0.30201488609180, -0.28707522835157, 0.63668831769968, -0.08933445698379, 0.26748546526547, 0.31575065592458, -0.05167003103028, 0.20469312344869, 0.30740222385551, 0.12547517705188, -0.36646693019473, 0.09941272473535, 0.44212863222901, -0.23152261991046, -0.04987818010922, -0.14505118390721, 0.61642124282645, 0.22409825233748, 0.13582061208037, -0.25663869148177, 0.17974973043061, -0.22477755355246, 0.01061018609145, -0.21194869279296, 0.22547776498217, 0.19219693887165, -0.06943291852600, 0.13200223755388, -0.36435383624350, -0.49708586423520, 0.14646390042151, 0.05555783179084, 0.21167714563070, 0.79438159162131, 0.31482649566347, -0.30427407514805, 0.37233876269862, -0.43863580350384, -0.15386153493824, -0.02274720177488, 0.57111424054979, -0.24825160734948, -0.27547951288634, -0.03281750814181, -0.22115942878737, 0.37554530736884, -0.35997716945447, -0.38248955080370, -0.14594001491481};
            var w = WLSTM_.Select(n => (float) n).ToArray();
            exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4*hiddenSize)));

            exe.Forward();

            var H = /*[shape  (3, 2, 4) ]*/new []{ -0.07717362560926, -0.03174267245757, -0.06279976850342, -0.17678396162162, 0.33282740121219, 0.13327322199755, 0.24218172865991, 0.08745033773443, -0.03499299243100, -0.02699850909830, -0.17013592099571, -0.22911622689362, 0.45732331843586, 0.38663963085803, 0.39674900870867, 0.15904178619328, 0.08773196176143, 0.36461912794732, 0.17173163133160, -0.24078595043418, 0.10308684567976, 0.67527555036585, 0.46219017627173, 0.25971241510063};
            H.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();


            ctx.Eval(exe.GetTensor(lstm.Y).Reshape(seqLength * batchSize, -1)).Print();

            ctx.ToGpuContext().Stream.Synchronize();
        }
    }
}
