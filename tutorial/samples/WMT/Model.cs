using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using Library = AleaTK.ML.Library;

namespace Tutorial.Samples
{
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

            EncoderInputs = Library.Variable<int>(PartialShape.Create(numInputSteps, cfg.BatchSize));
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
}