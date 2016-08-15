using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;

namespace AleaTKTest
{
    public static class GradientChecker
    {
        public static Func<float[], float[]> Evaluator(Executor executor, Variable<float> input, Variable<float> output)
        {
            return w =>
            {
                executor.AssignTensor(input, w.AsTensor());
                executor.Forward();
                var o = executor.GetTensor(output).Reshape(-1);
                return o.ToArray();
            };
        }

        public static Tensor<float> FiniteDifferenceGradient(Executor executor, Variable<float> input, Variable<float> output, float bump = 1e-5f)
        {
            var evaluator = Evaluator(executor, input, output);
            var inputTensor = executor.GetTensor(input);
            var outputGradientTensor = executor.GetGradient(output);
            var inputArray = inputTensor.Reshape(-1).ToArray();
            var outputGradientArray = outputGradientTensor.Reshape(-1).ToArray();
            var grad = AleaTKUtil.GradientChecker.FiniteDifferenceGradient(inputArray, outputGradientArray, evaluator, bump);
            var shape = inputTensor.Shape.AsArray;
            return grad.AsTensor().Reshape(shape);
        }
    }
}
