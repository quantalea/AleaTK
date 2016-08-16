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
        //public static Func<float[], float[]> Evaluator(Executor executor, Variable<float> input, Variable<float> output)
        //{
        //    return w =>
        //    {
        //        var inputTensor = executor.GetTensor(input);
        //        executor.AssignTensor(input, w.AsTensor(inputTensor.Shape));
        //        executor.Forward();
        //        var o = executor.GetTensor(output).Reshape(-1);
        //        return o.ToArray();
        //    };
        //}

        //public static Tensor<float> FiniteDifferenceGradient(Executor executor, Variable<float> input, Variable<float> output, float bump = 1e-5f)
        //{
        //    var evaluator = Evaluator(executor, input, output);
        //    var inputTensor = executor.GetTensor(input);
        //    var outputGradientTensor = executor.GetGradient(output);
        //    var inputArray = inputTensor.Reshape(-1).ToArray();
        //    var outputGradientArray = outputGradientTensor.Reshape(-1).ToArray();
        //    var grad = AleaTKUtil.GradientChecker.FiniteDifferenceGradient(inputArray, outputGradientArray, evaluator, bump);
        //    var shape = inputTensor.Shape.AsArray;
        //    return grad.AsTensor().Reshape(shape);
        //}

        public static Tensor<float> FiniteDifferenceGradient(Executor executor, Variable<float> input, double bump = 1e-5f, Variable<float> output = null)
        {
            if (output == null)
            {
                output = (Variable<float>) executor.Output;
            }

            // first, backup the x
            var ctx = executor.Context;
            var inputTensor = executor.GetTensor(input);
            var inputShape = inputTensor.Shape;
            var inputTensorBackup = ctx.Device.Allocate<float>(inputShape);
            ctx.Assign(inputTensorBackup, inputTensor);

            // evaluator
            Func<double[], double[]> evaluator = inputBlob =>
            {
                var inputBlobSingle = inputBlob.Select(x => (float) x).ToArray();
                executor.AssignTensor(input, inputBlobSingle.AsTensor(inputShape));
                executor.Forward();
                var outputTensor = executor.GetTensor(output);
                return outputTensor.ToArray().Select(x => (double) x).ToArray();
            };

            var inputArray = inputTensor.ToArray().Select(x => (double)x).ToArray();
            var outputGradientArray = executor.GetGradient(output).ToArray().Select(x => (double)x).ToArray();
            var inputGradientArray = AleaTKUtil.GradientChecker.FiniteDifferenceGradient(inputArray, outputGradientArray, evaluator, bump).Select(x => (float)x).ToArray();
            var inputGradientTensor = inputGradientArray.AsTensor(inputShape);

            // now we need recover the data
            executor.AssignTensor(input, inputTensorBackup);
            executor.Forward();

            return inputGradientTensor;
        }
    }
}
