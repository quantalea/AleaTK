using System;
using System.Runtime.InteropServices;

namespace AleaTKUtil
{
    public static class GradientChecker
    {
        /// <summary>
        /// Jacobian of a vector valued function around value defined by variable calculated with finite differences.
        /// </summary>
        /// <param name="variable"></param>
        /// <param name="function"></param>
        /// <param name="jacobi"></param>
        /// <param name="bump"></param>
        public static void EvaluateJacobi(float[] variable, Func<float[], float[]> function, out float[,] jacobi, float bump = 1e-5f)
        {
            var f0 = function(variable);

            var inputDim = variable.GetLength(0);
            var outputDim = f0.GetLength(0);
            jacobi = new float[outputDim, inputDim];

            for (var i = 0; i < inputDim; ++i)
            {
                var temp = variable[i];
                variable[i] += bump;

                var f = function(variable);

                for (var j = 0; j < outputDim; ++j)
                {
                    jacobi[j, i] = (f[j] - f0[j])/bump;
                }

                variable[i] = temp;
            }
        }

        public static float[] FiniteDifferenceGradient(float[] variable, float[] outputGrad, Func<float[], float[]> function, float bump = 1e-5f)
        {
            float[,] jacobi = null;
            EvaluateJacobi(variable, function, out jacobi, bump);

            var inputDim = variable.GetLength(0);
            var outputDim = jacobi.GetLength(0);

            if (outputDim != outputGrad.GetLength(0)) throw new Exception("Wrong size");

            var grad = Common.Dot(outputGrad, jacobi);
            return grad;
        }
    }
}
