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
        public static void EvaluateJacobi(double[] variable, Func<double[], double[]> function, out double[,] jacobi, double bump = 1e-5)
        {
            var f0 = function(variable);

            var inputDim = variable.Length;
            var outputDim = f0.Length;
            jacobi = new double[outputDim, inputDim];

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

        public static double[] FiniteDifferenceGradient(double[] variable, double[] outputGrad, Func<double[], double[]> function, double bump = 1e-5)
        {
            double[,] jacobi = null;
            EvaluateJacobi(variable, function, out jacobi, bump);

            var inputDim = variable.Length;
            var outputDim = jacobi.GetLength(0);

            if (outputDim != outputGrad.GetLength(0)) throw new Exception("Wrong size");

            var grad = Common.Dot(outputGrad, jacobi);
            return grad;
        }
    }
}
