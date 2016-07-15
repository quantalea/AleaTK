using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AleaTK.ML.Operator;

namespace AleaTK.ML
{
    public static class Library
    {
        public static Variable<T> Variable<T>()
        {
            return new Variable<T>(VariableType.Common);
        }

        public static Variable<T> Variable<T>(PartialShape shape)
        {
            return new Variable<T>(VariableType.Common, shape);
        }

        public static Variable<T> Parameter<T>(Expr<T> initializer)
        {
            return new Variable<T>(VariableType.Parameter, initializer);
        }

        public static Variable<T> Parameter<T>()
        {
            return new Variable<T>(VariableType.Parameter);
        }

        public static Variable<T> AuxVariable<T>()
        {
            return new Variable<T>(VariableType.Auxilliary);
        }

        public static Variable<T> Dot<T>(Variable<T> a, Variable<T> b)
        {
            var op = new Dot<T>(a, b);
            return op.C;
        }

        public static Variable<T> L2Loss<T>(Variable<T> pred, Variable<T> label)
        {
            var op = new L2Loss<T>(pred, label);
            return op.Loss;
        }

        public static Variable<T> Reshape<T>(this Variable<T> input, params long[] shape)
        {
            return new Reshape<T>(input, PartialShape.Create(shape)).Output;
        }
    }
}
