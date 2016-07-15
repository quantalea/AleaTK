using System;

namespace AleaTK.ExprImpl
{
    public class ScalarExpr<T> : RExpr<T>
    {
        public ScalarExpr(Shape shape, T value)
        {
            OpCode = OpCodes.Scalar;
            Shape = shape;
            Value = value;
        }

        public ScalarExpr(T value)
        {
            OpCode = OpCodes.Scalar;
            Shape = Shape.Scalar;
            Value = value;
        }

        public T Value { get; }

        public override Shape Shape { get; }

        protected override IRValue<T> GenerateRValue(Assignment assignment)
        {
            var device = assignment.Context.Device;
            var layout = new Layout(Shape);
            var value = Value;
            Func<long, T> rawReader = _ => value;
            return new TensorReader<T>(device, layout, rawReader);
        }
    }
}