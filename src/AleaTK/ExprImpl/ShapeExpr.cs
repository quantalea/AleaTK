using System.Linq;

namespace AleaTK.ExprImpl
{
    public class ReShapeExpr<T> : RExpr<T>
    {
        public ReShapeExpr(Expr<T> input, long[] dims, string opCode = OpCodes.ReShape)
        {
            // -1 means calc the shape, but only one -1 allowed.
            var numNegOne = dims.Select(x => x < 0 ? 1 : 0).Sum();
            Util.EnsureTrue(numNegOne == 0 || numNegOne == 1);

            if (numNegOne == 0)
            {
                var shape = new Shape(dims);
                // length must match old one
                Util.EnsureEqual(input.Shape.Length, shape.Length);
                Shape = shape;
            }
            else
            {
                var remainLength = dims.Select(x => x >= 0 ? x : 1L).Aggregate(ScalarOps.Mul);
                for (var i = 0; i < dims.Length; ++i)
                {
                    if (dims[i] < 0)
                    {
                        dims[i] = input.Shape.Length/remainLength;
                        break;
                    }
                }
                // check if it is multiply correct
                var shape = new Shape(dims);
                Util.EnsureEqual(input.Shape.Length, shape.Length);
                Shape = shape;
            }

            Input = input;
            AddOperand(Input);
        }

        public Expr<T> Input { get; }

        public override Shape Shape { get; }

        protected override IRValue<T> GenerateRValue(Assignment assignment)
        {
            var device = assignment.Context.Device;
            var input = assignment.GetInput(Input).ToRValue();
            var shape = Shape;

            if (input.Layout.IsInnerChangeMost && input.Layout.IsFullyUnitStride)
            {
                var read = input.BufferReader.GetFlatReader1();
                var layout = new Layout(shape);
                return new TensorReader<T>(device, layout, read);
            }

            return null;
        }
    }
}
