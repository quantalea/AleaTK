using System;

namespace AleaTK.ExprImpl
{
    public class Map1Expr<TInput, TResult> : RExpr<TResult>
    {
        public Map1Expr(Expr<TInput> input, Func<TInput, TResult> transform, string opCode = OpCodes.Map1)
        {
            Shape = input.Shape;
            OpCode = opCode;
            Input = input;
            Transform = transform;
            AddOperand(Input);
        }

        public Expr<TInput> Input { get; }

        public Func<TInput, TResult> Transform { get; }

        public override Shape Shape { get; }

        protected override IRValue<TResult> GenerateRValue(Assignment assignment)
        {
            var device = assignment.Context.Device;
            var input = assignment.GetInput(Input).ToRValue();
            var transform = Transform;
            var layout = input.Layout;
            var inputRawReader = input.BufferReader.RawReader;
            Func<long, TResult> rawReader = i => transform(inputRawReader(i));
            return new TensorReader<TResult>(device, layout, rawReader);
        }
    }

    public class Map2Expr<TInput1, TInput2, TResult> : RExpr<TResult>
    {
        public Map2Expr(Expr<TInput1> input1, Expr<TInput2> input2, Func<TInput1, TInput2, TResult> transform, string opCode = OpCodes.Map2)
        {
            Input1 = input1;
            Input2 = input2;
            Transform = transform;
            OpCode = opCode;
            Shape = Shape.Broadcast(input1.Shape, input2.Shape);
            AddOperand(Input1);
            AddOperand(Input2);
        }

        public Expr<TInput1> Input1 { get; }

        public Expr<TInput2> Input2 { get; }

        public Func<TInput1, TInput2, TResult> Transform { get; }

        public override Shape Shape { get; }

        public override void Prepare(Assignment assignment)
        {
            assignment.RequireLayoutInnerChangeMost(Input1);
            assignment.RequireLayoutInnerChangeMost(Input2);
        }

        protected override IRValue<TResult> GenerateRValue(Assignment assignment)
        {
            var device = assignment.Context.Device;
            var input1 = assignment.GetInput(Input1).ToRValue();
            var input2 = assignment.GetInput(Input2).ToRValue();
            var transform = Transform;
            var shape = Shape;

            if (Layout.CanFullyUnitStrideMapping(input1.Layout, input2.Layout))
            {
                var read1 = input1.BufferReader.GetFlatReader1(shape);
                var read2 = input2.BufferReader.GetFlatReader1(shape);
                var layout = new Layout(shape);
                Func<long, TResult> rawReader = i => transform(read1(i), read2(i));
                return new TensorReader<TResult>(device, layout, rawReader);
            }

            return null;
        }
    }
}