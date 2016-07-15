using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AleaTK.ExprImpl
{
    public class TakeExpr<T> : RExpr<T>
    {
        public TakeExpr(Expr<int> indices, Expr<T> source)
        {
            Indices = indices;
            Source = source;
            Util.EnsureTrue(Source.Shape.Rank == 2);
            Shape = Shape.Create(Indices.Shape.Concat(new[] {Source.Shape[1]}).ToArray());
            AddOperand(Indices);
            AddOperand(Source);
        }

        public Expr<int> Indices { get; }

        public Expr<T> Source { get; }

        public override Shape Shape { get; }

        protected override IRValue<T> GenerateRValue(Assignment assignment)
        {

            var device = assignment.Context.Device;
            var indices = assignment.GetInput(Indices).ToRValue();
            var source = assignment.GetInput(Source).ToRValue();
            Util.EnsureTrue(indices.Layout.IsInnerChangeMostFullyPacked);

            var indicesReader = indices.BufferReader.RawReader;
            var sourceReader = source.BufferReader.GetReader2();
            var layout = new Layout(Shape);
            var dim = source.Layout.Shape[1];
            Func<long, T> rawReader = i => sourceReader(indicesReader(i/dim), i%dim);
            return new TensorReader<T>(device, layout, rawReader);
        }
    }

    public class TakeGradExpr<T> : RExpr<T>
    {
        public TakeGradExpr(Expr<int> indices, Expr<T> outputGradient, int sourceRows, T zero, Func<T, T, T> add)
        {
            Indices = indices;
            OutputGradient = outputGradient;
            SourceRows = sourceRows;
            Zero = zero;
            Add = add;
            Shape = Shape.Create(SourceRows, OutputGradient.Shape[OutputGradient.Shape.Rank - 1]);
            AddOperand(Indices);
            AddOperand(OutputGradient);
        }

        public Expr<int> Indices { get; }

        public Expr<T> OutputGradient { get; }

        public int SourceRows { get; }

        public override Shape Shape { get; }

        public T Zero { get; }

        public Func<T, T, T> Add { get; } 

        protected override IRValue<T> GenerateRValue(Assignment assignment)
        {
            var device = assignment.Context.Device;
            var indices = assignment.GetInput(Indices).ToRValue();
            var dOutput = assignment.GetInput(OutputGradient).ToRValue();
            Util.EnsureTrue(indices.Layout.IsInnerChangeMostFullyPacked);
            Util.EnsureTrue(dOutput.Layout.IsInnerChangeMostFullyPacked);

            var indicesReader = indices.BufferReader.RawReader;
            var dOutputReader = dOutput.BufferReader.RawReader;
            var layout = new Layout(Shape);
            var sourceRows = SourceRows;
            var batchSize = Indices.Shape.Length;
            var zero = Zero;
            var add = Add;
            var dims = Shape[1];

            Func<long, T> rawReader = i =>
            {
                var row = i/ dims;
                var col = i% dims;
                var ret = zero;
                for (var j = 0; j < batchSize; ++j)
                {
                    var idx = indicesReader(j);
                    if (idx == row)
                    {
                        var value = dOutputReader(j*dims + col);
                        ret = add(ret, value);
                    }
                }
                return ret;
            };

            return new TensorReader<T>(device, layout, rawReader);
        }
    }
}
