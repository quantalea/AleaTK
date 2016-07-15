using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Reshape<T> : Differentiable
    { 
        public Reshape(Variable<T> input, PartialShape shape)
        {
            Util.EnsureTrue(input.Type != VariableType.Parameter);
            Shape = input.HasShape ? PartialShape.Reshape(input.Shape, shape) : shape;
            Input = input;
            Output = Variable<T>(Shape);
            AddInput(Input);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public PartialShape Shape { get; }

        public override void Forward(Executor executor)
        {
            if (executor.GetData(Input).Tensor.Layout.IsInnerChangeMostFullyPacked)
            {
                var tensor = executor.GetData(Input).Tensor;
                var shape = tensor.Layout.Shape.Reshape(Shape.AsArray);
                var layout = new Layout(shape);
                var newTensor = new Tensor(tensor.Device, tensor.Memory, layout, tensor.Ptr);
                executor.GetData(Output).SetTensor(newTensor);
            }
            else
            {
                throw new NotImplementedException("TODO: Assign a new tensor for different layout.");
            }
        }

        public override void Backward(Executor executor)
        {
            if (executor.GetData(Output).Gradient == null) return;

            if (executor.GetData(Output).Gradient.Layout.IsInnerChangeMostFullyPacked)
            {
                var gradient = executor.GetData(Output).Gradient;
                var shape = gradient.Layout.Shape.Reshape(executor.GetData(Input).Tensor.Layout.Shape.AsArray);
                var layout = new Layout(shape);
                var newGradient = new Tensor(gradient.Device, gradient.Memory, layout, gradient.Ptr);
                executor.GetData(Input).SetGradient(newGradient);
            }
            else
            {
                throw new NotImplementedException("TODO: Assign a new tensor for different layout.");
            }
        }
    }
}
