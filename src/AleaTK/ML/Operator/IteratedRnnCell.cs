using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class IteratedRnnCell<T> : Differentiable
    {
        public RnnType RnnType { get; }
        public bool IsTraining { get; }
        public int InputSize { get; }
        public int BatchSize { get; }
        public int HiddenSize { get; }
        public int NumLayers { get; }
        public double DropoutProbability { get; }
        public ulong DropoutSeed { get; }

        public Variable<T> Input { get; }
        public Variable<T> Output { get; }
        public Variable<T> W { get; }
        public Variable<T> HX { get; }
        public Variable<T> CX { get; }
        public Variable<T> HY { get; }
        public Variable<T> CY { get; }
        public Variable<T> H { get; }
        public Variable<T> C { get; }
        public Variable<byte> ReserveSpace { get; }

        public readonly Symbol RnnCellDescr = new Symbol();

        public IteratedRnnCell(RnnType rnnRnnType, Variable<T> input, int numLayers, int hiddenSize, bool isTraining, double dropoutProbability, ulong dropoutSeed = 1337UL)
        {
            RnnType = rnnRnnType;
            IsTraining = isTraining;
            NumLayers = numLayers;
            HiddenSize = hiddenSize;
            DropoutProbability = isTraining ? dropoutProbability : 0.0;
            DropoutSeed = dropoutSeed;

            Util.EnsureEqual(3, input.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(input.Shape[1] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(input.Shape[2] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Input = input;
            BatchSize = (int)input.Shape[1];
            InputSize = (int)input.Shape[2];

            // output Shape (seqLength, batchSize, hiddenSize)
            Output = Variable<T>(PartialShape.Create(-1, BatchSize, HiddenSize));

            // W shape will be determined during initialization
            W = Parameter<T>();

            // create variables for input hidden and cell state
            HX =Variable<T>(PartialShape.Create(NumLayers, BatchSize, HiddenSize));
            CX =Variable<T>(PartialShape.Create(NumLayers, BatchSize, HiddenSize));
            HY =Variable<T>(PartialShape.Create(NumLayers, BatchSize, HiddenSize));
            CY =Variable<T>(PartialShape.Create(NumLayers, BatchSize, HiddenSize));

            // state variable H and Y = (n - 1, layer, b, d), n is unknown
            var shape = PartialShape.Create(-1, NumLayers, BatchSize, HiddenSize);
            H = Library.Variable<T>(shape);
            C = Library.Variable<T>(shape);

            ReserveSpace = Library.Variable<byte>();

            // construct the graph
            AddInput(Input);
            AddInput(W);
            AddOutput(Output);
            AddAuxVar(HX);
            AddAuxVar(CX);
            AddAuxVar(HY);
            AddAuxVar(CY);
            AddAuxVar(H);
            AddAuxVar(C);
            AddAuxVar(ReserveSpace);
        }

        public override void Initialize(Executor executor)
        {
            var cell = new RnnCell<T>(executor, RnnType, W, InputSize, BatchSize, HiddenSize, NumLayers, IsTraining, DropoutProbability, DropoutSeed);
            executor.Objects[RnnCellDescr] = cell;

            cell.Initialize(executor);

            base.Initialize(executor);
        }

        public void AssignInitialStates(Executor executor, Tensor<T> hx, Tensor<T> cx)
        {
            executor.AssignTensor(HX, hx);
            executor.AssignTensor(CX, cx);
        }

        public void AssignTerminalGradient(Executor executor, Tensor<T> dhy, Tensor<T> dcy)
        {
            executor.AssignGradientDirectly(HY, dhy);
            executor.AssignGradientDirectly(CY, dcy);
        }

        public void ZeroInitialStates(Executor executor)
        {
            executor.AssignTensor(HX, Fill(Shape.Create(HX.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
            executor.AssignTensor(CX, Fill(Shape.Create(CX.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
        }

        public void ZeroTerminalGradient(Executor executor)
        {
            executor.AssignGradientDirectly(HY, Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
            executor.AssignGradientDirectly(CY, Fill(Shape.Create(CY.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
        }

        public override void Forward(Executor executor)
        {
            var cell = (RnnCell<T>) executor.Objects[RnnCellDescr];

            // input should be allocated by the training procedure, so no need to give shape
            var input = executor.GetTensor(Input);
            var seqLength = input.Shape[0];

            // output is the output of this op, so we need give shape to allocate it
            var output = executor.GetTensor(Output, Shape.Create(seqLength, BatchSize, HiddenSize));

            // hx and cx is input, it must be assigned before running forward
            // hy and cy is output, we give shape to allocate them
            var hx = executor.GetTensor(HX);
            var cx = executor.GetTensor(CX);
            var hy = executor.GetTensor(HY, Shape.Create(HY.Shape.AsArray));
            var cy = executor.GetTensor(CY, Shape.Create(CY.Shape.AsArray));

            // h and c are for record the intermediate states h and c, it is of length seqLenght - 1
            // if n = 1, then no need for this
            var shape = seqLength == 1 ? Shape.Create(1) : Shape.Create(seqLength - 1, NumLayers, BatchSize, HiddenSize);
            var h = seqLength == 1 ? null : executor.GetTensor(H, shape);
            var c = seqLength == 1 ? null : executor.GetTensor(C, shape);

            // reserveSpace, according to doc of cuDNN, must be kept for each step
            var reserveSpace = executor.GetTensor(ReserveSpace, Shape.Create(seqLength, cell.ReserveSize));

            // iterate through the sequence one time step at a time
            for (var t = 0; t < seqLength; ++t)
            {
                cell.Input = input.Slice(t);
                cell.Output = output.Slice(t);
                cell.ReserveSpace = reserveSpace.Slice(t);

                if (t == 0)
                {
                    cell.HX = hx;
                    cell.CX = cx;
                }
                else
                {
                    cell.HX = h.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.CX = c.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                }

                if (t == seqLength - 1)
                {
                    cell.HY = hy;
                    cell.CY = cy;
                }
                else
                {
                    cell.HY = h.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.CY = c.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                }

                cell.Forward(executor);
            }
        }

        public override void Backward(Executor executor)
        {
            var cell = (RnnCell<T>)executor.Objects[RnnCellDescr];

            // input and output should be there, cause this is backward()
            var input = executor.GetTensor(Input);
            var output = executor.GetTensor(Output);

            // dOutput should be allocated by the child op, but dInput should be allocated by us
            var dInput = executor.GetGradient(Input, input.Shape);
            var dOutput = executor.GetGradient(Output);

            // dhy and dcy should be set before calling backward, while dhx and dcx is part of
            // our output, so we give shape to allocate them
            var dhy = executor.GetGradient(HY);
            var dcy = executor.GetGradient(CY);
            var dhx = executor.GetGradient(HX, Shape.Create(HX.Shape.AsArray));
            var dcx = executor.GetGradient(CX, Shape.Create(CX.Shape.AsArray));

            // hx and cx are input, hy and cy are output of forward, they are allocated, no need
            // to give shape to allocate
            var hx = executor.GetTensor(HX);
            var cx = executor.GetTensor(CX);
            var hy = executor.GetTensor(HY);
            var cy = executor.GetTensor(CY);

            // h and c are there by forward, no need to give shape to allocate
            // dh and dc we need allocate
            // TODO: actually dh and dc is not needed for each step, we can use swap to reduce memory cost
            var seqLength = (int)input.Shape[0];
            var shape = seqLength == 1 ? Shape.Create(1) : Shape.Create(seqLength - 1, NumLayers, BatchSize, HiddenSize);
            var h = seqLength == 1 ? null : executor.GetTensor(H);
            var c = seqLength == 1 ? null : executor.GetTensor(C);
            var dh = seqLength == 1 ? null : executor.GetGradient(H, shape);
            var dc = seqLength == 1 ? null : executor.GetGradient(C, shape);

            // reserveSpace should be calculated by forward
            var reserveSpace = executor.GetTensor(ReserveSpace);

            for (var t = seqLength - 1; t >= 0; --t)
            {
                cell.Input = input.Slice(t);
                cell.Output = output.Slice(t);
                cell.DOutput = dOutput.Slice(t);
                cell.DInput = dInput.Slice(t);
                cell.ReserveSpace = reserveSpace.Slice(t);

                if (t == seqLength - 1)
                {
                    cell.HY = hy;
                    cell.CY = cy;
                    cell.DHY = dhy;
                    cell.DCY = dcy;
                }
                else
                {
                    cell.HY = h.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.CY = c.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.DHY = dh.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.DCY = dc.Slice(t).Reshape(NumLayers, BatchSize, HiddenSize);
                }

                if (t == 0)
                {
                    cell.HX = hx;
                    cell.CX = cx;
                    cell.DHX = dhx;
                    cell.DCX = dcx;
                }
                else
                {
                    cell.HX = h.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.CX = c.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.DHX = dh.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                    cell.DCX = dc.Slice(t - 1).Reshape(NumLayers, BatchSize, HiddenSize);
                }

                cell.Backward(executor);
            }
        }
    }
}