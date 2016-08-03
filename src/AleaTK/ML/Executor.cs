using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Alea.cuDNN;

namespace AleaTK.ML
{
    public sealed class Data
    {
        private Tensor _tensor = null;
        private Tensor _gradient = null;
        private int _gradientAggregationCounter = 0;

        public Data(Context context, Variable variable)
        {
            Context = context;
            Variable = variable;
        }

        public Context Context { get; }

        public Variable Variable { get; }

        public void ResetGradientAggregationCounter()
        {
            _gradientAggregationCounter = 0;
        }

        public int GradientAggregationCounter
        {
            get
            {
                _gradientAggregationCounter++;
                return _gradientAggregationCounter - 1;
            }
        }

        public void Initialize()
        {
            if (Variable.HasInitializer)
            {
                Variable.Initialize(Context, ref _tensor);
            }
        }

        public Tensor Tensor => _tensor;

        public Tensor Gradient => _gradient;

        public IValue TensorAsValue => Variable.TensorToValue(_tensor);

        public IValue GradientAsValue => Variable.TensorToValue(_gradient);

        public Expr TensorAsExpr => Variable.TensorToExpr(_tensor);

        public Expr GradientAsExpr => Variable.TensorToExpr(_gradient);

        public Tensor GetOrAllocateTensor(Layout layout, long length)
        {
            Variable.GetOrAllocate(Context.Device, layout, length, ref _tensor);
            return _tensor;
        }

        public Tensor GetOrAllocateGradient(Layout layout, long length)
        {
            Variable.GetOrAllocate(Context.Device, layout, length, ref _gradient);
            return _gradient;
        }

        public void SetTensor(Tensor tensor)
        {
            _tensor = tensor;
        }

        public void SetGradient(Tensor gradient)
        {
            _gradient = gradient;
        }
    }

    public class Executor : Disposable
    {
        private readonly Dictionary<Variable, Data> _data = new Dictionary<Variable, Data>();
        private readonly List<Differentiable> _forwardOrder = new List<Differentiable>();
        private readonly List<Differentiable> _backwardOrder;

        #region Native repo
        public DisposableRepository<TensorDescriptor> TensorDescRepo { get; } =
            new DisposableRepository<TensorDescriptor>(() => new TensorDescriptor());

        public DisposableRepository<FilterDescriptor> FilterDescRepo { get; } =
            new DisposableRepository<FilterDescriptor>(() => new FilterDescriptor());

        public DisposableDictionary<Symbol, TensorDescriptor> TensorDescDict { get; } =
            new DisposableDictionary<Symbol, TensorDescriptor>(_ => new TensorDescriptor());

        public DisposableDictionary<Symbol, FilterDescriptor> FilterDescDict { get; } =
            new DisposableDictionary<Symbol, FilterDescriptor>(_ => new FilterDescriptor());

        public DisposableDictionary<Symbol, DropoutDescriptor> DropoutDescDict { get; } =
            new DisposableDictionary<Symbol, DropoutDescriptor>(_ => new DropoutDescriptor()); 

        public DisposableDictionary<Symbol, RNNDescriptor> RnnDescDict { get; } =
            new DisposableDictionary<Symbol, RNNDescriptor>(_ => new RNNDescriptor());
        #endregion

        public Dictionary<Symbol, object> Objects { get; set; }

        public Dictionary<Symbol, IDisposable> Dispobales { get; set; }

        public Executor(Context ctx, Variable loss)
        {
            Context = ctx;
            AddData(loss);
            SetForwardOrder(loss);
            _backwardOrder = ((IEnumerable<Differentiable>) _forwardOrder).Reverse().ToList();
        }

        public Context Context { get; }

        #region Set order
        private void SetForwardOrder(Variable variable)
        {
            SetForwardOrder(variable, new HashSet<Differentiable>());
        }

        private void SetForwardOrder(Variable variable, HashSet<Differentiable> cache)
        {
            if (variable.HasOwner)
            {
                SetForwardOrder(variable.Owner, cache);
            }
        }

        private void SetForwardOrder(Differentiable op, HashSet<Differentiable> cache)
        {
            if (cache.Contains(op)) return;

            foreach (var input in op.Inputs)
            {
                SetForwardOrder(input, cache);
            }

            _forwardOrder.Add(op);
            cache.Add(op);
        }
        #endregion

        #region Add data
        private void AddData(Variable variable)
        {
            AddData(variable, new HashSet<Differentiable>());
        }

        private void AddData(Variable variable, HashSet<Differentiable> cache)
        {
            if (_data.ContainsKey(variable)) return;

            if (variable.HasOwner)
            {
                AddData(variable.Owner, cache);
            }
            else
            {
                _data.Add(variable, new Data(Context, variable));
            }
        }

        private void AddData(Differentiable op, HashSet<Differentiable> cache)
        {
            if (cache.Contains(op)) return;

            foreach (var input in op.Inputs)
            {
                AddData(input, cache);
            }

            foreach (var auxVar in op.AuxVars)
            {
                AddData(auxVar, cache);
            }

            foreach (var output in op.Outputs)
            {
                _data.Add(output, new Data(Context, output));
            }

            cache.Add(op);
        }
        #endregion

        public IEnumerable<Differentiable> ForwardOrder => _forwardOrder;

        public IEnumerable<Differentiable> BackwardOrder => _backwardOrder;

        public IEnumerable<Data> Data => _data.Values; 

        public virtual void Initalize()
        {
            // initialize variables which has inititlizers.
            foreach (var data in Data)
            {
                data.Initialize();
            }

            // call operator's init.
            foreach (var op in ForwardOrder)
            {
                op.Initialize(this);
            }
        }

        public void SetTensor<T>(Variable<T> variable, Tensor<T> tensor)
        {
            Util.EnsureTrue(tensor.Device == Context.Device, "Set tensor is reference, must be in same device.");
            var data = _data[variable];
            data.SetTensor(tensor.ToTensor());
        }

        public void SetGradient<T>(Variable<T> variable, Tensor<T> gradient)
        {
            Util.EnsureTrue(gradient.Device == Context.Device, "Set gradient is reference, must be in same device.");
            var data = _data[variable];
            data.SetGradient(gradient.ToTensor());
        }

        public Tensor<T> GetTensor<T>(Variable<T> variable, Shape shape)
        {
            var layout = new Layout(shape);
            var data = _data[variable];
            return data.GetOrAllocateTensor(layout, shape.Length).Cast<T>();
        }

        public Tensor<T> GetGradient<T>(Variable<T> variable, Shape shape)
        {
            var layout = new Layout(shape);
            var data = _data[variable];
            return data.GetOrAllocateGradient(layout, shape.Length).Cast<T>();
        }

        public Data GetData(Variable var)
        {
            return _data[var];
        }

        public Tensor<T> GetTensor<T>(Variable<T> variable)
        {
            var data = _data[variable];
            return data.Tensor.Cast<T>();
        }

        public Tensor<T> GetGradient<T>(Variable<T> variable)
        {
            var data = _data[variable];
            return data.Gradient.Cast<T>();
        }

        public Task AssignTensor<T>(Variable<T> variable, Tensor<T> srcTensor)
        {
            var data = _data[variable];
            var blob = data.GetOrAllocateTensor(srcTensor.Layout, srcTensor.Memory.Length);
            var dstTensor = blob.Cast<T>();
            return Context.Copy(dstTensor, srcTensor);
        }

        public Task AssignTensor<T>(Variable<T> variable, Expr<T> expr)
        {
            var shape = expr.Shape;
            var layout = new Layout(shape);
            var length = layout.Shape.Length;
            var data = _data[variable];
            var blob = data.GetOrAllocateTensor(layout, length);
            var tensor = blob.Cast<T>();
            return Context.Assign(tensor, expr);
        }

        public Task AssignGradient<T>(Variable<T> variable, Expr<T> expr)
        {
            if (!variable.HasOwner && variable.Type != VariableType.Parameter) return Task.Run(() => { });

            var data = _data[variable];
            var counter = data.GradientAggregationCounter;
            if (counter == 0)
            {
                var shape = data.Tensor.Layout.Shape;
                var layout = new Layout(shape);
                var length = layout.Shape.Length;
                var blob = data.GetOrAllocateGradient(layout, length);
                var tensor = blob.Cast<T>();
                return Context.Assign(tensor, expr);
            }
            else
            {
                var grad = data.GradientAsExpr;
                return Context.Assign(data.GradientAsValue, grad + expr);
            }
        }

        public Task AssignGradientDirectly<T>(Variable<T> variable, Tensor<T> srcTensor)
        {
            //if (!variable.HasOwner && variable.Type != VariableType.Parameter) return Task.Run(() => { });

            var data = _data[variable];
            var blob = data.GetOrAllocateGradient(srcTensor.Layout, srcTensor.Memory.Length);
            var dstTensor = blob.Cast<T>();
            return Context.Copy(dstTensor, srcTensor);
        }

        public Task AssignGradientDirectly<T>(Variable<T> variable, Expr<T> expr)
        {
            //if (!variable.HasOwner && variable.Type != VariableType.Parameter) return Task.Run(() => { });

            var data = _data[variable];
            var shape = data.Tensor.Layout.Shape;
            var layout = new Layout(shape);
            var length = layout.Shape.Length;
            var blob = data.GetOrAllocateGradient(layout, length);
            var tensor = blob.Cast<T>();
            return Context.Assign(tensor, expr);
        }

        public void Forward()
        {
            foreach (var op in ForwardOrder)
            {
                op.Forward(this);
            }
        }

        public void Backward()
        {
            foreach (var data in Data)
            {
                data.ResetGradientAggregationCounter();
            }

            foreach (var op in BackwardOrder)
            {
                op.Backward(this);
            }
        }
    }
}