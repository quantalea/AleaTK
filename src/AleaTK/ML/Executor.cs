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

        public Data(Context context, Variable variable)
        {
            Context = context;
            Variable = variable;
        }

        public Context Context { get; }

        public Variable Variable { get; }

        public int GradientAggregationCounter { get; set; } = 0;

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

        #region Storage
        public Dictionary<Symbol, object> Objects { get; } = new Dictionary<Symbol, object>();

        public Dictionary<Symbol, IDisposable> Disposables { get; } = new Dictionary<Symbol, IDisposable>();
        #endregion

        #region Properties
        public Context Context { get; }

        public Variable Output { get; }

        public bool AssignAllGradient { get; set; } = false;

        public IEnumerable<Differentiable> ForwardOrder => _forwardOrder;

        public IEnumerable<Differentiable> BackwardOrder => _backwardOrder;

        public IEnumerable<Data> Data => _data.Values;

        public Data GetData(Variable var)
        {
            return _data[var];
        }
        #endregion

        public Executor(Context ctx, Variable output)
        {
            Context = ctx;
            Output = output;
            AddData(output);
            SetForwardOrder(output);
            _backwardOrder = ((IEnumerable<Differentiable>)_forwardOrder).Reverse().ToList();
        }

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

        #region Get/Set variable tensor
        /// <summary>
        /// Get variable tensor. The variable tensor is assumed to be allocated already,
        /// if not, an exception will be thrown. This is usually used for getting input
        /// variable tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetTensor<T>(Variable<T> variable)
        {
            var data = _data[variable];
            return data.Tensor.Cast<T>();
        }

        /// <summary>
        /// Get variable tesnor. If the variable tensor is not allocated or it is
        /// allocated but not large enough for holding the shape, then a new allocation
        /// will be triggered. This is usually used for output variable tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<T> GetTensor<T>(Variable<T> variable, Shape shape)
        {
            var layout = new Layout(shape);
            var data = _data[variable];
            return data.GetOrAllocateTensor(layout, shape.Length).Cast<T>();
        }

        /// <summary>
        /// Set variable tensor to an exists tensor (referencing tensor). The tensor must 
        /// be allocated in the same device of this executor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="tensor"></param>
        public void SetTensor<T>(Variable<T> variable, Tensor<T> tensor)
        {
            Util.EnsureTrue(tensor.Device == Context.Device, "Set gradient is reference, must be in same device.");
            var data = _data[variable];
            data.SetTensor(tensor.ToTensor());
        }

        /// <summary>
        /// Assign variable tensor from another tensor (copy may happen if the src tensor 
        /// is not in the same device of this executor).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="srcTensor"></param>
        /// <returns></returns>
        public Task AssignTensor<T>(Variable<T> variable, Tensor<T> srcTensor)
        {
            var data = _data[variable];
            var blob = data.GetOrAllocateTensor(srcTensor.Layout, srcTensor.Memory.Length);
            var dstTensor = blob.Cast<T>();
            return Context.Copy(dstTensor, srcTensor);
        }

        /// <summary>
        /// Assign variable tensor from an expression.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
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
        #endregion

        #region Get/Set variable gradient
        /// <summary>
        /// Get variable gradient. The variable gradient is assumed to be allocated already,
        /// if not, an exception will be thrown. This is usually used for getting output
        /// variable gradient.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetGradient<T>(Variable<T> variable)
        {
            var data = _data[variable];
            return data.Gradient.Cast<T>();
        }

        /// <summary>
        /// Get variable gradient together with its aggregation counter. The variable gradient
        /// is assumed to be allocated already, if not, an exception will be thrown. This is
        /// usually used for getting output variable gradient.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="aggregationCounter"></param>
        /// <returns></returns>
        public Tensor<T> GetGradient<T>(Variable<T> variable, out int aggregationCounter)
        {
            var data = _data[variable];
            aggregationCounter = data.GradientAggregationCounter;
            return data.Gradient.Cast<T>();
        }

        /// <summary>
        /// Get variable gradient. If the variable gradient is not allocated or it is
        /// allocated but not large enough for holding the shape, then a new allocation
        /// will be triggered. This is usually used for input variable gradient.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<T> GetGradient<T>(Variable<T> variable, Shape shape)
        {
            var layout = new Layout(shape);
            var data = _data[variable];
            return data.GetOrAllocateGradient(layout, shape.Length).Cast<T>();
        }

        /// <summary>
        /// Get variable gradient together with its aggregation counter. If the variable gradient
        /// is not allocated or it is allocated but not large enough for holding the shape, then
        /// a new allocation will be triggered. This is usually used for input variable gradient.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="shape"></param>
        /// <param name="aggregationCounter"></param>
        /// <returns></returns>
        public Tensor<T> GetGradient<T>(Variable<T> variable, Shape shape, out int aggregationCounter)
        {
            var layout = new Layout(shape);
            var data = _data[variable];
            aggregationCounter = data.GradientAggregationCounter;
            return data.GetOrAllocateGradient(layout, shape.Length).Cast<T>();
        }

        /// <summary>
        /// Set variable gradient to an exists tensor (referencing tensor).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="gradient"></param>
        /// <param name="counter"></param>
        public void SetGradient<T>(Variable<T> variable, Tensor<T> gradient, int counter = 1)
        {
            Util.EnsureTrue(gradient.Device == Context.Device, "Set gradient is reference, must be in same device.");
            var data = _data[variable];
            data.GradientAggregationCounter = counter;
            data.SetGradient(gradient.ToTensor());
        }

        /// <summary>
        /// Assign variable gradient from an expression. If replace is false, then the previouse
        /// gradient will be add to current gradient.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="expr"></param>
        /// <param name="aggregateCounter"></param>
        /// <param name="replace"></param>
        /// <returns></returns>
        public Task AssignGradient<T>(Variable<T> variable, Expr<T> expr, int aggregateCounter = 1, bool replace = false)
        {
            if (!replace && !AssignAllGradient && !variable.HasOwner && variable.Type != VariableType.Parameter) return Task.Run(() => { });

            var data = _data[variable];
            if (replace)
            {
                var shape = data.Tensor.Layout.Shape;
                var layout = new Layout(shape);
                var length = layout.Shape.Length;
                var blob = data.GetOrAllocateGradient(layout, length);
                var tensor = blob.Cast<T>();
                data.GradientAggregationCounter = aggregateCounter;
                return Context.Assign(tensor, expr);
            }
            else
            {
                var currentAggregationCounter = data.GradientAggregationCounter;
                data.GradientAggregationCounter += aggregateCounter;
                if (currentAggregationCounter <= 0)
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
        }

        /// <summary>
        /// Assign variable gradient by a tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <param name="tensor"></param>
        /// <param name="aggregateCounter"></param>
        /// <param name="replace"></param>
        /// <returns></returns>
        public Task AssignGradient<T>(Variable<T> variable, Tensor<T> tensor, int aggregateCounter = 1, bool replace = false)
        {
            if (!replace && !AssignAllGradient && !variable.HasOwner && variable.Type != VariableType.Parameter) return Task.Run(() => { });

            if (Context.Device == tensor.Device)
            {
                return AssignGradient(variable, (Expr<T>) tensor, aggregateCounter, replace);
            }
            else if (replace)
            {
                var data = _data[variable];
                var blob = data.GetOrAllocateGradient(tensor.Layout, tensor.Memory.Length);
                var dstTensor = blob.Cast<T>();
                data.GradientAggregationCounter = aggregateCounter;
                return Context.Copy(dstTensor, tensor);
            }
            else
            {
                throw new NotImplementedException();
            }
        }
        #endregion

        public int GetGradientAggregationCounter(Variable var)
        {
            var data = _data[var];
            return data.GradientAggregationCounter;
        }

        public void SetGradientAggregationCounter(Variable var, int aggregationCounter)
        {
            var data = _data[var];
            data.GradientAggregationCounter = aggregationCounter;
        }

        public int IncreaseGradientAggregationCounter(Variable var, int increasing = 1)
        {
            var data = _data[var];
            var oldCounter = data.GradientAggregationCounter;
            data.GradientAggregationCounter += increasing;
            return oldCounter;
        }

        public void ClearGradientAggregationCounters()
        {
            foreach (var data in Data)
            {
                data.GradientAggregationCounter = 0;
            }
        }

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

        public void Forward()
        {
            foreach (var op in ForwardOrder)
            {
                op.Forward(this);
            }
        }

        public void Backward(bool clearGradientAggretionCounter = true)
        {
            if (clearGradientAggretionCounter)
            {
                foreach (var data in Data)
                {
                    data.GradientAggregationCounter = 0;
                }
            }

            foreach (var op in BackwardOrder)
            {
                op.Backward(this);
            }
        }
    }
}