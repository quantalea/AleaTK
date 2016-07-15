using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Alea.Parallel;

namespace AleaTK
{
    public interface IValue
    {
        Device Device { get; }

        Layout Layout { get; }

        BufferMemory Memory { get; }

        bool IsRValue { get; }

        bool IsLValue { get; }

        IValue<T> Cast<T>();
    }

    public interface IValue<T> : IValue
    {
        ILValue<T> ToLValue();

        IRValue<T> ToRValue();
    }

    public interface IRValue<T> : IValue<T>
    {
        BufferReader<T> BufferReader { get; }
    }

    public interface ILValue<T> : IValue<T>
    {
        Buffer<T> Buffer { get; }
    }

    public static class OpCodes
    {
        public const string Unspecified = "Unspecified";
        public const string Tensor = "Tensor";
        public const string Scalar = "Scalar";
        public const string Map1 = "Map1";
        public const string Map2 = "Map2";
        public const string Reduce = "Reduce";
        public const string Transpose = "Transpose";
        public const string Dot = "Dot";
        public const string Add = "Add";
        public const string Sub = "Sub";
        public const string Mul = "Mul";
        public const string Div = "Div";
        public const string Neg = "Neg";
        public const string ReduceSum = "ReduceSum";
        public const string ReduceMax = "ReduceMax";
        public const string Random = "Random";
        public const string Exp = "Exp";
        public const string Log = "Log";
        public const string Sqrt = "Sqrt";
        public const string Max = "Max";
        public const string ReLUGrad = "ReLUGrad";
        public const string ReShape = "ReShape";
        public const string Tanh = "Tanh";
        public const string TakeGrad = "TakeGrad";
    }

    public abstract class Expr : Disposable
    {
        private readonly List<Expr> _operands = new List<Expr>();

        public string OpCode { get; protected set; } = OpCodes.Unspecified;

        public IEnumerable<Expr> Operands => _operands;

        protected void AddOperand(Expr operand)
        {
            _operands.Add(operand);
        }

        public abstract Type DataType { get; }

        public abstract Shape Shape { get; }

        public abstract void Prepare(Assignment assignment);

        public abstract void Execute(Assignment assignment);

        public Expr<T> CastExpr<T>()
        {
            Util.EnsureEqual(DataType, typeof(T));
            return (Expr<T>)this;
        }

        public static Expr operator +(Expr a, Expr b)
        {
            return Library.ExprRegistry.Create(OpCodes.Add, a.DataType, a, b);
        }

        public static Expr operator -(Expr a, Expr b)
        {
            return Library.ExprRegistry.Create(OpCodes.Sub, a.DataType, a, b);
        }

        public static Expr operator *(Expr a, Expr b)
        {
            return Library.ExprRegistry.Create(OpCodes.Mul, a.DataType, a, b);
        }
    }

    public abstract class Expr<T> : Expr
    {
        public override Type DataType => typeof(T);

        public static Expr<T> operator -(Expr<T> a)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Neg, a);
        }

        public static Expr<T> operator +(Expr<T> a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Add, a, b);
        }

        public static Expr<T> operator +(T a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Add, a.AsScalar(), b);
        }

        public static Expr<T> operator +(Expr<T> a, T b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Add, a, b.AsScalar());
        }

        public static Expr<T> operator -(Expr<T> a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Sub, a, b);
        }

        public static Expr<T> operator *(Expr<T> a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Mul, a, b);
        }

        public static Expr<T> operator *(T a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Mul, a.AsScalar(), b);
        }

        public static Expr<T> operator /(Expr<T> a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Div, a, b);
        }

        public static Expr<T> operator /(T a, Expr<T> b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Div, a.AsScalar(), b);
        }

        public static Expr<T> operator /(Expr<T> a, T b)
        {
            return Library.ExprRegistry.Create<T>(OpCodes.Div, a, b.AsScalar());
        }
    }

    // RExpr means, during execution, we only need to generate an rValue,
    // if the output is lValue, this base class will assign them. This 
    // hides lot of implementation of assignment for different layout.
    public abstract class RExpr<T> : Expr<T>
    {
        protected abstract IRValue<T> GenerateRValue(Assignment assignment);

        public override void Prepare(Assignment assignment)
        {
        }

        public override void Execute(Assignment assignment)
        {
            var rValue = GenerateRValue(assignment);
            if (rValue == null) throw new NotImplementedException();

            var execution = assignment.ExecutionOf(this);

            if (execution.SpecifiedOutput == null)
            {
                var needAssign =
                    execution.RequireOutputLValue ||
                    rValue.Layout == null ||
                    (execution.RequireLayoutInnerChangeMost && !rValue.Layout.IsInnerChangeMost);

                if (needAssign)
                {
                    var lValue = assignment.TempTensor<T>(Shape);
                    assignment.Assign(lValue, rValue);
                    execution.Output = lValue;
                }
                else
                {
                    execution.Output = rValue;
                }
            }
            else
            {
                var needAllocate =
                    (execution.RequireLayoutInnerChangeMost && !execution.SpecifiedOutput.Layout.IsInnerChangeMost);

                if (needAllocate)
                {
                    var lValue = assignment.TempTensor<T>(Shape);
                    assignment.Assign(lValue, rValue);
                    execution.Output = lValue;

                    if (assignment.Expr == this)
                    {
                        assignment.Assign(execution.SpecifiedOutput.Cast<T>().ToLValue(), lValue);
                    }
                }
                else
                {
                    assignment.Assign(execution.SpecifiedOutput.Cast<T>().ToLValue(), rValue);
                    execution.Output = execution.SpecifiedOutput;
                }
            }
        }
    }

    public abstract class LExpr<T> : Expr<T>
    {
        protected abstract bool Execute(Assignment assignment, ILValue<T> output);

        public override void Prepare(Assignment assignment)
        {
            assignment.RequireOutputLValue(this);
        }

        public override void Execute(Assignment assignment)
        {
            var execution = assignment.ExecutionOf(this);
            ILValue<T> lValue;

            if (execution.SpecifiedOutput == null)
            {
                lValue = assignment.TempTensor<T>(Shape);
            }
            else
            {
                var needAllocate =
                    // do allocate if it is not fully packed C order
                    !execution.SpecifiedOutput.Layout.IsInnerChangeMostFullyPacked ||
                    // shape not match, means there will be broadcasting
                    (Shape != null && !execution.SpecifiedOutput.Layout.Shape.SequenceEqual(Shape));

                lValue = needAllocate ? assignment.TempTensor<T>(Shape) : execution.SpecifiedOutput.Cast<T>().ToLValue();
            }

            if (!Execute(assignment, lValue)) throw new NotImplementedException();

            if (assignment.Expr == this && execution.SpecifiedOutput != null && execution.SpecifiedOutput != lValue)
            {
                assignment.Assign(execution.SpecifiedOutput.Cast<T>().ToLValue(), lValue.ToRValue());
            }

            execution.Output = lValue;
        }
    }

    public class Assignment : Disposable
    {
        public enum Status
        {
            Preparing = 0,
            Prepared
        }

        public sealed class Execution
        {
            #region Prepare Pass
            // cache flag for prepare pass
            public bool Prepared { get; set; } = false;

            // exprs which references the expr of this execution as operand
            public HashSet<Expr> References { get; } = new HashSet<Expr>();

            public void AddReference(Expr reference, int referenceOperandId)
            {
                References.Add(reference);
                // TODO: more graph properties
            }

            // does this expr been requested to have lvalue?
            // the default value is false, means we can create rvalue
            // but once this is set to true, you cannot change it back to false
            public bool RequireOutputLValue { get; set; } = false;

            public bool RequireLayoutInnerChangeMost { get; set; } = false;

            // specified output? usually for leaf expr, they set this as input
            // and for the output of this assignment.
            // this MUST be a lvalue
            public IValue SpecifiedOutput { get; set; } = null;

            public Task Task { get; set; } = null;
            #endregion

            #region Schedule Pass
            // cache flag for schedule pass
            public bool Scheduled { get; set; } = false;

            public IValue Output { get; set; } = null;
            #endregion
        }

        private readonly Dictionary<Expr, Execution> _executions = new Dictionary<Expr, Execution>();
        private Status _status = Status.Preparing;
        private readonly List<DeviceMemoryRepository.Receipt> _deviceMemoryReceipts = new List<DeviceMemoryRepository.Receipt>();

        public Assignment(Context context, IValue value, Expr expr)
        {
            Util.EnsureEqual(true, value.IsLValue, "Assignment should be LValue.");
            Context = context;
            Value = value;
            Expr = expr;
        }

        public Tensor<T> TempTensor<T>(Shape shape)
        {
            if (Context.Type == ContextType.Gpu)
            {
                var memoryRcpt = Context.Device.ToGpuDevice().MemoryRepository.Acquire<T>(shape.Length);
                _deviceMemoryReceipts.Add(memoryRcpt);
                var memory = memoryRcpt.Memory;
                var bufferMemory = new BufferMemory(memory, 0L, memory.LongLength);
                var buffer = new Buffer<T>(Context.Device, bufferMemory, new Layout(shape), memory.Ptr.Reinterpret<T>());
                var tensor = new Tensor<T>(buffer);
                return tensor;
            }

            return Context.Device.Allocate<T>(shape);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var rcpt in _deviceMemoryReceipts)
                {
                    rcpt.Dispose();
                }
                _deviceMemoryReceipts.Clear();
            }
            base.Dispose(disposing);
        }

        public Context Context { get; }

        public IValue Value { get; }

        public Expr Expr { get; }

        public Task Task { get; private set; }

        public void SpecifyOutput<T>(Expr expr, Tensor<T> output)
        {
            Util.EnsureEqual(Status.Preparing, _status);
            Util.EnsureEqual(Context.Device, output.Device, "Assignment should work on same device.");
            var execution = _executions[expr];
            execution.SpecifiedOutput = output;
        }

        private void SpecifyOutput(Expr expr, IValue output)
        {
            Util.EnsureEqual(Status.Preparing, _status);
            Util.EnsureEqual(Context.Device, output.Device);
            Util.EnsureEqual(true, output.IsLValue);
            var execution = _executions[expr];
            execution.SpecifiedOutput = output;
        }

        public void RequireOutputLValue(Expr expr)
        {
            Util.EnsureEqual(Status.Preparing, _status);
            _executions[expr].RequireOutputLValue = true;
        }

        public void RequireLayoutInnerChangeMost(Expr expr)
        {
            Util.EnsureEqual(Status.Preparing, _status);
            _executions[expr].RequireLayoutInnerChangeMost = true;
        }

        private void Prepare(Expr expr, Expr reference, int referenceOperandId)
        {
            Execution execution;
            var found = _executions.TryGetValue(expr, out execution);
            if (!found)
            {
                execution = new Execution();
                _executions.Add(expr, execution);
            }

            // update graph properties
            if (reference != null)
            {
                execution.AddReference(reference, referenceOperandId);
            }

            if (execution.Prepared) return;

            // depth first walk
            expr.Operands.Iter((operand, operandId) => Prepare(operand, expr, operandId));
            expr.Prepare(this);
            execution.Task = new Task(() => expr.Execute(this));
            execution.Prepared = true;
        }

        protected void Prepare()
        {
            Util.EnsureEqual(Status.Preparing, _status);
            Prepare(Expr, null, -1);
            SpecifyOutput(Expr, Value);
            Task = _executions[Expr].Task;
            _status = Status.Prepared;
        }

        public IValue<T> GetInput<T>(Expr<T> operand)
        {
            Util.EnsureEqual(Status.Prepared, _status);
            var execution = _executions[operand];
            execution.Task.Wait();
            return execution.Output.Cast<T>();
        }

        public Execution ExecutionOf(Expr expr)
        {
            return _executions[expr];
        }

        private void Schedule(Expr expr)
        {
            var execution = _executions[expr];

            if (execution.Scheduled) return;

            // depth first walk
            foreach (var operand in expr.Operands)
            {
                Schedule(operand);
            }

            // schedule
            // TODO:@RDE (Runtime Dependency Engine)
            execution.Task.Start();

            // set flag
            execution.Scheduled = true;
        }

        public void Schedule()
        {
            Util.EnsureEqual(Status.Prepared, _status);
            Schedule(Expr);
        }

        public Task Run()
        {
            Prepare();
            Schedule();
            var task = Task;
            task.Wait();
            Dispose();
            return task;
        }

        public static Task Run<T>(Context context, Tensor<T> tensor, Expr<T> expr)
        {
            return (new Assignment(context, tensor, expr)).Run();
        }

        public static Task Run(Context context, IValue tensor, Expr expr)
        {
            return (new Assignment(context, tensor, expr)).Run();
        }

        public bool AssignByFlat1<T>(ILValue<T> lValue, IRValue<T> rValue)
        {
            var length = lValue.Layout.Shape.Length;
            var read = rValue.BufferReader.GetFlatReader1(lValue.Layout.Shape);
            var write = lValue.Buffer.FlatWriter1;

            if (Context.Type == ContextType.Gpu)
            {
                var stream = Context.ToGpuContext().Stream;
                stream.For(0L, length, i => write(i, read(i)));
                return true;
            }

            if (Context.Type == ContextType.Cpu)
            {
                for (var i = 0L; i < length; ++i) write(i, read(i));
                return true;
            }

            return false;
        }

        public bool AssignByRank2<T>(ILValue<T> lValue, IRValue<T> rValue)
        {
            var rows = lValue.Layout.Shape[0];
            var cols = lValue.Layout.Shape[1];
            var read = rValue.BufferReader.GetReader2(lValue.Layout.Shape);
            var write = lValue.Buffer.Writer2;

            if (Context.Type == ContextType.Gpu)
            {
                var stream = Context.ToGpuContext().Stream;
                stream.For(0L, rows * cols, i =>
                {
                    var row = i / cols;
                    var col = i % cols;
                    write(row, col, read(row, col));
                });
                return true;
            }

            if (Context.Type == ContextType.Cpu)
            {
                for (var i = 0L; i < rows; ++i)
                {
                    for (var j = 0L; j < cols; ++j)
                    {
                        write(i, j, read(i, j));
                    }
                }
                return true;
            }

            return false;
        }

        public void Assign<T>(ILValue<T> lValue, IRValue<T> rValue)
        {
            if (lValue == rValue) return;
            Util.EnsureTrue(Shape.Broadcast(rValue.Layout.Shape, lValue.Layout.Shape).SequenceEqual(lValue.Layout.Shape));
            if (Layout.CanFullyUnitStrideMapping(rValue.Layout, lValue.Layout) && AssignByFlat1(lValue, rValue)) return;
            if (rValue.Layout.Rank == 2 && lValue.Layout.Rank == 2 && AssignByRank2(lValue, rValue)) return;
            throw new NotImplementedException();
        }
    }

    public delegate Expr ExprFactory(dynamic exprParam, Expr[] inputExprs);

    public class ExprRegistry
    {
        private struct Key
        {
            public string OpCode;
            public Type[] InputTypes;
            public Type OutputType;

            public override string ToString()
            {
                var inputs = string.Join(", ", InputTypes.Select(ty => ty.Name));
                return $"[{OpCode}({inputs}) => {OutputType.Name}]";
            }
        }

        private class KeyComparer : IEqualityComparer<Key>
        {
            public bool Equals(Key x, Key y)
            {
                return x.OpCode == y.OpCode &&
                       x.InputTypes.Length == y.InputTypes.Length &&
                       x.InputTypes.SequenceEqual(y.InputTypes) &&
                       x.OutputType == y.OutputType;
            }

            public int GetHashCode(Key key)
            {
                var hash = 19;
                hash = hash * 31 + key.OpCode.GetHashCode();
                hash = hash * 31 + key.InputTypes.Length.GetHashCode();
                hash = key.InputTypes.Aggregate(hash, (current, inputType) => current * 31 + inputType.GetHashCode());
                hash = hash * 31 + key.OutputType.GetHashCode();
                return hash;
            }
        }

        private readonly Dictionary<Key, ExprFactory> _registry =
            new Dictionary<Key, ExprFactory>(new KeyComparer());

        public Expr Create(string opCode, dynamic exprParam, Type outputType, params Expr[] inputExprs)
        {
            var key = new Key
            {
                OpCode = opCode,
                InputTypes = inputExprs.Select(x => x.DataType).ToArray(),
                OutputType = outputType
            };

            ExprFactory create;
            var found = _registry.TryGetValue(key, out create);
            if (!found) throw new InvalidOperationException($"{key} not found in registry!");
            var expr = create(exprParam, inputExprs);
            return expr;
        }

        public Expr Create(string opCode, Type outputType, params Expr[] inputExprs)
        {
            return Create(opCode, null, outputType, inputExprs);
        }

        public Expr<T> Create<T>(string opCode, dynamic exprParam, params Expr[] inputExprs)
        {
            return Create(opCode, exprParam, typeof(T), inputExprs).CastExpr<T>();
        }

        public Expr<T> Create<T>(string opCode, params Expr[] inputExprs)
        {
            return Create(opCode, typeof(T), inputExprs).CastExpr<T>();
        }

        public void Register(ExprFactory factory, string opCode, Type outputType, params Type[] inputTypes)
        {
            var key = new Key
            {
                OpCode = opCode,
                InputTypes = inputTypes,
                OutputType = outputType
            };
            _registry.Add(key, factory);
        }

        public void Register<T>(ExprFactory factory, string opCode, params Type[] inputTypes)
        {
            Register(factory, opCode, typeof(T), inputTypes);
        }
    }
}
