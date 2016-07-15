using System;
using System.Threading.Tasks;
using AleaTK.ML.Operator;

namespace AleaTK.ML
{
    public enum VariableType
    {
        Common = 0,
        Parameter,
        Auxilliary
    }

    public abstract class Variable : Symbol
    {
        private readonly PartialShape _shape;
        private readonly Expr _initializer;
        private Differentiable _owner;

        protected Variable(Type dataType, VariableType type)
        {
            DataType = dataType;
            Type = type;
            _shape = null;
            _initializer = null;
            _owner = null;
        }

        protected Variable(Type dataType, VariableType type, PartialShape shape)
        {
            DataType = dataType;
            Type = type;
            _shape = shape;
            _initializer = null;
            _owner = null;
        }

        protected Variable(Type dataType, VariableType type, Expr initializer)
        {
            DataType = dataType;
            Type = type;
            _shape = initializer != null ? new PartialShape(initializer.Shape.AsArray) : null;
            _initializer = initializer;
            _owner = null;
        }

        public Type DataType { get; }

        public VariableType Type { get; }

        public PartialShape Shape
        {
            get
            {
                Util.EnsureTrue(_shape != null, "This variable doesn't have shape.");
                return _shape;
            }
        }

        public Expr UntypedInitializer
        {
            get
            {
                Util.EnsureTrue(_initializer != null, "This variable doesn't have initializer.");
                return _initializer;
            }
        }

        public Differentiable Owner
        {
            get { return _owner; }
            set
            {
                Util.EnsureTrue(_owner == null, "Owner is already set.");
                _owner = value;
            }
        }

        public bool HasShape => _shape != null;

        public bool HasInitializer => _initializer != null;

        public bool HasOwner => _owner != null;

        public abstract IValue TensorToValue(Tensor blob);

        public abstract Expr TensorToExpr(Tensor blob);

        public abstract void GetOrAllocate(Device device, Layout layout, long length, ref Tensor blob);

        public abstract Task Initialize(Context ctx, ref Tensor blob);
    }

    public class Variable<T> : Variable
    {
        public Variable(VariableType type) : base(typeof (T), type)
        {
        }

        public Variable(VariableType type, PartialShape shape) : base(typeof (T), type, shape)
        {
        }

        public Variable(VariableType type, Expr<T> initalizer) : base(typeof (T), type, initalizer)
        {
        }

        public Expr<T> Initalizer => UntypedInitializer.CastExpr<T>();

        public override IValue TensorToValue(Tensor blob)
        {
            return blob.Cast<T>();
        }

        public override Expr TensorToExpr(Tensor blob)
        {
            return blob.Cast<T>();
        }

        private void VerifyShape(Shape targetShape)
        {
            if (!HasShape) return;

            var myShape = Shape;
            Util.EnsureEqual(myShape.Rank, targetShape.Rank, "Tensor shape doesn't match variable shape.");
            for (var i = 0; i < myShape.Rank; ++i)
            {
                if (myShape[i] >= 0)
                {
                    Util.EnsureEqual(myShape[i], targetShape[i], "Shape must match.");
                }
            }
        }

        public override void GetOrAllocate(Device device, Layout layout, long length, ref Tensor blob)
        {
            VerifyShape(layout.Shape);

            Tensor<T> tensor;

            if (blob == null)
            {
                tensor = device.Allocate<T>(layout, length);
                blob = tensor.ToTensor();
                return;
            }

            if (Layout.Match(blob.Layout, layout))
            {
                return;
            }

            if (blob.Memory.Memory.Length >= length)
            {
                var memory = new BufferMemory(blob.Memory.Memory, 0L, length);
                blob = new Tensor(device, memory, layout, memory.Memory.Handle);
                return;
            }

            tensor = device.Allocate<T>(layout, length);
            blob = tensor.ToTensor();
        }

        public override Task Initialize(Context ctx, ref Tensor blob)
        {
            if (!HasInitializer) return Task.Run(() => { });
            var shape = Initalizer.Shape;
            var layout = new Layout(shape);
            var length = layout.Shape.Length;
            GetOrAllocate(ctx.Device, layout, length, ref blob);
            var tensor = blob.Cast<T>();
            return ctx.Assign(tensor, Initalizer);
        }

        public static Variable<T> operator +(Variable<T> lhs, Variable<T> rhs)
        {
            var op = new Add<T>(lhs, rhs);
            return op.C;
        }
    }
}