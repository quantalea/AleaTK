using System;
using System.Linq;
using Alea;
using Alea.Parallel.Device;

namespace AleaTK.ML {
    public class Batcher: IDisposable {
        [GpuParam]
        private readonly int cols_, rows_, outputs_;
        private readonly bool doReset_;

        public long Index { get; private set; }
        public int Rows => rows_;
        public int Cols => cols_;
        public int Outputs => outputs_;

        public Batcher(Context context, float[,] data, float[,] labels, bool doReset=true) {
            doReset_ = doReset;
            Context = context;
            Random = new Random(0);
            rows_ = data.GetLength(0);
            cols_ = data.GetLength(1);
            outputs_ = labels.GetLength(1);

            Indices = Enumerable.Range(0, Rows).ToArray();
            Data = data;
            Labels = labels;

            IndicesTensor = context.Allocate(Indices);
            DataTensor1 = context.Allocate(data);
            LabelsTensor1 = context.Allocate(labels);
            DataTensor2 = context.Device.Allocate<float>(Shape.Create(Rows, Cols));
            LabelsTensor2 = context.Device.Allocate<float>(Shape.Create(Rows, Outputs));
            DataTensor = DataTensor1;
            LabelsTensor = LabelsTensor1;

            if (!doReset_) return;
            Index = -1;
            Reset();
        }

        #region props

        public Context Context { get; }
        public Random Random { get; }
        public int[] Indices { get; }
        public float[,] Data { get; }
        public float[,] Labels { get; }
        public Tensor<int> IndicesTensor { get; }
        public Tensor<float> DataTensor { get; private set; }
        public Tensor<float> DataTensor1 { get; }
        public Tensor<float> DataTensor2 { get; }
        public Tensor<float> LabelsTensor { get; private set; }
        public Tensor<float> LabelsTensor1 { get; }
        public Tensor<float> LabelsTensor2 { get; }

        #endregion

        private void ShuffleIndices() {
            var rng = Random;
            var array = Indices;
            var n = array.Length;
            while (n > 1) {
                var k = rng.Next(n--);
                var temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        public void Reset() {
            if (!doReset_ || Index == 0L || Context.Type != ContextType.Gpu) return;

            Index = 0L;
            ShuffleIndices();
            Context.Copy(IndicesTensor, Indices.AsTensor());
            var stream = Context.ToGpuContext().Stream;
            var srcData = DataTensor == DataTensor1 ? DataTensor1.Buffer.Ptr : DataTensor2.Buffer.Ptr;
            var dstData = DataTensor == DataTensor1 ? DataTensor2.Buffer.Ptr : DataTensor1.Buffer.Ptr;
            var srcLabels = LabelsTensor == LabelsTensor1 ? LabelsTensor1.Buffer.Ptr : LabelsTensor2.Buffer.Ptr;
            var dstLabels = LabelsTensor == LabelsTensor1 ? LabelsTensor2.Buffer.Ptr : LabelsTensor1.Buffer.Ptr;
            var idx = IndicesTensor.Buffer.Ptr;
            DeviceFor.For(stream, 0, Rows, i => {
                var j = idx[i];
                var srcDataOffset = srcData + i*cols_;
                var dstDataOffset = dstData + j* cols_;
                for (var k = 0; k < cols_; ++k) dstDataOffset[k] = srcDataOffset[k];
                var srcLabelsOffset = srcLabels + i*outputs_;
                var dstLabelsOffset = dstLabels + j* outputs_;
                for (var k = 0; k < outputs_; ++k) dstLabelsOffset[k] = srcLabelsOffset[k];
            });
            DataTensor = DataTensor == DataTensor1 ? DataTensor2 : DataTensor1;
            LabelsTensor = LabelsTensor == LabelsTensor1 ? LabelsTensor2 : LabelsTensor1;
        }

        public static Buffer<T> CreateBuffer<T>(Tensor<T> t, long rows, int cols, long idx) {
            return new Buffer<T>(t.Device, t.Memory, new Layout(Shape.Create(rows, cols)), t.Buffer.Ptr.LongPtr(idx * cols));
        }

        public bool Next(long batchSize, Executor executor, Variable<float> dataVar, Variable<float> labelsVar) {
            if (Index >= Rows) {
                Reset();
                return false;
            }
            var size = Index + batchSize >= Rows ? Rows - Index : batchSize;
            var dataBuffer = CreateBuffer(DataTensor, size, Cols, Index);
            var labelsBuffer = CreateBuffer(LabelsTensor, size, Outputs, Index);
            Index += batchSize;

            executor.SetTensor(dataVar, new Tensor<float>(dataBuffer));
            executor.SetTensor(labelsVar, new Tensor<float>(labelsBuffer));
            return true;
        }

        public void Dispose() {
            foreach (var t in new[] {DataTensor, DataTensor1, DataTensor2, LabelsTensor1, LabelsTensor2}) t.Buffer.Dispose();
            IndicesTensor.Buffer.Dispose();
        }
    }
}