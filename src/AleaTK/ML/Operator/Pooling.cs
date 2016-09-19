using System;
using Alea.cuDNN;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Pooling2D<T> : Differentiable, ILayer<T> {
        public Pooling2D(Variable<T> data, PoolingMode mode, int kernelH, int kernelW, int strideH, int strideW)
        {
            Descriptor = new PoolingDescriptor();
            Descriptor.Set2D(mode, NanPropagation.NOT_PROPAGATE_NAN, kernelH, kernelW, 0, 0, strideH, strideW);

            var dataType = Dnn.DataTypeOf<T>();
            var dataDesc = new TensorDescriptor();
            dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, 10, (int)data.Shape[1], (int)data.Shape[2], (int)data.Shape[3]);

            int n, c, h, w;
            Descriptor.Get2dForwardOutputDim(dataDesc, out n, out c, out h, out w);

            Data = data;
            Output = Variable<T>(PartialShape.Create(-1, c, h, w));

            AddInput(Data);
            AddOutput(Output);

            dataDesc.Dispose();
        }

        public PoolingDescriptor Descriptor { get; }

        public Variable<T> Data { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var output = executor.GetTensor(Output, Shape.Create(data.Shape[0], Output.Shape[1], Output.Shape[2], Output.Shape[3]));

            if (executor.Context.Type == ContextType.Gpu)
            {
                var dnn = executor.Context.ToGpuContext().Dnn;

                using (var dataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var outputDescRcpt = executor.TensorDescRepo.Acquire())
                {
                    var dataDesc = dataDescRcpt.Value;
                    var outputDesc = outputDescRcpt.Value;
                    var dataType = Dnn.DataTypeOf<T>();

                    dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)data.Shape[1], (int)data.Shape[2], (int)data.Shape[3]);
                    outputDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Output.Shape[1], (int)Output.Shape[2], (int)Output.Shape[3]);

                    dnn.PoolingForward(Descriptor, ScalarOps.Conv<T>(1.0), dataDesc, data.Buffer.Ptr, ScalarOps.Conv<T>(0.0), outputDesc, output.Buffer.Ptr);

                    return;
                }
            }

            throw new NotImplementedException();
        }

        public override void Backward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var output = executor.GetTensor(Output);
            var dOutput = executor.GetGradient(Output);
            var dData = executor.GetGradient(Data, Shape.Create(data.Shape.AsArray));

            if (executor.Context.Type == ContextType.Gpu)
            {
                var dnn = executor.Context.ToGpuContext().Dnn;

                using (var dataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var outputDescRcpt = executor.TensorDescRepo.Acquire())
                using (var dDataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var dOutputDescRcpt = executor.TensorDescRepo.Acquire())
                {
                    var dataDesc = dataDescRcpt.Value;
                    var outputDesc = outputDescRcpt.Value;
                    var dDataDesc = dDataDescRcpt.Value;
                    var dOutputDesc = dOutputDescRcpt.Value;
                    var dataType = Dnn.DataTypeOf<T>();

                    dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)data.Shape[1], (int)data.Shape[2], (int)data.Shape[3]);
                    outputDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Output.Shape[1], (int)Output.Shape[2], (int)Output.Shape[3]);
                    dDataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)data.Shape[1], (int)data.Shape[2], (int)data.Shape[3]);
                    dOutputDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Output.Shape[1], (int)Output.Shape[2], (int)Output.Shape[3]);

                    dnn.PoolingBackward(Descriptor, ScalarOps.Conv<T>(1.0), outputDesc, output.Buffer.Ptr, dOutputDesc,
                        dOutput.Buffer.Ptr, dataDesc, data.Buffer.Ptr, ScalarOps.Conv<T>(0.0), dDataDesc, dData.Buffer.Ptr);

                    return;
                }
            }

            throw new NotImplementedException();
        }
    }
}