using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.cuDNN;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Convolution2D<T> : Differentiable
    {
        public Convolution2D(Variable<T> data, int kernelH, int kernelW, int numFilter)
        {
            Util.EnsureTrue(data.Shape.Rank == 4);
            Util.EnsureTrue(data.Shape[1] > 0);
            Util.EnsureTrue(data.Shape[2] > 0);
            Util.EnsureTrue(data.Shape[3] > 0);

            var numInputFilter = data.Shape[1];
            var numOutputFilter = numFilter;
            var height = data.Shape[2];
            var width = data.Shape[3];

            // fixed padding and stride now
            ConvolutionDesc = new ConvolutionDescriptor();
            ConvolutionDesc.Set2D(0, 0, 1, 1, 1, 1, ConvolutionMode.CROSS_CORRELATION);

            using (var dataDesc = new TensorDescriptor())
            using (var weightDesc = new FilterDescriptor())
            {
                var dataType = Dnn.DataTypeOf<T>();
                var tempN = 100; // for temp mini batch size
                dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, tempN, (int)numInputFilter, (int)height, (int)width);
                weightDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, numOutputFilter, (int)numInputFilter, kernelH, kernelW);

                // get output dimension
                int n, c, h, w;
                ConvolutionDesc.Get2DForwardOutputDim(dataDesc, weightDesc, out n, out c, out h, out w);

                //Console.WriteLine($"{c},{h},{w}");

                // Create variables
                var scale = Sqrt(3.0.AsScalar<T>() / ((double)(numInputFilter * kernelH * kernelW)).AsScalar<T>());

                Data = data;
                Weight = Parameter(scale * (2.0.AsScalar<T>() * RandomUniform<T>(Shape.Create(numOutputFilter, numInputFilter, kernelH, kernelW), 0UL, 0UL) - 1.0.AsScalar<T>()));
                Bias = Parameter(Fill(Shape.Create(c), ScalarOps.Conv<T>(0.1)));
                Output = Variable<T>(PartialShape.Create(-1, c, h, w));
                Workspace1 = AuxVariable<byte>();
                Workspace2 = AuxVariable<byte>();

                AddInput(Data);
                AddInput(Weight);
                AddInput(Bias);
                AddOutput(Output);
                AddAuxVar(Workspace1);
                AddAuxVar(Workspace2);
            }
        }

        public ConvolutionDescriptor ConvolutionDesc { get; }

        public Variable<T> Data { get; }

        public Variable<T> Weight { get; }

        public Variable<T> Bias { get; }

        public Variable<T> Output { get; }

        public Variable<byte> Workspace1 { get; }

        public Variable<byte> Workspace2 { get; }

        public override void Forward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var weight = executor.GetTensor(Weight);
            var bias = executor.GetTensor(Bias);
            var output = executor.GetTensor(Output, Shape.Create(data.Shape[0], Output.Shape[1], Output.Shape[2], Output.Shape[3]));

            if (executor.Context.Type == ContextType.Gpu)
            {
                var convDesc = ConvolutionDesc;
                var dnn = executor.Context.ToGpuContext().Dnn;

                using (var dataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var weightDescRcpt = executor.FilterDescRepo.Acquire())
                using (var biasDescRcpt = executor.TensorDescRepo.Acquire())
                using (var outputDescRcpt = executor.TensorDescRepo.Acquire())
                {
                    var dataDesc = dataDescRcpt.Value;
                    var weightDesc = weightDescRcpt.Value;
                    var biasDesc = biasDescRcpt.Value;
                    var outputDesc = outputDescRcpt.Value;
                    var dataType = Dnn.DataTypeOf<T>();

                    dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Data.Shape[1], (int)Data.Shape[2], (int)Data.Shape[3]);
                    weightDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)weight.Shape[0], (int)weight.Shape[1], (int)weight.Shape[2], (int)weight.Shape[3]);
                    biasDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, 1, (int)output.Shape[1], 1, 1);
                    outputDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)output.Shape[0], (int)output.Shape[1], (int)output.Shape[2], (int)output.Shape[3]);

                    ConvolutionFwdAlgo algo;
                    IntPtr workspaceSize;
                    dnn.GetConvolutionForwardAlgorithm(dataDesc, weightDesc, convDesc, outputDesc,
                        ConvolutionFwdPreference.PREFER_FASTEST, IntPtr.Zero, out algo);
                    dnn.GetConvolutionForwardWorkspaceSize(dataDesc, weightDesc, convDesc, outputDesc, algo, out workspaceSize);
                    var workspace = workspaceSize.ToInt64() > 0L
                        ? executor.GetTensor(Workspace1, Shape.Create(workspaceSize.ToInt64()))
                        : null;
                    //Console.WriteLine($"==> {algo} {workspaceSize}");

                    // step 1, convolute
                    dnn.ConvolutionForward(ScalarOps.Conv<T>(1.0), dataDesc, data.Buffer.Ptr, weightDesc, weight.Buffer.Ptr,
                        convDesc, algo, workspace?.Buffer.Ptr ?? new deviceptr<byte>(), workspaceSize, ScalarOps.Conv<T>(0.0), outputDesc, output.Buffer.Ptr);

                    // step 2, add bias
                    dnn.AddTensor(ScalarOps.Conv<T>(1.0), biasDesc, bias.Buffer.Ptr, ScalarOps.Conv<T>(1.0), outputDesc, output.Buffer.Ptr);
                    return;
                }
            }

            throw new NotImplementedException();
        }

        public override void Backward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var weight = executor.GetTensor(Weight);
            var dOutput = executor.GetGradient(Output);
            var dWeight = executor.GetGradient(Weight, Shape.Create(Weight.Shape.AsArray));
            var dBias = executor.GetGradient(Bias, Shape.Create(Bias.Shape.AsArray));
            var dData = executor.GetGradient(Data, Shape.Create(data.Shape.AsArray));

            if (executor.Context.Type == ContextType.Gpu)
            {
                var convDesc = ConvolutionDesc;
                var dnn = executor.Context.ToGpuContext().Dnn;

                using (var dataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var weightDescRcpt = executor.FilterDescRepo.Acquire())
                using (var dDataDescRcpt = executor.TensorDescRepo.Acquire())
                using (var dOutputDescRcpt = executor.TensorDescRepo.Acquire())
                using (var dBiasDescRcpt = executor.TensorDescRepo.Acquire())
                using (var dWeightDescRcpt = executor.FilterDescRepo.Acquire())
                {
                    var dataDesc = dataDescRcpt.Value;
                    var weightDesc = weightDescRcpt.Value;
                    var dDataDesc = dDataDescRcpt.Value;
                    var dOutputDesc = dOutputDescRcpt.Value;
                    var dBiasDesc = dBiasDescRcpt.Value;
                    var dWeightDesc = dWeightDescRcpt.Value;
                    var dataType = Dnn.DataTypeOf<T>();

                    dataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Data.Shape[1], (int)Data.Shape[2], (int)Data.Shape[3]);
                    dDataDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)data.Shape[0], (int)Data.Shape[1], (int)Data.Shape[2], (int)Data.Shape[3]);
                    dOutputDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)dOutput.Shape[0], (int)dOutput.Shape[1], (int)dOutput.Shape[2], (int)dOutput.Shape[3]);
                    dBiasDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, 1, (int)dOutput.Shape[1], 1, 1);
                    dWeightDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)weight.Shape[0], (int)weight.Shape[1], (int)weight.Shape[2], (int)weight.Shape[3]);
                    weightDesc.Set4D(dataType, TensorFormat.CUDNN_TENSOR_NCHW, (int)weight.Shape[0], (int)weight.Shape[1], (int)weight.Shape[2], (int)weight.Shape[3]);

                    ConvolutionBwdFilterAlgo filterAlgo;
                    IntPtr filterWorkspaceSize;
                    dnn.GetConvolutionBackwardFilterAlgorithm(dataDesc, dOutputDesc, convDesc, dWeightDesc,
                        ConvolutionBwdFilterPreference.PREFER_FASTEST, IntPtr.Zero, out filterAlgo);
                    dnn.GetConvolutionBackwardFilterWorkspaceSize(dataDesc, dOutputDesc, convDesc, dWeightDesc, filterAlgo, out filterWorkspaceSize);
                    var filterWorkspace = filterWorkspaceSize.ToInt64() > 0L
                        ? executor.GetTensor(Workspace1, Shape.Create(filterWorkspaceSize.ToInt64()))
                        : null;
                    //Console.WriteLine($"==> {filterAlgo} {filterWorkspaceSize}");

                    ConvolutionBwdDataAlgo dataAlgo;
                    IntPtr dataWorkspaceSize;
                    dnn.GetConvolutionBackwardDataAlgorithm(weightDesc, dOutputDesc, convDesc, dDataDesc,
                        ConvolutionBwdDataPreference.PREFER_FASTEST, IntPtr.Zero, out dataAlgo);
                    dnn.GetConvolutionBackwardDataWorkspaceSize(dWeightDesc, dOutputDesc, convDesc, dDataDesc, dataAlgo, out dataWorkspaceSize);
                    var dataWorkspace = dataWorkspaceSize.ToInt64() > 0L
                        ? executor.GetTensor(Workspace2, Shape.Create(dataWorkspaceSize.ToInt64()))
                        : null;
                    //Console.WriteLine($"==> {dataAlgo} {dataWorkspaceSize}");

                    // filter
                    dnn.ConvolutionBackwardFilter(ScalarOps.Conv<T>(1.0), dataDesc, data.Buffer.Ptr, dOutputDesc,
                        dOutput.Buffer.Ptr, convDesc, filterAlgo, filterWorkspace?.Buffer.Ptr ?? new deviceptr<byte>(), filterWorkspaceSize,
                        ScalarOps.Conv<T>(0.0), dWeightDesc, dWeight.Buffer.Ptr);

                    // data
                    dnn.ConvolutionBackwardData(ScalarOps.Conv<T>(1.0), weightDesc, weight.Buffer.Ptr, dOutputDesc,
                        dOutput.Buffer.Ptr, convDesc, dataAlgo, dataWorkspace?.Buffer.Ptr ?? new deviceptr<byte>(), dataWorkspaceSize,
                        ScalarOps.Conv<T>(0.0), dDataDesc, dData.Buffer.Ptr);

                    // bias
                    dnn.ConvolutionBackwardBias(ScalarOps.Conv<T>(1.0), dOutputDesc, dOutput.Buffer.Ptr, ScalarOps.Conv<T>(0.0), dBiasDesc, dBias.Buffer.Ptr);

                    return;
                }
            }

            throw new NotImplementedException();
        }
    }
}
