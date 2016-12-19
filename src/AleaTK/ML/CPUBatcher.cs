using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AleaTK.ML
{
    /// <summary>
    /// CPUBatcher allocates training/testing data on the GPU memory when it is needed/necessary. 
    /// At initialization data is allocated on the heap and when it is needed by the user of the API then allocation on the GPU is made.
    /// Thus, we prevent flooding the GPU by allocating allocating the whole training/test data on it.
    /// </summary>
    public class CPUBatcher : IDisposable
    {
        public long Index { get; private set; }
        public float[,] Data { get; }
        public float[,] Labels { get; }
        public Context Context { get; }
        public Tensor<float> DataTensor { get; private set; }
        public Tensor<float> LabelsTensor { get; private set; }

        private readonly bool reset;
        private readonly int inputLayers;
        private readonly int rows;
        private readonly int columns;
        private readonly int outputsColumns;
        private readonly Random random;
        private int[] indices;
        /// <summary>
        /// Designated initializer of CPU batcher
        /// </summary>
        /// <param name="context">Contex</param>
        /// <param name="data">Flattend input data: each row represents a data instance flattend by columns and number of input layers</param>
        /// <param name="labels"></param>
        /// <param name="doReset">Indicates that the batcher should start over splitting up input data to batches</param>
        /// <param name="mirroring">Indicates if the input data should be mirrored horizontally</param>
        /// <param name="numberOfInputLayers">Specifies how many layers contained in one row of the data.</param>
        public CPUBatcher(Context context, float[,] data, float[,] labels, bool doReset = true, bool mirroring = false, int numberOfInputLayers = 3)
        {
            Context = context;
            random = new Random(0);
            inputLayers = numberOfInputLayers;
            if (mirroring)
            {
                Data = mirrorData(data);
                Labels = mirrorLabels(labels);
            } else
            {
                Data = data;
                Labels = labels;
            }
            reset = doReset;
            rows = Data.GetLength(0);
            columns = Data.GetLength(1);
            outputsColumns = Labels.GetLength(1);

            indices = Enumerable.Range(0, rows).ToArray();
            if (!reset) return;
            Index = -1;
            Reset();
        }

        private float[,] mirrorData(float[,] data)
        {
            float[,] result = new float[data.GetLength(0) * 2, data.GetLength(1)];
            double channelLength = data.GetLength(1) / inputLayers;
            double rowLength = Math.Sqrt(channelLength);

            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    result[2 * i, j] = data[i, j];
                    int currentChannel = (int)Math.Floor(j / channelLength);

                    int currentRow = (int)Math.Floor((j - currentChannel * (int)channelLength) / rowLength);
                    int currentColumn = j - currentChannel * (int)channelLength - currentRow * (int)rowLength;
                    result[2 * i + 1, currentChannel * (int)channelLength + (currentRow + 1) * (int)rowLength - 1 - currentColumn] = data[i, j];
                }
            }
            return result;
        }

        private float[,] mirrorLabels(float[,] labels)
        {
            float[,] result = new float[labels.GetLength(0) * 2, labels.GetLength(1)];
            for (int i = 0; i < labels.GetLength(0); i++)
            {
                for (int j = 0; j < labels.GetLength(1); j++)
                {
                    result[2 * i, j] = labels[i, j];
                    result[2 * i + 1, j] = labels[i, j];
                }
            }
            return result;
        }

        public bool Next(long batchSize, Executor executor, Variable<float> dataVar, Variable<float> labelsVar)
        {
            if (Index >= rows)
            {
                Reset();
                return false;
            }

            var size = Index + batchSize >= rows ? rows - Index : batchSize;
            float[,] data = new float[size, columns];
            float[,] labels = new float[size, outputsColumns];

            for (long i = Index; i < Index + size; i++)
            {
                int dataIndex = indices[i];
                long iT = i - Index;
                for (long j = 0; j < columns; j++)
                {
                    data[iT, j] = Data[dataIndex, j];
                    if (j < outputsColumns)
                    {
                        labels[iT, j] = Labels[dataIndex, j];
                    }
                }
            }
            Index += batchSize;

            DataTensor = executor.Context.Allocate(data);
            LabelsTensor = executor.Context.Allocate(labels);

            executor.SetTensor(dataVar, DataTensor);
            executor.SetTensor(labelsVar, LabelsTensor);
            return true;
        }

        public void Dispose()
        {
            DataTensor.Dispose();
            LabelsTensor.Dispose();
        }

        public void Reset()
        {
            if (!reset || Index == 0L || Context.Type != ContextType.Gpu) return;
            Index = 0L;
            ShuffleIndices();
        }

        private void ShuffleIndices()
        {
            var rng = random;
            var array = indices;
            var n = array.Length;
            while (n > 1)
            {
                var k = rng.Next(n--);
                var temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
