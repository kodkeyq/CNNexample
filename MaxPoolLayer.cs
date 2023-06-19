using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;

namespace CNNproject
{
    internal class MaxPoolLayer
    {

        private int _kernelSize;
        private int _inputWidth;
        private int _inputHeight;
        private int _inputDepth;

        private int _outputWidth;
        private int _outputHeight;
        private int _outputDepth;

        private double[,,] _maxPoolTensor;
        MaxPoolLayer(int kernelSize, int inputWidth, int inputHeight, int inputDepth)
        {
            _kernelSize = kernelSize;
            _inputWidth = inputWidth;
            _inputHeight = inputHeight;
            _inputDepth = inputDepth;

            InitializeMaxPoolTensor();
        }

        private void InitializeMaxPoolTensor()
        {
            int _inc = 0;
            if (_inputWidth % _kernelSize > 0) { _inc = 1; }
            _outputWidth = _inputWidth / _kernelSize + _inc;

            _inc = 0;
            if (_inputHeight % _kernelSize > 0) { _inc = 1; }
            _outputHeight = _inputHeight / _kernelSize + _inc;

            _outputDepth = _inputDepth;
            _maxPoolTensor = new double[_outputWidth, _outputHeight, _outputDepth];
        }

        public int OutputWidth { get { return _outputWidth; } }
        public int OutputHeight { get { return _outputHeight; } }
        public int OutputDepth { get { return _outputDepth; } }

        public void ForwardPropagation(double[,,] data)
        {
            GetMaxPoolTensorValues(data);
        }

        public double[,,] MaxPoolTensor { get { return _maxPoolTensor; } }

        private double GetMaxValue(double[,,] data, int i, int j, int k )
        {
            double max = 0.0d;

            for (int x = 0; x < _kernelSize; x++)
            {
                int a = i * _kernelSize + x;
                for (int y = 0; y < _kernelSize; y++)
                {
                    int b = j * _kernelSize + y;

                    if ((a < _inputWidth) && (b <_inputHeight))
                    {
                        max = Math.Max(max, data[a, b, k]);
                    }
                }
            }
            return max;
        }
        private void GetMaxPoolTensorValues(double[,,] data)
        {
            for (int i = 0; i < _outputWidth; i++) 
            { 
                for (int j = 0; j < _outputHeight; j++)
                {
                    for (int k = 0; k < _outputDepth; k++)
                    {
                        _maxPoolTensor[i, j, k] = GetMaxValue(data, i, j, k);
                    }
                }    
            }
        }



    }
}
