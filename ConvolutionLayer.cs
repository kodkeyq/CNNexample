using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNproject
{
    internal class ConvolutionLayer
    {
        private double[,,] _convolutionTensor;
        private double[,,,] _kernelTensor;
        private double[,,] _bTensor;
        private double[,,] _ReLUTensor;

        public double[,,] OutputTensor { get { return _ReLUTensor; } }

        public int OutputWidth { get { return _convolutionTensorWidth; } }
        public int OutputHeight { get { return _convolutionTensorHeight; } }
        public int OutputDepth { get { return _convolutionTensorDepth; } }

        private int _kernelSize;
        private int _padding;
        private int _stride;

        private int _convolutionTensorWidth;
        private int _convolutionTensorHeight;
        private int _convolutionTensorDepth;

        private int _inputWidth;
        private int _inputHeight;
        private int _inputDepth;

        ConvolutionLayer(int kernelSize, int padding, int stride, int inputWidth, int inputHeight, int inputDepth, int convolutionTensorDepth)
        {
            _kernelSize = kernelSize;
            _padding = padding;
            _stride = stride;

            _inputWidth = inputWidth;
            _inputHeight = inputHeight;
            _inputDepth = inputDepth;

            _convolutionTensorDepth = convolutionTensorDepth;

            InitializeKernelTensor();
            InitializeConvolutionTensor();
            InitializeBTensor();
            InitializeReLUTensor();
        }

        private void InitializeKernelTensor()
        {
            _kernelTensor = new double[_kernelSize, _kernelSize, _inputDepth, _convolutionTensorDepth];
            for (int i = 0; i < _kernelSize; i++)
            {
                for (int j = 0; j < _kernelSize; j++)
                {
                    for (int k = 0; k < _inputDepth; k++)
                    {
                        _kernelTensor[_kernelSize, _kernelSize, _inputDepth, _convolutionTensorDepth] = 0.05;
                    }
                }
            }
        }

        private void InitializeConvolutionTensor()
        {
            double _k = _kernelSize;
            double _s = _stride;
            double _p = _padding;
            double _iW = _inputWidth;
            double _iH = _inputHeight;
            _convolutionTensorWidth = (int)Math.Floor(((_iW + 2.0d * _p - _k) / _s) + 1.0d);
            _convolutionTensorHeight = (int)Math.Floor(((_iH + 2.0d * _p - _k) / _s) + 1.0d);
            _convolutionTensor = new double[_convolutionTensorWidth, _convolutionTensorHeight, _convolutionTensorDepth];
        }

        private void InitializeBTensor()
        {
            _bTensor = new double[_convolutionTensorWidth, _convolutionTensorHeight, _convolutionTensorDepth];
        }

        private void InitializeReLUTensor()
        {
            _ReLUTensor = new double[_convolutionTensorWidth, _convolutionTensorHeight, _convolutionTensorDepth];
        }

        // well we simply multiply kernel elements on exact shifted elements of data and then take SUM of it. Why? I DONT KNOW, but that is correct one
        private double GetConvolutionValue(double[,,] data, int c_i, int c_j, int c_k)
        {
            int _d_i = c_i * _stride - _padding;
            int _d_j = c_j * _stride - _padding;

            double sum = 0.0d;
            for (int i = 0; i < _kernelSize; i++)
            {
                int _x = _d_i + i;
                for (int j = 0; j < _kernelSize; j++)
                {
                    int _y = _d_j + j;
                    for (int k = 0; k < _inputDepth; k++)
                    {
                        if ((_x >= 0) && (_y >= 0) && (_x < _inputWidth) && (_y < _inputHeight))
                        {
                            sum += _kernelTensor[i, j, k, c_k] * data[_x, _y, k];
                        }
                    }
                }
            }
            return sum;
        }

        private void GetConvolutionValues(double[,,] data)
        {
            for (int i = 0; i < _convolutionTensorWidth; i++)
            {
                for (int j = 0; j < _convolutionTensorHeight; j++)
                {
                    for (int k = 0; k < _convolutionTensorDepth; k++)
                    {
                        _convolutionTensor[i, j, k] = GetConvolutionValue(data, i, j, k);
                    }
                }
            }
        }

        private double LeakedReLU(double x)
        {
            double y = 0.0d;

            if (x >= 0) { y = x; }
            else { y = 0.01d * x; }

            return y;
        }
        private void GetReLUValues()
        {
            for (int i = 0; i < _convolutionTensorWidth; i++)
            {
                for (int j = 0; j < _convolutionTensorHeight; j++)
                {
                    for (int k = 0; k <_convolutionTensorDepth; k++)
                    {
                        _ReLUTensor[i, j, k] = LeakedReLU(_convolutionTensor[i, j, k] + _bTensor[i, j, k]);
                    }
                }
            }
        }



        public void ForwardPropagation(double[,,] data)
        {
            GetConvolutionValues(data);
            GetReLUValues();
        }

        public void BackPropagation()
        {

        }

    }
}
