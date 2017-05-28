using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorFlowSharp.Android
{
    public class NativeBinding : TensorFlow.NativeBinding
    {
        private NativeBinding()
        {

        }

        public static void Init()
        {
            Current = new NativeBinding();
        }

        protected override unsafe void InternalMemoryCopy(void* source, void* destination, long destinationSizeInBytes, long sourceBytesToCopy)
        {
            Buffer.MemoryCopy(source, destination, destinationSizeInBytes, sourceBytesToCopy);
        }
    }
}
