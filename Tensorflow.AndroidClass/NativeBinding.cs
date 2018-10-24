using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;
using Droid = Android;

namespace TensorFlowSharp.Android
{
    public class NativeBinding : TensorFlow.NativeBinding
    {
        private NativeBinding()
        {
            InternalPrintFunc = new Print((s) => { Droid.Util.Log.Debug("TensorFlowSharp", s); });
        }

        protected override Print InternalPrintFunc { get; set; }

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
