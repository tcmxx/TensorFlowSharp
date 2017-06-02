using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowSharp.Windows
{
    public class NativeBinding : TensorFlow.NativeBinding
    {
        protected override Print InternalPrintFunc { get; set; }

        private NativeBinding()
        {
            InternalPrintFunc = new Print((string s) => { Console.WriteLine(s); });
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
