using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlowSharp.UWP
{
    public class NativeBinding : TensorFlow.NativeBinding
    {
        protected override Print InternalPrintFunc { get; set; }

        private NativeBinding()
        {
            InternalPrintFunc = new Print((s) => { Debug.WriteLine(s); });
        }

        public static void Init()
        {
            Current = new NativeBinding();

            try
            {
                Log("Tf Version: " + TensorFlow.TFCore.Version);
            }
            catch (Exception ex)
            {
                throw new DllNotFoundException("Add libtensorflow.dll to your application!", ex);
            }
        }

        protected override unsafe void InternalMemoryCopy(void* source, void* destination, long destinationSizeInBytes, long sourceBytesToCopy)
        {
            Buffer.MemoryCopy(source, destination, destinationSizeInBytes, sourceBytesToCopy);
        }
    }
}
