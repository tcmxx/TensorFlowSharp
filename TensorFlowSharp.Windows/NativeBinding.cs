using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowSharp.Windows
{
    public class NativeBinding : TensorFlow.NativeBinding
    {
        protected override Print InternalPrintFunc { get; set; }

        public bool IsGpu { get; private set; }

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetDllDirectory(string lpPathName);

        private NativeBinding(bool isGpu = false)
        {
            InternalPrintFunc = new Print((string s) => { Console.WriteLine(s); });

            IsGpu = isGpu;
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            if (isGpu)
            {
                SetDllDirectory(Path.Combine(baseDir, "gpu"));
            }
            else
            {
                SetDllDirectory(Path.Combine(baseDir, "cpu"));
            }

            var version = TensorFlow.TFCore.Version;
        }

        public static void Init(bool isGpu = false)
        {
            Current = new NativeBinding(isGpu);
        }

        protected override unsafe void InternalMemoryCopy(void* source, void* destination, long destinationSizeInBytes, long sourceBytesToCopy)
        {
            Buffer.MemoryCopy(source, destination, destinationSizeInBytes, sourceBytesToCopy);
        }
    }
}
