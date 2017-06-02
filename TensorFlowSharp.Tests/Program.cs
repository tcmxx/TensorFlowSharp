using Learn.Mnist;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlowSharp.Tests
{
    public class Program
    {
        public static void Main(string[] args)
        {
            TensorFlowSharp.Windows.NativeBinding.Init();

            Console.WriteLine("TensorFlow version: " + TFCore.Version);

            //var b = TFCore.GetAllOpList ();

            Tester.Print("Start Tests");

            var t = new Tester();

            t.TestParametersWithIndexes();
            t.AddControlInput();
            t.TestImportGraphDef();
            t.TestSession();
            t.TestOperationOutputListSize();
            t.TestVariable();

            // Current failing test
            t.TestOutputShape();
            //t.AttributesTest ();
            //t.WhileTest();

            //var n = new Mnist ();
            //n.ReadDataSets ("/Users/miguel/Downloads", numClasses: 10);

            t.BasicConstantOps();
            t.BasicVariables();
            t.BasicMultidimensionalArray();
            t.BasicMatrix();

            t.NearestNeighbor();

            Console.Write("Tests are finished");
            Console.ReadLine();
        }
    }
}
