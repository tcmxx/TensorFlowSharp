using Learn.Mnist;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TensorFlow;

namespace TensorFlowSharp.Tests
{
    public class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            TensorFlowSharp.Windows.NativeBinding.Init();

            Console.WriteLine("TensorFlow version: " + TFCore.Version);

            while (true)
            {
                Console.Write(">>> ");
                string inp = Console.ReadLine();
                string inpLower = inp.ToLower();

                try
                {
                    switch (inpLower)
                    {
                        case "auto":
                            AutoTest();
                            break;
                        case "help":
                            Help();
                            break;
                        case "importpb":
                            ImportPb();
                            break;
                        case "rnn":
                            RNNTest();
                            break;
                        case "exit":
                            return;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                }
            }
        }

        public static void RNNTest()
        {
            RNNTester t = new RNNTester();
            t.Run();
        }

        public static void ImportPb()
        {
            OpenFileDialog ofd = new OpenFileDialog();
            if (DialogResult.OK == ofd.ShowDialog())
            {
                string path = ofd.FileName;
                Console.WriteLine($"Load Pb: {path}");
                if (File.Exists(path))
                {
                    using (var graph = new TFGraph())
                    {
                        var model = File.ReadAllBytes(path);
                        graph.Import(model);
                        foreach (TFOperation op in graph.GetEnumerator())
                        {
                            Console.WriteLine($"{op.OpType.PadRight(20)} {op.Name}");
                        }
                    }
                }
                else
                {
                    Console.WriteLine("File is not exist");
                }
            }
        }

        public static void Help()
        {
            Console.WriteLine(
                "help:    \t Show help\n" +
                "auto:    \t Auto test\n" +
                "importpb: \t Pb model import test");
        }

        public static void AutoTest()
        {
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
