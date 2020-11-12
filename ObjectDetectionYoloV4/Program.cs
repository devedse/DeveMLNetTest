using System;
using System.IO;

namespace ObjectDetectionYoloV4
{
    public static class Program
    {
        public const string assetsRelativePath = @"../../../../ObjectDetectionYoloV4/assets";
        public static string assetsPath = GetAbsolutePath(assetsRelativePath);

        public static void Main()
        {
            var yolo = CreateYolo();

            SimpleTest(yolo);

            Console.WriteLine("===== End of Process..Hit any Key =====");
            Console.ReadLine();
        }

        public static Yolo CreateYolo()
        {
            //var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
            var modelFilePath = Path.Combine(assetsPath, "Model", "yolov4_-1_3_416_416_dynamic.onnx");
            var fullModelFilePath = Path.GetFullPath(modelFilePath);
            var yolo = new Yolo(fullModelFilePath);

            return yolo;
        }

        private static void SimpleTest(Yolo yolo)
        {
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            try
            {
                var yoloOutput = yolo.ProcessDirectory(imagesFolder);

                var boxDrawer = new BoxDrawer();
                foreach (var yo in yoloOutput)
                {
                    boxDrawer.DrawBoxes(yo, outputFolder);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}



