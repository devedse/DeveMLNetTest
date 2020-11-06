using System;
using System.IO;

namespace ObjectDetection
{
    public static class Program
    {
        public static void Main()
        {
            var assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            //var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
            var modelFilePath = Path.Combine(assetsPath, "Model", "yolov4_-1_3_416_416_dynamic.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            try
            {
                var yolo = new Yolo(modelFilePath);
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

            Console.WriteLine("===== End of Process..Hit any Key =====");
            Console.ReadLine();
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



