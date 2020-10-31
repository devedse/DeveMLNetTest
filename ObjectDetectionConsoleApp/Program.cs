using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Microsoft.ML;
using ObjectDetection.YoloParser;
using ObjectDetection.DataStructures;
using Microsoft.ML.Data;
using System.Numerics;

namespace ObjectDetection
{
    class Program
    {
        public static void Main()
        {
            var assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            //var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
            var modelFilePath = Path.Combine(assetsPath, "Model", "yolov4_-1_3_416_416_dynamic.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            // Initialize MLContext
            MLContext mlContext = new MLContext();

            try
            {
                // Load Data
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                // Create instance of model scorer
                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

                // Use model to score data
                var modelOutput = modelScorer.Score(imageDataView);

                // Post-process model output
                YoloOutputParser parser = new YoloOutputParser();



                var boxes = modelOutput.GetColumn<float[]>("boxes").ToList();
                var confs = modelOutput.GetColumn<float[]>("confs").ToList();



                if (boxes.Count != confs.Count)
                {
                    throw new InvalidOperationException("Errrorrrr");
                }

                List<IList<YoloBoundingBox>> outputBoxes = new List<IList<YoloBoundingBox>>();
                //Boxes.Count == images.count because the model can handle multiple images at once
                for (int i = 0; i < boxes.Count; i++)
                {
                    var boxesHere = boxes[i];
                    var confsHere = confs[i];

                    var parsedBoxes = parser.ParseOutputs(boxesHere, confsHere);
                    parsedBoxes = parser.FilterBoundingBoxes(parsedBoxes, 5, 0.5f);
                    outputBoxes.Add(parsedBoxes);
                }

                // Draw bounding boxes for detected objects in each of the images
                for (var i = 0; i < images.Count(); i++)
                {
                    string imageFileName = images.ElementAt(i).Label;
                    IList<YoloBoundingBox> detectedObjects = outputBoxes.ElementAt(i);

                    DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

                    LogDetectedObjects(imageFileName, detectedObjects);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("========= End of Process..Hit any Key ========");
            Console.ReadLine();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                //// Get Bounding Box Dimensions
                //var x = (uint)Math.Max(box.Dimensions.X * OnnxModelScorer.ImageNetSettings.imageWidth, 0);
                //var y = (uint)Math.Max(box.Dimensions.Y * OnnxModelScorer.ImageNetSettings.imageWidth, 0);
                //var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width * OnnxModelScorer.ImageNetSettings.imageWidth);
                //var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height * OnnxModelScorer.ImageNetSettings.imageWidth);

                //// Resize To Image
                //x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                //y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                //width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                //height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;


                var x = (int)(originalImageWidth * box.Dimensions.X);
                var y = (int)(originalImageHeight * box.Dimensions.Y);
                var width = (int)(originalImageWidth * box.Dimensions.Width);
                var height = (int)(originalImageHeight * box.Dimensions.Height);

                // Bounding Box Text
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    // Draw text on image 
                    //thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    //thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y + height + 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, new Point(atPoint.X, atPoint.Y + (int)size.Height + 1 + (int)height + 1));

                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }

            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }

            image.Save(Path.Combine(outputImageLocation, imageName));
        }

        private static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");
        }
    }
}



