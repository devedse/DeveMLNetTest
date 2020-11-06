using ObjectDetectionYoloV4.YoloParser;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;

namespace ObjectDetectionYoloV4
{
    public class BoxDrawer
    {
        public void DrawBoxes(YoloOutput yo, string outputDir)
        {
            DrawBoundingBox(yo.ImagePath, outputDir, Path.GetFileName(yo.ImagePath), yo.Boxes);

            LogDetectedObjects(yo.ImagePath, yo.Boxes);
        }

        private static void DrawBoundingBox(string inputImagePath, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(inputImagePath);

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
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
                    Point atPoint = new Point(x, y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    // Draw text on image 
                    thumbnailGraphic.FillRectangle(colorBrush, x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                    //thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y + height + 1), (int)size.Width, (int)size.Height);
                    //thumbnailGraphic.DrawString(text, drawFont, fontBrush, new Point(atPoint.X, atPoint.Y + (int)size.Height + 1 + (int)height + 1));

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
