using System.Drawing;

namespace ObjectDetectionYoloV4.YoloParser
{
    public class YoloBoundingBox
    {
        public YoloBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }

        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        public Color BoxColor { get; set; }
    }
    
}