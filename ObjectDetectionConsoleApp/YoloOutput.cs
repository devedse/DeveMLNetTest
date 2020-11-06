using ObjectDetection.YoloParser;
using System.Collections.Generic;

namespace ObjectDetection
{
    public class YoloOutput
    {
        public string ImagePath { get; }
        public IList<YoloBoundingBox> Boxes { get; }

        public YoloOutput(string imagePath, IList<YoloBoundingBox> boxes)
        {
            ImagePath = imagePath;
            Boxes = boxes;
        }
    }
}
