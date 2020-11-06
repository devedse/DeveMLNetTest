using ObjectDetectionYoloV4.YoloParser;
using System.Collections.Generic;

namespace ObjectDetectionYoloV4
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
