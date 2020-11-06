namespace ObjectDetectionYoloV4.YoloParser
{
    public class YoloBoxDimensions
    {
        public int BoxNumber { get; }

        public float X { get; }
        public float Y { get; }
        public float Height { get; }
        public float Width { get; }

        public float[] OriginalStuff { get; }

        public YoloBoxDimensions(int boxNumber, float[] originalStuff)
        {
            BoxNumber = boxNumber;
            OriginalStuff = originalStuff;

            X = originalStuff[0];
            Y = originalStuff[1];
            Width = originalStuff[2] - originalStuff[0];
            Height = originalStuff[3] - originalStuff[1];

            //X = boxes[i + 0] - boxes[i + 2] / 2f,
            //Y = boxes[i + 1] - boxes[i + 3] / 2f,
            //Width = boxes[i + 0] + boxes[i + 2] / 2f,
            //Height = boxes[i + 1] + boxes[i + 3] / 2f,
        }

        public override string ToString()
        {
            return $"{BoxNumber}: {{ OriginalStuff: ({string.Join(",", OriginalStuff)}) }}";
        }
    }
}
