using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    public class OutputYoloV4
    {
        [VectorType(4)]
        public float[] b1 { get; set; }
    }
}
