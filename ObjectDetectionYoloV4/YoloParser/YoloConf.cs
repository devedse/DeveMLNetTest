using System.Linq;

namespace ObjectDetectionYoloV4.YoloParser
{
    public class YoloConf
    {
        public int MatchingBoxNumber { get; }
        public float[] Confs { get; }
        public float MaxConf { get; }
        public int MaxConfIndex { get; }

        public YoloConf(int matchingBoxNumber, float[] confs)
        {
            MatchingBoxNumber = matchingBoxNumber;
            Confs = confs;
            var temp = Confs.Select((v, i) => new { Value = v, Index = i }).OrderByDescending(t => t.Value).First();
            MaxConf = temp.Value;
            MaxConfIndex = temp.Index;
        }

        public override string ToString()
        {
            return $"{MatchingBoxNumber}: {{ MaxConfIndex: {MaxConfIndex}, MaxConf: {MaxConf} }}";
        }
    }
}
