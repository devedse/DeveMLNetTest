using System.Collections.Generic;

namespace DeveLicensePlateTaggerHelper.LicensePlateDetection
{
    public class Size
    {
        public int width { get; set; }
        public int height { get; set; }
    }

    public class Asset
    {
        public string format { get; set; }
        public string id { get; set; }
        public string name { get; set; }
        public string path { get; set; }
        public Size size { get; set; }
        public int state { get; set; }
        public int type { get; set; }
    }

    public class BoundingBox
    {
        public double height { get; set; }
        public double width { get; set; }
        public double left { get; set; }
        public double top { get; set; }
    }

    public class Point
    {
        public double x { get; set; }
        public double y { get; set; }
    }

    public class Region
    {
        public string id { get; set; }
        public string type { get; set; }
        public List<string> tags { get; set; }
        public BoundingBox boundingBox { get; set; }
        public List<Point> points { get; set; }
    }

    public class VottJsonAsset
    {
        public Asset asset { get; set; }
        public List<Region> regions { get; set; }
        public string version { get; set; }
    }
}
