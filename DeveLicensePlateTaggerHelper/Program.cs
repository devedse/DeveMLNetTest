using DeveLicensePlateTaggerHelper.LicensePlateDetection;
using System;

namespace DeveLicensePlateTaggerHelper
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            var yolo = ObjectDetectionYoloV4.Program.CreateYolo();

            var l = new LicensePlateHelper();
            l.TagVehiclesInExistingData(yolo);
        }
    }
}
