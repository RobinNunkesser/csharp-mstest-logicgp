namespace logicGP.Tests.Util.ML;

public class Metrics
{
    public bool IsBinaryClassification { get; set; } = false;
    public bool IsMulticlassClassification { get; set; } = false;
    public double MacroAccuracy { get; set; }
    public double Accuracy { get; set; }
    public double AreaUnderRocCurve { get; set; }
    public double F1Score { get; set; }
    public double AreaUnderPrecisionRecallCurve { get; set; }
}