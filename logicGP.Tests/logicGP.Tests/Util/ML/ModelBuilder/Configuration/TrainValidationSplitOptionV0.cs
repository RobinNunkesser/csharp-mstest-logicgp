namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class TrainValidationSplitOptionV0 : MBConfig, IValidationOption
{
    public override int Version => 0;
    public override string Type => "TrainValidateSplitValidationOption";
    public float? SplitRatio { get; set; }
}